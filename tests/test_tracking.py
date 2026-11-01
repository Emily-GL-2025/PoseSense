from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.engine.results import Results

from workout.tracking import select_tracker, tracked_people


def make_result(pose_factory, boxes, ids=False):
    poses = (
        np.stack([pose_factory().pose for _ in boxes])
        if len(boxes)
        else np.empty((0, 17, 3), np.float32)
    )
    poses[:, :, 0] *= 1280
    poses[:, :, 1] *= 720
    return Results(
        np.zeros((720, 1280, 3), np.uint8),
        "camera.jpg",
        {0: "person"},
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7 if ids else 6),
        keypoints=torch.from_numpy(poses),
    )


def test_actual_results_ids_alignment_and_normalization(pose_factory):
    result = make_result(
        pose_factory, [[1, 2, 3, 4, 9, 0.9, 0], [5, 6, 7, 8, 4, 0.8, 0]], ids=True
    )
    people = tracked_people(result)
    assert [p.track_id for p in people] == [9, 4]
    np.testing.assert_allclose(people[0].pose, pose_factory().pose, atol=1e-6)
    assert [p.track_id for p in tracked_people(result, 4)] == [4]
    assert [p.track_id for p in tracked_people(result, 99)] == [9, 4]
    assert tracked_people(make_result(pose_factory, [[1, 2, 3, 4, 0.9, 0]])) == []


def test_installed_fasttracker_persists_and_never_returns_stale_pose(
    pose_factory, tmp_path
):
    from ultralytics.trackers.track import on_predict_postprocess_end, on_predict_start

    tracker_path = select_tracker()
    assert Path(tracker_path).name == "fasttrack.yaml"
    assert Path(select_tracker("bytetrack")).name == "bytetrack.yaml"
    predictor = SimpleNamespace(
        args=SimpleNamespace(task="pose", tracker=tracker_path),
        dataset=SimpleNamespace(bs=1, mode="image"),
        device=torch.device("cpu"),
        save_dir=tmp_path,
    )
    on_predict_start(predictor, persist=True)
    first_id = None
    for index in range(2):
        predictor.results = [
            make_result(pose_factory, [[100 + index, 100, 300 + index, 600, 0.9, 0]])
        ]
        on_predict_postprocess_end(predictor, persist=True)
        people = tracked_people(predictor.results[0])
        assert len(people) == 1
        if first_id is None:
            first_id = people[0].track_id
        assert people[0].track_id == first_id
    predictor.results = [make_result(pose_factory, [])]
    on_predict_postprocess_end(predictor, persist=True)
    assert tracked_people(predictor.results[0]) == []
