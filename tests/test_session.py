import json

import numpy as np
import pytest

from workout.gesture import GestureDetector, right_hand_raised
from workout.packets import AnalysisStatus, PosePacket, clean_pose
from workout.recorder import SessionRecorder
from workout.session import SessionController, SessionState


def start_session(factory):
    session = SessionController(duration_seconds=180)
    for frame in range(41):
        packet = session.process_frame([factory("raised")], frame, 10 + frame * 0.05)
    assert packet is not None
    assert session.start_time == 12
    return session


def test_anatomical_right_and_confidence(pose_factory):
    assert right_hand_raised(pose_factory("closed")) is False
    person = pose_factory("closed")
    person.pose[9, 1] = 0.1  # only anatomical left wrist
    assert right_hand_raised(person) is False
    person = pose_factory("raised")
    assert right_hand_raised(person) is True
    for index in (0, 6, 8, 10):
        noisy = pose_factory("raised")
        noisy.pose[index, 2] = 0.1
        assert right_hand_raised(noisy) is None


def test_gesture_exact_timer_and_below(pose_factory):
    detector = GestureDetector()
    for i in range(61):
        assert detector.update([pose_factory("closed")], i * 0.05) is None
    for i in range(40):
        assert detector.update([pose_factory("raised", 8)], 4 + i * 0.05) is None
    assert detector.update([pose_factory("raised", 8)], 6) == 8


def test_gesture_noise_tolerance_and_no_selection_on_missing(pose_factory):
    detector = GestureDetector()
    for i in range(39):
        detector.update([pose_factory("raised")], i * 0.05)
    assert detector.update([], 1.95) is None
    assert detector.update([], 2.0) is None
    # The grace is 0.15 seconds from the last known raised posture.
    assert detector.update([pose_factory("raised")], 2.04) == 7


def test_gesture_lowering_and_long_unobserved_gap_reset(pose_factory):
    detector = GestureDetector()
    for i in range(39):
        detector.update([pose_factory("raised")], i * 0.05)
    detector.update([pose_factory("closed")], 1.95)
    assert detector.update([pose_factory("raised")], 2.0) is None
    assert detector.update([pose_factory("raised")], 5.0) is None
    assert detector.progress == 0


def test_selection_lock_missing_rows_recovery_and_frozen_reps(pose_factory):
    session = start_session(pose_factory)
    assert len(session.recorder) == 1
    session.process_frame(
        [pose_factory(), pose_factory("raised", 3)],
        41,
        12.05,
        AnalysisStatus(2, "OPEN", 40, session.generation),
    )
    assert session.selected_track_id == 7
    assert session.rep_count == 2
    old_generation = session.generation
    missing = session.process_frame(
        [pose_factory("raised", 3)],
        42,
        12.1,
        AnalysisStatus(3, "CLOSED", 41, old_generation),
    )
    assert session.state == SessionState.ACTIVE_LOST
    assert session.rep_count == 2
    assert not missing.present and np.isnan(missing.pose).all()
    assert session.recorder.track_ids[-1] == -1
    returning = session.process_frame(
        [pose_factory("open")], 43, 12.15, AnalysisStatus(4, "OPEN", 42, old_generation)
    )
    assert returning.present and session.state == SessionState.ACTIVE_TRACKED
    assert session.generation == old_generation + 2
    assert session.exercise_state == "UNKNOWN" and session.rep_count == 2
    assert len(session.recorder) == 4


def test_different_id_requires_deliberate_reacquisition(pose_factory):
    session = start_session(pose_factory)
    for i in range(1, 41):
        session.process_frame([pose_factory("closed", 3)], 40 + i, 12 + i * 0.05)
    assert session.selected_track_id == 7 and session.state == SessionState.ACTIVE_LOST
    for i in range(20):
        session.process_frame([pose_factory("raised", 3)], 81 + i, 14.1 + i * 0.05)
        assert session.selected_track_id == 7
    session.process_frame([pose_factory("raised", 3)], 101, 15.1)
    assert session.selected_track_id == 3 and session.initial_track_id == 7
    event = next(e for e in session.events if e["event"] == "reassigned_by_gesture")
    assert event["old_track_id"] == 7 and event["new_track_id"] == 3


def test_deadline_180_seconds_even_while_lost(pose_factory):
    session = start_session(pose_factory)
    session.process_frame([], 41, 191.999)
    assert session.state == SessionState.ACTIVE_LOST
    assert session.process_frame([], 42, 192) is None
    assert session.state == SessionState.FINISHED
    assert session.elapsed(999) == 180
    assert len(session.recorder) == 2
    assert session.stop_reason == "completed"
    assert session.process_frame([pose_factory()], 43, 193) is None


def test_clock_can_expire_without_another_frame(pose_factory):
    session = start_session(pose_factory)
    session.advance_time(400)
    assert session.end_time == 192 and len(session.recorder) == 1
    with pytest.raises(ValueError, match="monotonic"):
        session.advance_time(399)


def test_recorder_masks_shapes_ownership_and_monotonicity(pose_factory):
    recorder = SessionRecorder()
    pose = pose_factory().pose
    pose[9, 2] = 0.2
    pose[4, 0] = np.inf
    recorder.append(PosePacket(10, 0, 7, pose, True, 1), 0)
    recorder.append(PosePacket(11, 0.1, 7, pose, False, 2), 0)
    pose[:] = 1
    arrays = recorder.arrays()
    assert arrays["poses"].shape == (2, 17, 3)
    assert arrays["poses"].dtype == np.float32
    assert np.isnan(arrays["poses"][0, [4, 9], :2]).all()
    assert arrays["poses"][0, 9, 2] == np.float32(0.2)
    assert np.isnan(arrays["poses"][1]).all()
    assert arrays["present"].tolist() == [True, False]
    assert arrays["observed_track_ids"].tolist() == [7, -1]
    for name, array in arrays.items():
        if name != "poses":
            assert array.shape == (2,)
    with pytest.raises(ValueError, match="exactly once"):
        recorder.append(PosePacket(11, 0.2, 7, pose, True, 2), 0)
    with pytest.raises(ValueError, match="monotonic"):
        recorder.append(PosePacket(12, 0.01, 7, pose, True, 2), 0)
    assert len(recorder) == 2


@pytest.mark.parametrize("completed", [True, False])
def test_save_roundtrip_partial_and_complete(tmp_path, pose_factory, completed):
    session = start_session(pose_factory)
    session.process_frame([], 41, 12.1)
    session.finish(192 if completed else 15, "quit")
    metadata = session.metadata(
        camera_width=1280, camera_height=720, nominal_source_fps=30
    )
    saved = session.recorder.save(tmp_path, metadata)
    assert saved == session.recorder.save(tmp_path, metadata)
    with np.load(saved, allow_pickle=False) as data:
        assert data["poses"].shape == (2, 17, 3)
        assert data["completed"].item() is completed
        assert data["session_duration"].item() == (180 if completed else 3)
        assert data["target_duration"].item() == 180
        assert data["selected_track_id_at_start"].item() == 7
        assert json.loads(data["events_json"].item())[1]["event"] == "lost"
        assert json.loads(data["metadata_json"].item())["recorded_frames"] == 2
        assert all(data[name].dtype != object for name in data.files)
    assert len(list(tmp_path.iterdir())) == 1


def test_bad_pose_shape_and_empty_recorder():
    with pytest.raises(ValueError, match="17, 3"):
        clean_pose(np.zeros((16, 3)))
    assert SessionRecorder().arrays()["poses"].shape == (0, 17, 3)


def test_full_simulated_three_minutes_preserves_every_active_frame(
    tmp_path, pose_factory
):
    session = start_session(pose_factory)
    person = pose_factory()
    for index in range(1, 5400):
        people = [] if 100 <= index < 150 else [person]
        session.process_frame(people, 40 + index, 12 + index / 30)
    assert session.process_frame([person], 5440, 192) is None
    assert len(session.recorder) == 5400
    path = session.recorder.save(
        tmp_path,
        session.metadata(camera_width=1280, camera_height=720, nominal_source_fps=30),
    )
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_indices"], np.arange(40, 5440))
        assert data["poses"].shape == (5400, 17, 3)
        assert np.isnan(data["poses"][100:150]).all()
        assert data["present"].sum() == 5350
        assert np.all(np.diff(data["timestamps"]) > 0)
        assert data["timestamps"][-1] < 180
        assert data["session_duration"].item() == 180
        assert data["target_duration"].item() == 180
        assert data["completed"].item() is True


def test_shutdown_accepts_only_current_generation_tracked_results(pose_factory):
    session = start_session(pose_factory)
    session.finish(13, "quit")
    session.accept_status(
        AnalysisStatus(1, "CLOSED", 40, session.generation), final=True
    )
    assert session.rep_count == 1
    session.accept_status(
        AnalysisStatus(8, "CLOSED", 39, session.generation), final=True
    )
    assert session.rep_count == 1
    missing = start_session(pose_factory)
    missing.process_frame([], 41, 12.1)
    missing.finish(13, "quit")
    missing.accept_status(
        AnalysisStatus(8, "CLOSED", 41, missing.generation), final=True
    )
    assert missing.rep_count == 0
