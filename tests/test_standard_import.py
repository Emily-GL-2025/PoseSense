"""Importer tests use synthetic poses and mocked predictions, never a camera."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict

import cv2
import numpy as np
import pytest

import standard_import as importer
from sessions.playback import Recording
from workout.analysis.jumping_jack import JumpingJackConfig, JumpingJackCounter
from workout.packets import TrackedPerson


def test_import_and_help_do_not_load_inference():
    script = (
        "import sys; import standard_import; "
        "assert not any(x in sys.modules for x in ('torch', 'cv2', 'ultralytics')); "
        "standard_import.parse_args(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--labels" in result.stdout


@pytest.mark.parametrize(
    "milliseconds,fps,override,expected,clock",
    [
        ([400, 433, 480], 30, None, [0, 0.033, 0.080], "decoder_POS_MSEC"),
        ([0, 0, 0], 25, None, [0, 0.04, 0.08], "fallback_metadata_fps"),
        ([0, 100, 50], 20, None, [0, 0.05, 0.10], "fallback_metadata_fps"),
        ([0, np.nan, 50], 20, None, [0, 0.05, 0.10], "fallback_metadata_fps"),
        ([0, 500, 1000], 25, 10, [0, 0.1, 0.2], "forced_constant_fps"),
        ([0], float("nan"), None, [0], "decoder_POS_MSEC"),
    ],
)
def test_clocks(milliseconds, fps, override, expected, clock):
    times, source = importer.video_times(milliseconds, fps, override)
    np.testing.assert_allclose(times, expected)
    assert source == clock


def test_invalid_clocks():
    with pytest.raises(ValueError, match="no decoded"):
        importer.video_times([], 30)
    with pytest.raises(ValueError, match="provide --fps"):
        importer.video_times([0, 0], float("nan"))


def test_measure_matches_actual_classifier(pose_factory):
    cfg = JumpingJackConfig()
    rng = np.random.default_rng(42)
    poses = [pose_factory(p).pose for p in ("open", "closed", "jitter")]
    for _ in range(500):
        pose = rng.uniform(0, 1, (17, 3)).astype(np.float32)
        pose[:, 2] = 0.99
        poses.append(pose)
    for pose in poses:
        metrics, gates, label = importer.measure(pose, cfg)
        actual = JumpingJackCounter(cfg).classify(pose)
        assert (actual.value if actual else None) == (
            label if label in ("OPEN", "CLOSED") else None
        )
        if label in ("OPEN", "CLOSED"):
            assert np.isfinite(metrics).all()
            assert gates[9 if label == "OPEN" else 8] == 1


def test_exact_boundaries_and_masking(pose_factory):
    # Binary-representable geometry makes wrist equality exact.
    pose = pose_factory().pose
    pose[5, :2], pose[6, :2] = (0.25, 0.25), (0.5, 0.25)
    pose[15, :2], pose[16, :2] = (0.25, 0.75), (0.75, 0.75)
    pose[9, 1] = pose[10, 1] = 0.125
    cfg = JumpingJackConfig(wrist_margin=0.25, open_ankle_ratio=2)
    assert importer.measure(pose, cfg)[2] == "DEAD_BAND"  # equality is not wrist pass
    pose[9, 1] = pose[10, 1] = 0
    assert importer.measure(pose, cfg)[2] == "OPEN"  # foot equality passes
    pose[9, 2] = 0.49
    raw = pose.copy()
    arrays, _ = importer.analyze([pose], np.array([0.0]), [7], [], cfg)
    assert np.isnan(arrays["poses"][0, 9, :2]).all()
    np.testing.assert_array_equal(arrays["raw_poses"][0], raw)
    assert arrays["gates"][0, 0] == 0 and arrays["gates"][0, 9] == -1


def test_sequence_labels_and_denominators(pose_factory):
    closed, opened = (pose_factory(p).pose for p in ("closed", "open"))
    invalid = np.full((17, 3), np.nan, np.float32)
    raw = [closed, closed, opened, opened, closed, closed, invalid]
    times = np.arange(7) * 0.1
    labels = [
        {"start": 0.2, "end": 0.4, "phase": "OPEN"},
        {"start": 0.6, "end": 0.8, "phase": "OPEN"},
    ]
    cfg = JumpingJackConfig()
    arrays, runs = importer.analyze(raw, times, [7] * 6 + [-1], labels, cfg)
    assert arrays["rep_counts"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert arrays["state"][-1] == "UNKNOWN"
    assert len(runs) == 3 and all(run["meets_confirmation"] for run in runs)
    report = importer.summarize(arrays, cfg)
    assert report["reference_recognition"]["OPEN"] == {
        "labeled_frames": 3,
        "recognized_frames": 2,
        "missed_including_invalid": 1,
    }
    assert report["groups"]["reference_OPEN"]["gates"]["open_all"] == {
        "pass": 2,
        "fail": 0,
        "not_evaluable": 1,
    }
    assert report["keypoint_distributions"]["left_wrist"]["x"]["n"] == 6
    json.dumps(report, allow_nan=False)


def test_short_holds_gaps_and_identity_cannot_bridge_reps(pose_factory):
    closed, opened = (pose_factory(p).pose for p in ("closed", "open"))
    cfg = JumpingJackConfig()
    for times, ids in [
        ([0, 0.01, 0.02, 0.03, 0.04, 0.05], [7] * 6),
        ([0, 0.1, 0.2, 0.3, 1, 1.1], [7] * 6),
        ([0, 0.1, 0.2, 0.3, 0.4, 0.5], [7] * 4 + [9] * 2),
    ]:
        arrays, _ = importer.analyze(
            [closed, closed, opened, opened, closed, closed],
            np.array(times),
            ids,
            [],
            cfg,
        )
        assert arrays["rep_counts"][-1] == 0


def test_missing_and_unlabeled_are_not_standards():
    cfg = JumpingJackConfig()
    arrays, runs = importer.analyze(
        np.full((3, 17, 3), np.nan), np.arange(3) * 0.1, [-1] * 3, [], cfg
    )
    report = importer.summarize(arrays, cfg)
    assert report["missing_frames"] == 3
    assert report["groups"]["reference_OPEN"]["frames"] == 0
    assert report["groups"]["all_frames"]["distributions"]["ankle_ratio"] == {"n": 0}
    assert runs == []
    json.dumps(report, allow_nan=False)


def test_labels_validate_and_inputs_deduplicate(tmp_path):
    video = tmp_path / "demo.mp4"
    video.touch()
    labels = tmp_path / "labels.json"
    data = {"demo.mp4": [{"start": 1, "end": 2, "phase": "OPEN"}]}
    labels.write_text(json.dumps(data))
    assert importer.resolve_videos([tmp_path, video]) == [video.resolve()]
    assert importer.load_labels(labels, [video]) == data
    data["demo.mp4"].append({"start": 1.5, "end": 3, "phase": "CLOSED"})
    labels.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Overlapping"):
        importer.load_labels(labels, [video])
    with pytest.raises(ValueError, match="exactly one"):
        importer.load_labels(labels, [video, video])


def fake_video(monkeypatch, pose_factory, *, ambiguous=False, fail_inference=False):
    import ultralytics

    import workout.tracking

    calls, released = [], []
    people = [
        [pose_factory("closed")],
        [pose_factory("closed")],
        [pose_factory("open")],
        [pose_factory("open")],
        [pose_factory("closed")],
        [pose_factory("closed")],
        [TrackedPerson(99, pose_factory("open").pose)],  # never auto-switch!
    ]
    if ambiguous:
        people[0].append(TrackedPerson(99, pose_factory().pose))

    class Capture:
        def __init__(self, path):
            assert isinstance(path, str) and path.endswith(".mp4")
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_COUNT: 7.0,
                cv2.CAP_PROP_POS_MSEC: (self.index - 1) * 100.0,
            }[prop]

        def read(self):
            self.index += 1
            return self.index <= 7, np.zeros((72, 128, 3), np.uint8)

        def release(self):
            released.append(True)

    class Model:
        task = "pose"

        def __init__(self, _path):
            self.index = 0

        def track(self, **kwargs):
            calls.append(kwargs)
            if fail_inference:
                raise RuntimeError("inference failed")
            result = people[self.index]
            self.index += 1
            return [result]

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(ultralytics, "YOLO", Model)
    monkeypatch.setattr(workout.tracking, "select_tracker", lambda _: "mock.yaml")
    monkeypatch.setattr(workout.tracking, "tracked_people", lambda result: result)
    return calls, released


def test_full_cli_report_and_playback_compatibility(
    monkeypatch, tmp_path, pose_factory
):
    calls, released = fake_video(monkeypatch, pose_factory)
    video = tmp_path / "demo.mp4"
    video.touch()
    model = tmp_path / "model.pt"
    model.write_bytes(b"mock model")
    output = tmp_path / "results"
    assert (
        importer.main(
            [
                str(video),
                "--model",
                str(model),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    folder = next(output.iterdir())
    report = json.loads((folder / "report.json").read_text())
    assert report["analysis_config"] == asdict(JumpingJackConfig())
    assert report["clips"][0]["final_rep_count"] == 1
    assert report["clips"][0]["missing_frames"] == 1
    assert report["pooled"]["frames"] == 7
    recording = Recording.load(folder / "001_demo.npz")
    assert recording.camera_width == 128 and recording.camera_height == 72
    assert recording.track_ids.tolist() == [7] * 6 + [-1]
    assert len(calls) == 7 and released == [True]
    assert calls[0]["compile"] is False and calls[0]["quantize"] == 16
    assert calls[0]["imgsz"] == 640 and calls[0]["device"] == "cpu"
    with np.load(folder / "001_demo.npz", allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype.kind != "O"
    document = (folder / "report.html").read_text(encoding="utf-8")
    assert "Independent label comparison" in document
    assert "rule_OPEN" in document
    assert "<script" not in document


@pytest.mark.parametrize("failure", ["ambiguous", "inference"])
def test_capture_released_on_failure(monkeypatch, tmp_path, pose_factory, failure):
    _, released = fake_video(
        monkeypatch,
        pose_factory,
        ambiguous=failure == "ambiguous",
        fail_inference=failure == "inference",
    )
    args = importer.parse_args([str(tmp_path / "demo.mp4"), "--device", "cpu"])
    with pytest.raises((ValueError, RuntimeError)):
        importer.import_video(args.videos[0], args, [], JumpingJackConfig())
    assert released == [True]


def test_svg_and_degenerate_distribution():
    assert importer.distribution([np.nan, np.inf]) == {"n": 0}
    stats = importer.distribution([1, 1, 1])
    assert stats["std"] == 0 and stats["median"] == 1
    svg = importer.histogram_svg(stats, (-0.1, 1.8))
    root = ET.fromstring(svg)
    assert root.tag == "svg" and len(root.findall("rect")) == 24
    assert len(root.findall("path")) == 2
    nearly_constant = importer.distribution(np.diff(np.arange(10) * 0.1))
    assert sum(nearly_constant["histogram"]["counts"]) == 9
