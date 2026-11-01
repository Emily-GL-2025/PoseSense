"""Exercise application cleanup/save paths without needing an interactive camera."""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import ultralytics

import workout.tracking
import workout_session
from workout.gesture import Hold
from workout.packets import AnalysisStatus
from workout.session import SessionController, SessionState


@pytest.fixture(autouse=True)
def simulated_window(monkeypatch):
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args: (0, 0, 1280, 900))


@pytest.mark.parametrize("view", ["video", "skeleton"])
@pytest.mark.parametrize(
    "failure",
    [
        "quit",
        "read",
        "inference",
        "keyboard",
        "camera_open",
        "q",
        "window_close",
        "completed",
        "toggle",
        "completed_toggle",
        "completed_read",
        "completed_inference",
        "completed_keyboard",
        "completed_window_close",
        "completed_headless",
        "controls",
    ],
)
def test_runtime_saves_partial_and_cleans_all_resources(
    monkeypatch, tmp_path, pose_factory, failure, view
):
    events = []
    plots, displayed = [], []
    saved_before_preview = []
    completed = failure.startswith("completed")
    session = SessionController()
    now = time.perf_counter()
    session.start_gesture.holds[7] = Hold(now - 2, now)
    session.start_gesture.grace_seconds = 5
    sessions = []

    def create_session(**kwargs):
        current = SessionController(**kwargs) if sessions else session
        sessions.append(current)
        return current

    monkeypatch.setattr(workout_session, "SessionController", create_session)
    if completed:
        advance_time = session.advance_time

        def expire_after_first_frame(now):
            if len(session.recorder) and session.active:
                session.finish(
                    session.start_time + session.duration_seconds, "completed"
                )
            else:
                advance_time(now)

        monkeypatch.setattr(session, "advance_time", expire_after_first_frame)

    class FakeCapture:
        reads = 0

        def isOpened(self):
            return failure != "camera_open"

        def set(self, *_args):
            pass

        def get(self, key):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720,
                cv2.CAP_PROP_FPS: 30,
            }[key]

        def read(self):
            self.reads += 1
            if completed and self.reads > 1:
                paths = list(tmp_path.glob("*.npz"))
                assert len(paths) == 1
                saved_before_preview.append(paths[0].read_bytes())
                assert session.rep_count == 3
            return (
                failure not in ("read", "completed_read") or self.reads == 1
            ), np.zeros((720, 1280, 3), np.uint8)

        def release(self):
            events.append("camera_released")

    class FakeModel:
        calls = 0

        def track(self, **kwargs):
            self.calls += 1
            assert kwargs["persist"] is True
            assert kwargs["quantize"] == 16 and kwargs["compile"] == "reduce-overhead"
            assert kwargs["imgsz"] == 640
            if self.calls == 2:
                if failure in ("inference", "completed_inference"):
                    raise RuntimeError("test inference error")
                if failure in ("keyboard", "completed_keyboard"):
                    raise KeyboardInterrupt()

            def plot(**kwargs):
                assert kwargs["labels"] is kwargs["boxes"]
                assert kwargs["conf"] is False
                plots.append(self.calls)
                return np.zeros((720, 1280, 3), np.uint8)

            return [SimpleNamespace(speed={}, plot=plot)]

    class FakeWorker:
        poses = SimpleNamespace(dropped=0)
        process = SimpleNamespace(is_alive=lambda: True)
        last_status = None
        forced_shutdown = False

        def __init__(self, *_args):
            self.closed = False
            self.session = sessions[-1]

        def start(self):
            events.append("worker_started")

        def publish(self, packet):
            assert not self.closed
            assert len(self.session.recorder) > 0  # recording precedes lossy IPC

        def poll(self):
            assert not self.closed

        def close(self):
            assert not self.closed
            self.closed = True
            events.append("worker_closed")
            if completed and self.session.last_packet is not None:
                self.last_status = AnalysisStatus(
                    3,
                    "CLOSED",
                    self.session.last_packet.frame_index,
                    self.session.generation,
                )
                return self.last_status

    model = FakeModel()
    monkeypatch.setattr(ultralytics, "YOLO", lambda *_args: model)
    monkeypatch.setattr(
        workout.tracking, "TrackerTimer", lambda model: SimpleNamespace(last_ms=0)
    )
    monkeypatch.setattr(
        workout.tracking, "select_tracker", lambda value: "fasttrack.yaml"
    )
    monkeypatch.setattr(
        workout.tracking, "tracked_people", lambda *args: [pose_factory("raised")]
    )
    monkeypatch.setattr(workout_session, "AnalysisWorker", FakeWorker)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: FakeCapture())
    monkeypatch.setattr(
        cv2, "destroyAllWindows", lambda: events.append("windows_closed")
    )
    args = workout_session.parse_args(
        ["--headless", "--frames", "2", "--output-dir", str(tmp_path), "--view", view]
    )
    if failure == "completed_toggle":
        args.metrics = tmp_path / "metrics.json"
        args.warmup = 0
    if failure in ("q", "window_close", "toggle", "controls") or (
        completed and failure != "completed_headless"
    ):
        args.headless = False
        if completed or failure == "controls":
            args.frames = 0
        monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
        monkeypatch.setattr(
            cv2, "imshow", lambda name, image: displayed.append(image.shape)
        )
        keys = iter(
            [ord(key) for key in "D+FV-FDq"]
            if failure == "controls"
            else [ord("v"), ord("V")]
            if failure == "toggle"
            else [-1, ord("v"), ord("V"), ord("q")]
            if failure == "completed_toggle"
            else [-1, ord("q")]
        )
        monkeypatch.setattr(
            cv2,
            "getWindowProperty",
            lambda *args: (
                0
                if failure == "window_close"
                or (failure == "completed_window_close" and model.calls >= 2)
                else 1
            ),
        )
        monkeypatch.setattr(
            cv2,
            "waitKey",
            lambda delay: (
                next(keys)
                if failure in ("toggle", "controls") or completed
                else ord("q")
                if failure == "q" or delay == 30
                else -1
            ),
        )
    if failure in ("inference", "camera_open", "completed_inference"):
        with pytest.raises(RuntimeError):
            workout_session.run(args)
    else:
        summary = workout_session.run(args)
    expected_events = [
        "worker_started",
        "camera_released",
        "worker_closed",
        "windows_closed",
    ]
    if completed:
        expected_events[1:3] = ["worker_closed", "camera_released"]
        if failure != "completed_headless":
            expected_events = [
                "worker_started",
                "worker_closed",
                "worker_started",
                "camera_released",
                "worker_closed",
                "windows_closed",
            ]
            assert len(sessions) == 2
            ready = sessions[-1]
            assert ready.state == SessionState.WAITING
            assert ready.rep_count == 0 and ready.start_time is None
            assert ready.selected_track_id is None and ready.last_packet is None
            assert len(ready.recorder) == 0 and ready.duration_seconds == args.duration
            assert ready.start_gesture is not session.start_gesture
    assert events == expected_events
    if failure == "toggle":
        assert model.calls == 2 and len(plots) == 1 and len(displayed) == 2
        assert {shape[:2] for shape in displayed} == {(900, 1280)}
        assert session.generation == 1  # presentation never resets continuity
    elif failure == "completed_toggle":
        assert model.calls == 4 and len(displayed) == 4  # fresh inference in every view
        assert {shape[:2] for shape in displayed} == {(900, 1280)}
        assert summary["measured_frames"] == 4  # ready mode still runs live inference
    elif view == "skeleton" and failure != "controls":
        assert plots == []  # never build/copy the camera overlay in skeleton mode
    if failure in ("completed", "completed_window_close"):
        assert model.calls == 2 and len(displayed) == 2
    elif failure in ("completed_read", "completed_headless"):
        assert model.calls == 1
    if failure == "controls":
        assert model.calls == len(displayed) == 8
        assert session.generation == 1 and session.selected_track_id == 7
    paths = list(tmp_path.glob("*.npz"))
    if failure == "camera_open":
        assert paths == []
    else:
        assert len(paths) == 1
        assert all(
            snapshot == paths[0].read_bytes() for snapshot in saved_before_preview
        )
        with np.load(paths[0], allow_pickle=False) as data:
            assert data["completed"].item() is completed
            assert data["poses"].shape[0] == (
                8
                if failure == "controls"
                else 2
                if failure in ("quit", "toggle")
                else 1
            )
            if completed:
                assert data["session_duration"].item() == session.duration_seconds
                assert data["target_duration"].item() == session.duration_seconds
                assert data["final_rep_count"].item() == session.rep_count == 3
            assert data["stop_reason"].item() == (
                "completed"
                if completed
                else {
                    "quit": "frame_limit",
                    "read": "camera_read_failed",
                    "inference": "error",
                    "keyboard": "keyboard_interrupt",
                    "q": "quit",
                    "window_close": "quit",
                    "toggle": "frame_limit",
                    "controls": "quit",
                }[failure]
            )


@pytest.mark.parametrize("view", ["video", "skeleton"])
@pytest.mark.parametrize("quit_during_second", [False, True])
def test_repeated_workouts_require_fresh_gestures_and_save_separate_sessions(
    monkeypatch, tmp_path, pose_factory, view, quit_during_second
):
    clock = [0.0]
    sessions, workers, saved_snapshots = [], [], {}
    monkeypatch.setattr(workout_session.time, "perf_counter", lambda: clock[0])

    def create_session(**kwargs):
        session = SessionController(**kwargs)
        sessions.append(session)
        return session

    class FakeCapture:
        reads = 0
        releases = 0

        def isOpened(self):
            return True

        def set(self, *_args):
            pass

        def get(self, key):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720,
                cv2.CAP_PROP_FPS: 10,
            }[key]

        def read(self):
            self.reads += 1
            assert self.reads < 100, "The next start gesture was never accepted"
            clock[0] = self.reads / 10
            for path in tmp_path.glob("*.npz"):
                snapshot = saved_snapshots.setdefault(path, path.read_bytes())
                assert path.read_bytes() == snapshot
            return True, np.zeros((720, 1280, 3), np.uint8)

        def release(self):
            self.releases += 1

    class FakeWorker:
        poses = SimpleNamespace(dropped=0)
        process = SimpleNamespace(is_alive=lambda: True)
        forced_shutdown = False
        last_status = None

        def __init__(self, *_args):
            self.last_packet = None
            self.closed = False
            workers.append(self)
            self.rep_count = len(workers)

        def start(self):
            pass

        def publish(self, packet):
            assert not self.closed
            if self.last_packet is None:
                assert packet.timestamp == 0 and packet.accepted_reps == 0
            self.last_packet = packet

        def poll(self):
            assert not self.closed
            if self.last_packet is not None:
                self.last_status = AnalysisStatus(
                    self.rep_count,
                    "CLOSED",
                    self.last_packet.frame_index,
                    self.last_packet.generation,
                )
            return self.last_status

        def close(self):
            status = self.poll()
            self.closed = True
            return status

    capture = FakeCapture()
    camera_opens, model_loads = [], []

    def open_camera(*args):
        camera_opens.append(args)
        return capture

    def load_model(*args):
        model_loads.append(args)
        return SimpleNamespace(
            track=lambda **kwargs: [
                SimpleNamespace(
                    speed={}, plot=lambda **kwargs: np.zeros((720, 1280, 3), np.uint8)
                )
            ]
        )

    def key(_delay):
        if quit_during_second:
            ready_to_quit = len(sessions) == 2 and len(sessions[-1].recorder) >= 2
        else:
            ready_to_quit = len(list(tmp_path.glob("*.npz"))) == 2
        return ord("q") if ready_to_quit else -1

    monkeypatch.setattr(workout_session, "SessionController", create_session)
    monkeypatch.setattr(workout_session, "AnalysisWorker", FakeWorker)
    monkeypatch.setattr(ultralytics, "YOLO", load_model)
    monkeypatch.setattr(cv2, "VideoCapture", open_camera)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "waitKey", key)
    monkeypatch.setattr(
        workout.tracking, "TrackerTimer", lambda model: SimpleNamespace(last_ms=0)
    )
    monkeypatch.setattr(
        workout.tracking, "select_tracker", lambda value: "fasttrack.yaml"
    )
    monkeypatch.setattr(
        workout.tracking,
        "tracked_people",
        lambda *args: [pose_factory("raised", track_id=7 if len(sessions) == 1 else 9)],
    )
    args = workout_session.parse_args(
        ["--duration", "0.5", "--view", view, "--output-dir", str(tmp_path)]
    )
    summary = workout_session.run(args)
    assert len(camera_opens) == len(model_loads) == capture.releases == 1
    assert len(sessions) == len(workers) == (2 if quit_during_second else 3)
    assert all(worker.closed for worker in workers)
    assert sessions[1].start_time >= sessions[0].end_time + 2.0
    assert all(path.read_bytes() == data for path, data in saved_snapshots.items())
    if not quit_during_second:
        assert sessions[-1].state == SessionState.WAITING
        assert sessions[-1].rep_count == 0 and len(sessions[-1].recorder) == 0
    paths = list(tmp_path.glob("*.npz"))
    assert len(paths) == 2
    frame_ranges = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as recording:
            track_id = recording["selected_track_id_at_start"].item()
            assert track_id in (7, 9)
            assert recording["target_duration"].item() == 0.5
            assert recording["timestamps"][0] == 0
            assert np.all(recording["timestamps"] < 0.5)
            assert recording["final_rep_count"].item() == (1 if track_id == 7 else 2)
            completed = track_id == 7 or not quit_during_second
            assert recording["completed"].item() is completed
            assert recording["stop_reason"].item() == (
                "completed" if completed else "quit"
            )
            frame_ranges[track_id] = recording["frame_indices"].copy()
            if track_id == 9:
                assert summary["saved_session"] == str(path)
                assert summary["recorded_frames"] == len(recording["poses"])
    assert frame_ranges[7][-1] < frame_ranges[9][0]
