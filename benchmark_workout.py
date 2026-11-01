"""Finite A/B runs of the actual applications; never edit the reference.

Run variants in separate processes to avoid sharing compiled models or trackers.
Headless runs still plot poses, render both HUDs and resize, but skip GUI events.
"""

import argparse
import time
from pathlib import Path

from workout.benchmark import FrameMetrics


def run_original(args) -> dict:
    from unittest.mock import patch

    import cv2
    import torch
    import ultralytics

    import smoke_pose5 as reference

    metrics = FrameMetrics(args.warmup)
    original_capture = cv2.VideoCapture
    original_wait_key = cv2.waitKey
    original_draw_hud = reference.draw_hud
    latest = {}
    loop_start = 0.0

    class TimedCapture:
        def __init__(self, *values, **kwargs):
            self.capture = original_capture(*values, **kwargs)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def read(self):
            nonlocal loop_start
            loop_start = time.perf_counter()
            return self.capture.read()

    def draw_hud(image, **values):
        latest.update(values)
        return original_draw_hud(image, **values)

    def wait_key(delay):
        key = -1 if args.headless else original_wait_key(delay)
        metrics.add(
            loop_ms=(time.perf_counter() - loop_start) * 1000,
            processing_ms=latest["total_ms"],
            tracker_ms=0.0,
            preprocess_ms=latest["preprocess_ms"],
            inference_ms=latest["inference_ms"],
            postprocess_ms=latest["postprocess_ms"],
            plot_ms=latest["plot_ms"],
        )
        return ord("q") if metrics.seen >= args.frames else key

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(reference, "parse_args", return_value=args))
        stack.enter_context(patch.object(reference, "draw_hud", side_effect=draw_hud))
        stack.enter_context(patch.object(cv2, "VideoCapture", TimedCapture))
        stack.enter_context(patch.object(cv2, "waitKey", side_effect=wait_key))
        if args.headless:
            stack.enter_context(patch.object(cv2, "namedWindow"))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
        reference.main()
    result = metrics.save(
        args.metrics,
        variant="original",
        torch=torch.__version__,
        ultralytics=ultralytics.__version__,
        device=latest.get("device_name"),
        camera_width=latest.get("camera_width"),
        camera_height=latest.get("camera_height"),
        nominal_source_fps=latest.get("camera_fps"),
        headless=args.headless,
        backend=args.backend,
        imgsz=args.imgsz,
        quantize=16,
        compile="reduce-overhead",
        detection_confidence=0.25,
        tracker=None,
        recorded_frames=0,
        analysis_dropped_packets=0,
    )
    print(f"Benchmark: {result['average_loop_fps']:.2f} FPS; {args.metrics}")
    return result


def main() -> None:
    import workout_session

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--variant", choices=("original", "workout"), required=True)
    selector.add_argument(
        "--synthetic-session",
        action="store_true",
        help="Workout only: feed scripted poses after real inference to benchmark active recording",
    )
    chosen, remaining = selector.parse_known_args()
    args = workout_session.parse_args(remaining)
    args.synthetic_session = chosen.synthetic_session
    if chosen.synthetic_session and chosen.variant != "workout":
        selector.error("--synthetic-session requires --variant workout")
    if not args.frames:
        args.frames = 330
    if args.frames <= args.warmup:
        selector.error("--frames must exceed --warmup")
    if args.metrics is None:
        args.metrics = Path("benchmarks") / f"{chosen.variant}.json"
    if chosen.variant == "original":
        run_original(args)
    elif chosen.synthetic_session:
        run_synthetic_session(args)
    else:
        workout_session.run(args)


def run_synthetic_session(args) -> dict:
    """Real camera/YOLO/tracker, explicitly synthetic session input only.

    This exercises selection, loss, counter IPC and recording without requiring a
    human to exercise during a benchmark. It is NOT an accuracy measurement.
    """
    from unittest.mock import patch

    import numpy as np

    import workout_session
    from workout.packets import TrackedPerson
    from workout.session import SessionController

    closed = np.full((17, 3), 0.5, np.float32)
    closed[:, 2] = 0.99
    closed[0, :2] = (0.5, 0.2)
    closed[5, :2], closed[6, :2] = (0.4, 0.4), (0.6, 0.4)
    closed[9, :2], closed[10, :2] = (0.4, 0.65), (0.6, 0.65)
    closed[15, :2], closed[16, :2] = (0.45, 0.9), (0.55, 0.9)
    raised = closed.copy()
    raised[10, 1], raised[8, 1] = 0.1, 0.3
    opened = closed.copy()
    opened[9, 1] = opened[10, 1] = 0.1
    opened[15, 0], opened[16, 0] = 0.25, 0.75

    class ScriptedSession(SessionController):
        def process_frame(self, people, frame_index, now=None, status=None):
            elapsed = self.elapsed(now)
            pose = (
                raised
                if self.start_time is None
                else (closed if int(elapsed / 0.5) % 2 == 0 else opened)
            )
            scripted = (
                []
                if self.active and 3 <= elapsed < 3.3
                else [TrackedPerson(999999, pose)]
            )
            return super().process_frame(scripted, frame_index, now, status)

    args.output_dir = args.metrics.parent / "synthetic_sessions"
    print(
        "SYNTHETIC SESSION benchmark: scripted poses; real camera/YOLO/tracker timings."
    )
    with patch.object(workout_session, "SessionController", ScriptedSession):
        return workout_session.run(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
