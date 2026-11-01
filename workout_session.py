"""Configurable jumping-jack session (default 30 seconds) built around smoke_pose5.py.

Heavy imports live inside run(): Windows multiprocessing spawn loads this module
in the CPU worker, which must not import torch, create a model, or open a camera.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from workout.analysis.worker import AnalysisWorker
from workout.benchmark import FrameMetrics
from workout.session import SESSION_SECONDS, SessionController, SessionState

WINDOW_NAME = "PoseSense - Workout Session"
MODEL_PATH = "yolo26m-pose.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("msmf", "dshow", "auto"), default="msmf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument(
        "--duration",
        type=float,
        default=SESSION_SECONDS,
        help="Workout duration in seconds (default: 30)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-width", type=int, default=2560)
    parser.add_argument("--max-height", type=int, default=1600)
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start fullscreen; F toggles"
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.0,
        help="Workout text scale, 0.75 to 2.0; +/- adjusts live",
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Show technical details; D toggles"
    )
    parser.add_argument(
        "--view",
        choices=("video", "skeleton"),
        default="video",
        help="Initial display; V switches video / skeleton, D shows technical checks",
    )
    parser.add_argument(
        "--tracker", choices=("auto", "fasttrack", "bytetrack"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sessions"))
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip OpenCV window for finite benchmarks",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after N camera frames; 0 means unlimited",
    )
    parser.add_argument(
        "--metrics", type=Path, help="Write benchmark JSON after shutdown"
    )
    parser.add_argument(
        "--warmup", type=int, default=30, help="Exclude first N frames from metrics"
    )
    parser.add_argument(
        "--analysis-delay",
        type=float,
        default=0,
        help="Artificial CPU-worker delay in seconds",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    if not math.isfinite(args.ui_scale) or not 0.75 <= args.ui_scale <= 2.0:
        parser.error("ui-scale must be finite and between 0.75 and 2.0")
    if min(args.width, args.height, args.imgsz, args.max_width, args.max_height) <= 0:
        parser.error("Image/display dimensions must be positive")
    if args.frames < 0 or args.warmup < 0 or args.analysis_delay < 0:
        parser.error("frames, warmup and analysis-delay must be nonnegative")
    if (
        not math.isfinite(args.fps)
        or args.fps <= 0
        or not math.isfinite(args.analysis_delay)
    ):
        parser.error("fps must be finite and positive; analysis-delay must be finite")
    return args


def run(args: argparse.Namespace) -> dict:
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    import smoke_pose5 as reference
    from workout.live_display import LiveSkeletonView
    from workout.tracking import TrackerTimer, select_tracker, tracked_people
    from workout.workout_display import WorkoutDisplay

    device = reference.choose_device(args.device)
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    tracker_path = select_tracker(args.tracker)
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {device_name}")
    print(f"Tracker: {tracker_path} (persistent, no ReID)")
    print(f"Pose model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    tracker_timer = TrackerTimer(model)
    session = SessionController(duration_seconds=args.duration)
    worker = AnalysisWorker(args.analysis_delay)
    metrics = FrameMetrics(args.warmup)
    capture = None
    saved = None
    annotated = None
    result = None
    people = []
    hud_values = {}
    view = args.view
    skeleton_view = LiveSkeletonView()
    workout_display = WorkoutDisplay(
        max_width=args.max_width,
        max_height=args.max_height,
        ui_scale=args.ui_scale,
        fullscreen=args.fullscreen,
        diagnostics=args.diagnostics,
    )
    camera_width, camera_height, camera_fps = args.width, args.height, args.fps
    source_width, source_height = args.width, args.height
    frame_index = 0
    loop_ema = processing_ema = None
    previous_loop = time.perf_counter()
    analysis_note = ""
    next_health_check = 0.0
    stop_reason = "quit"
    event_count = 0
    runtime_info = {
        "variant": "workout",
        "model": MODEL_PATH,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": device_name,
        "imgsz": args.imgsz,
        "quantize": 16,
        "compile": "reduce-overhead",
        "tracker": Path(tracker_path).name,
        "detection_confidence": 0.1,
        "headless": args.headless,
        "backend": args.backend,
        "analysis_delay_seconds": args.analysis_delay,
        "synthetic_session": getattr(args, "synthetic_session", False),
        "initial_view": view,
        "initial_ui_scale": args.ui_scale,
        "initial_diagnostics": args.diagnostics,
        "initial_fullscreen": args.fullscreen,
    }
    finalized = False
    last_recording = None

    def finalize_session() -> None:
        nonlocal finalized, saved, last_recording
        if finalized:
            return
        finalized = True
        runtime_info["final_view"] = view
        session.finish(time.perf_counter(), stop_reason)
        try:
            final_status = worker.close()
            session.accept_status(final_status, final=True)
        finally:
            if session.start_time is not None:
                saved = session.recorder.save(
                    args.output_dir,
                    session.metadata(
                        camera_width=camera_width,
                        camera_height=camera_height,
                        nominal_source_fps=camera_fps,
                        runtime=runtime_info,
                        analysis_dropped_packets=worker.poses.dropped,
                        analysis_processed_packets=worker.last_status.processed_packets
                        if worker.last_status
                        else 0,
                        analysis_forced_shutdown=worker.forced_shutdown,
                        analysis_error=worker.last_status.error
                        if worker.last_status
                        else "",
                    ),
                )
                print(
                    f"Saved {len(session.recorder)} frames, {session.rep_count} reps: {saved.resolve()}"
                )
                last_recording = (session, worker, saved)
                workout_display.notify_saved(session.rep_count, time.perf_counter())

    def rearm_session() -> bool:
        nonlocal session, worker, finalized, saved
        nonlocal event_count, analysis_note, next_health_check
        finalize_session()
        if args.headless:
            return False
        session = SessionController(duration_seconds=args.duration)
        worker = AnalysisWorker(args.analysis_delay)
        finalized = False
        saved = None
        event_count = 0
        analysis_note = ""
        next_health_check = 0.0
        worker.start()
        print("Ready for next workout: raise anatomical RIGHT hand for 2 s.")
        return True

    try:
        worker.start()
        capture = cv2.VideoCapture(
            args.camera, reference.get_camera_backend(args.backend)
        )
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera} using {args.backend}"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        camera_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(camera_fps):
            camera_fps = 0.0
        print(f"Camera: {camera_width}x{camera_height} @ {camera_fps:.1f} FPS")
        print(
            "Raise anatomical RIGHT hand for 2 s. V: view, D: details, F: fullscreen, +/-: text size. "
            "Q / Esc / close window: save partial and quit."
        )
        if not args.headless:
            workout_display.open(WINDOW_NAME)
        previous_loop = time.perf_counter()
        while True:
            loop_start = time.perf_counter()
            session.advance_time(loop_start)
            if session.state == SessionState.FINISHED and not rearm_session():
                break
            delta_ms = (loop_start - previous_loop) * 1000
            previous_loop = loop_start
            loop_ema = delta_ms if loop_ema is None else 0.1 * delta_ms + 0.9 * loop_ema
            ok, frame = capture.read()
            if not ok:
                stop_reason = "camera_read_failed"
                print("Camera frame read failed; saving any active session.")
                break
            source_height, source_width = frame.shape[:2]
            # Check after capture too, so a stalled camera cannot extend recording.
            session.advance_time(time.perf_counter())
            if session.state == SessionState.FINISHED and not rearm_session():
                break
            frame_index += 1
            processing_start = time.perf_counter()
            result = model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                device=device,
                imgsz=args.imgsz,
                quantize=16,
                compile="reduce-overhead",
                verbose=False,
            )[0]
            now = time.perf_counter()
            if session.state != SessionState.FINISHED:
                status = worker.poll()
                if status is not None and status.error:
                    analysis_note = "ERROR (recording continues)"
                    print(status.error)
                if now >= next_health_check:
                    next_health_check = now + 1
                    if not worker.process.is_alive():
                        analysis_note = "OFFLINE (recording continues)"
                    elif session.last_packet is not None and (
                        worker.last_status is None
                        or frame_index - worker.last_status.last_processed_frame
                        > max(30, camera_fps)
                    ):
                        analysis_note = "LAGGING"
                    elif not (worker.last_status and worker.last_status.error):
                        analysis_note = ""
            people = tracked_people(result, session.selected_track_id)
            if session.state != SessionState.FINISHED:
                packet = session.process_frame(people, frame_index, now, status)
                if packet is not None:
                    worker.publish(packet)
            if session.state == SessionState.FINISHED and not rearm_session():
                break
            for event in session.events[event_count:]:
                print("Session:", event)
            event_count = len(session.events)
            speeds = result.speed or {}
            plot_start = time.perf_counter()
            if view == "video":
                annotated = result.plot(
                    labels=workout_display.diagnostics,
                    boxes=workout_display.diagnostics,
                    conf=False,
                )
            elif workout_display.diagnostics:
                annotated = skeleton_view.render(
                    session,
                    people,
                    camera_width=source_width,
                    camera_height=source_height,
                    now=now,
                    frame_index=frame_index,
                    dropped=worker.poses.dropped,
                    analysis_note=analysis_note,
                    saved=saved,
                )
            else:
                annotated = None
            hud_values = {
                "frame_index": frame_index,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "camera_fps": camera_fps,
                "loop_fps": 1000 / loop_ema if loop_ema else 0,
                "processing_fps": 1000 / processing_ema if processing_ema else 0,
                "preprocess_ms": speeds.get("preprocess", 0),
                "inference_ms": speeds.get("inference", 0),
                "postprocess_ms": speeds.get("postprocess", 0),
                # Full rendering timings are available on the following frame.
                "plot_ms": hud_values.get("plot_ms", 0),
                "total_ms": hud_values.get("total_ms", 0),
                "device_name": device_name,
                "backend_name": args.backend,
            }
            if workout_display.diagnostics:
                if view == "video":
                    reference.draw_hud(annotated, **hud_values)
                else:
                    skeleton_view.draw_performance(annotated, hud_values)
            display = workout_display.render(
                session,
                people,
                now=now,
                size=workout_display.viewport_size(WINDOW_NAME)
                if not args.headless
                else workout_display.size,
                camera_width=source_width,
                camera_height=source_height,
                video=annotated if view == "video" else None,
                diagnostic_image=annotated
                if view == "skeleton" and workout_display.diagnostics
                else None,
                analysis_note=analysis_note,
                dropped=worker.poses.dropped,
            )
            plot_ms = (time.perf_counter() - plot_start) * 1000
            processing_ms = (time.perf_counter() - processing_start) * 1000
            processing_ema = (
                processing_ms
                if processing_ema is None
                else 0.1 * processing_ms + 0.9 * processing_ema
            )
            hud_values.update(plot_ms=plot_ms, total_ms=processing_ms)
            key = -1
            if not args.headless:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
            if args.metrics and session.state != SessionState.FINISHED:
                metrics.add(
                    loop_ms=(time.perf_counter() - loop_start) * 1000,
                    processing_ms=processing_ms,
                    tracker_ms=tracker_timer.last_ms,
                    preprocess_ms=float(speeds.get("preprocess", 0)),
                    inference_ms=float(speeds.get("inference", 0)),
                    postprocess_ms=float(speeds.get("postprocess", 0)),
                    plot_ms=plot_ms,
                )
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("v"), ord("V")):
                view = "skeleton" if view == "video" else "video"
            workout_display.handle_key(key, WINDOW_NAME)
            if (
                not args.headless
                and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if args.frames and frame_index >= args.frames:
                stop_reason = "frame_limit"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; saving any active session.")
    except BaseException:
        stop_reason = "error"
        raise
    finally:
        # Completed sessions were saved before rearming. Early exits save any
        # active workout here; quitting while waiting creates no recording.
        try:
            if capture is not None:
                capture.release()
        finally:
            try:
                finalize_session()
            finally:
                cv2.destroyAllWindows()
    recorded_session, recorded_worker, recorded_path = (
        last_recording if last_recording is not None else (session, worker, None)
    )
    summary = {
        **runtime_info,
        "camera_width": camera_width,
        "camera_height": camera_height,
        "nominal_source_fps": camera_fps,
        "recorded_frames": len(recorded_session.recorder),
        "session_state": recorded_session.state.value,
        "final_rep_count": recorded_session.rep_count,
        "analysis_dropped_packets": recorded_worker.poses.dropped,
        "analysis_processed_packets": recorded_worker.last_status.processed_packets
        if recorded_worker.last_status
        else 0,
        "saved_session": str(recorded_path) if recorded_path else None,
    }
    if args.metrics:
        summary = metrics.save(args.metrics, **summary)
        print(f"Benchmark: {summary['average_loop_fps']:.2f} FPS; {args.metrics}")
    return summary


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    run(parse_args())
