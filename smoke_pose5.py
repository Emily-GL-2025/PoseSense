from __future__ import annotations

import argparse
import time

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "yolo26n-pose.pt"
WINDOW_NAME = "YOLO26 Pose - Live Camera"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

DEFAULT_MAX_DISPLAY_WIDTH = 1280
DEFAULT_MAX_DISPLAY_HEIGHT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 Pose live laptop-camera smoke test."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )

    parser.add_argument(
        "--backend",
        choices=("msmf", "dshow", "auto"),
        default="msmf",
        help="Windows camera backend. Default: msmf",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Requested camera width. Default: 1280",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Requested camera height. Default: 720",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Requested camera FPS. Default: 30",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size. Default: 640",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_DISPLAY_WIDTH,
        help="Maximum display width.",
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_DISPLAY_HEIGHT,
        help="Maximum display height.",
    )

    return parser.parse_args()


def choose_device(requested: str) -> int | str:
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        return 0

    if torch.cuda.is_available():
        return 0

    return "cpu"


def get_camera_backend(name: str) -> int:
    if name == "msmf":
        return cv2.CAP_MSMF

    if name == "dshow":
        return cv2.CAP_DSHOW

    return cv2.CAP_ANY


def fit_for_display(
    image,
    max_width: int,
    max_height: int,
):
    """
    Scale down while preserving aspect ratio.

    Never stretches or enlarges the image.
    """
    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale >= 1.0:
        return image

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_hud(
    image,
    *,
    frame_index: int,
    camera_width: int,
    camera_height: int,
    camera_fps: float,
    loop_fps: float,
    processing_fps: float,
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    plot_ms: float,
    total_ms: float,
    device_name: str,
    backend_name: str,
):
    lines = [
        f"Frame: {frame_index}",
        (
            f"Camera: {camera_width}x{camera_height} "
            f"@ {camera_fps:.1f} FPS"
        ),
        f"Loop FPS: {loop_fps:.1f}",
        f"Processing FPS: {processing_fps:.1f}",
        (
            f"YOLO ms: pre {preprocess_ms:.1f} | "
            f"infer {inference_ms:.1f} | "
            f"post {postprocess_ms:.1f}"
        ),
        (
            f"Other ms: plot {plot_ms:.1f} | "
            f"total {total_ms:.1f}"
        ),
        f"Backend: {backend_name.upper()}",
        f"Device: {device_name}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 24

    max_text_width = 0

    for line in lines:
        text_size, _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )

        max_text_width = max(
            max_text_width,
            text_size[0],
        )

    panel_width = min(
        image.shape[1] - 10,
        max_text_width + 24,
    )

    panel_height = min(
        image.shape[0] - 10,
        len(lines) * line_height + 12,
    )

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (panel_width, panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        image,
        0.40,
        0,
        image,
    )

    y = 26

    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        y += line_height


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    device = choose_device(args.device)

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if device == 0:
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    print("Inference device:", device_name)

    # ----------------------------------------------------------
    # YOLO
    # ----------------------------------------------------------

    print()
    print("Loading YOLO26 Pose...")

    model = YOLO(MODEL_PATH)

    # ----------------------------------------------------------
    # Camera
    # ----------------------------------------------------------

    backend = get_camera_backend(args.backend)

    print()
    print(
        f"Opening camera {args.camera} "
        f"using backend: {args.backend}"
    )

    capture = cv2.VideoCapture(
        args.camera,
        backend,
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera} "
            f"using backend '{args.backend}'."
        )

    try:
        # Request desired camera mode.
        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            args.width,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            args.height,
        )

        capture.set(
            cv2.CAP_PROP_FPS,
            args.fps,
        )

        # Query what the camera actually accepted.
        camera_width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        camera_height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        camera_fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        print()
        print("Requested camera mode:")
        print(
            f"  {args.width} x {args.height} "
            f"@ {args.fps:.1f} FPS"
        )

        print()
        print("Reported camera mode:")
        print(
            f"  {camera_width} x {camera_height} "
            f"@ {camera_fps:.1f} FPS"
        )

        print()
        print("Controls:")
        print("  Q / Esc : quit")
        print()

        # AUTOSIZE prevents aspect-ratio distortion.
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE,
        )

        # ------------------------------------------------------
        # Timing state
        # ------------------------------------------------------

        frame_index = 0

        processing_ms_ema: float | None = None
        loop_ms_ema: float | None = None

        ema_alpha = 0.10

        previous_loop_start = time.perf_counter()

        # ------------------------------------------------------
        # Main camera loop
        # ------------------------------------------------------

        while True:
            loop_start = time.perf_counter()

            loop_delta_ms = (
                loop_start - previous_loop_start
            ) * 1000.0

            previous_loop_start = loop_start

            if loop_ms_ema is None:
                loop_ms_ema = loop_delta_ms
            else:
                loop_ms_ema = (
                    ema_alpha * loop_delta_ms
                    + (1.0 - ema_alpha)
                    * loop_ms_ema
                )

            loop_fps = (
                1000.0 / loop_ms_ema
                if loop_ms_ema > 0
                else 0.0
            )

            # --------------------------------------------------
            # Capture
            # --------------------------------------------------

            ok, frame = capture.read()

            if not ok:
                print()
                print("Camera frame read failed.")
                break

            frame_index += 1

            processing_start = time.perf_counter()

            # --------------------------------------------------
            # YOLO Pose
            # --------------------------------------------------

            results = model.predict(
                source=frame,
                device=device,
                imgsz=args.imgsz,
                quantize=16,
                compile="reduce-overhead",
                verbose=False,
            )

            result = results[0]

            speeds = result.speed or {}

            preprocess_ms = float(
                speeds.get("preprocess", 0.0)
            )

            inference_ms = float(
                speeds.get("inference", 0.0)
            )

            postprocess_ms = float(
                speeds.get("postprocess", 0.0)
            )

            # --------------------------------------------------
            # Pose overlay
            # --------------------------------------------------

            plot_start = time.perf_counter()

            annotated_frame = result.plot()

            plot_ms = (
                time.perf_counter()
                - plot_start
            ) * 1000.0

            total_processing_ms = (
                time.perf_counter()
                - processing_start
            ) * 1000.0

            # --------------------------------------------------
            # Smoothed processing FPS
            # --------------------------------------------------

            if processing_ms_ema is None:
                processing_ms_ema = total_processing_ms
            else:
                processing_ms_ema = (
                    ema_alpha * total_processing_ms
                    + (1.0 - ema_alpha)
                    * processing_ms_ema
                )

            processing_fps = (
                1000.0 / processing_ms_ema
                if processing_ms_ema > 0
                else 0.0
            )

            # --------------------------------------------------
            # HUD
            # --------------------------------------------------

            draw_hud(
                annotated_frame,
                frame_index=frame_index,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_fps=camera_fps,
                loop_fps=loop_fps,
                processing_fps=processing_fps,
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
                plot_ms=plot_ms,
                total_ms=total_processing_ms,
                device_name=device_name,
                backend_name=args.backend,
            )

            # --------------------------------------------------
            # Display, preserving aspect ratio
            # --------------------------------------------------

            display_frame = fit_for_display(
                annotated_frame,
                max_width=args.max_width,
                max_height=args.max_height,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            # waitKey(1) allows Windows to process GUI events.
            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            # User clicked the window close button.
            if (
                cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            ):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print()
    print("Camera closed.")


if __name__ == "__main__":
    main()