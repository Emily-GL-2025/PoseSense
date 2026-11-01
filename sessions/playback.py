r"""Play the keypoints/skeleton in a workout_session.py NPZ recording.

Run from the project folder:
    .venv\Scripts\python.exe sessions/playback.py
    .venv\Scripts\python.exe sessions/playback.py sessions/workout_<timestamp>.npz

Without a filename, open the newest NPZ next to this script. No camera, YOLO,
torch, or workout package is needed; this standalone player only reads the NPZ.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "PoseSense - Recorded Skeleton"
HEADER_HEIGHT, FOOTER_HEIGHT = 80, 100
BACKGROUND = (22, 19, 17)  # OpenCV colors are BGR
TEXT = (233, 234, 237)
MUTED = (160, 157, 150)
LEFT_COLOR = (200, 220, 50)  # anatomical left: cyan
RIGHT_COLOR = (80, 165, 250)  # anatomical right: orange

# COCO skeleton connections, converted from Ultralytics' one-based list to
# zero-based indices. No inference library is imported just to draw these lines.
SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


@dataclass
class Recording:
    path: Path
    poses: np.ndarray
    timestamps: np.ndarray
    present: np.ndarray
    frame_indices: np.ndarray
    track_ids: np.ndarray
    rep_counts: np.ndarray
    camera_width: int
    camera_height: int
    duration: float

    @classmethod
    def load(cls, path: Path) -> Recording:
        """Load owned arrays and close the archive; never enable NumPy pickle."""
        with np.load(path, allow_pickle=False) as archive:
            poses = archive["poses"].astype(np.float32, copy=False)
            timestamps = archive["timestamps"].astype(np.float64, copy=False)
            present = archive["present"].astype(np.bool_, copy=False)
            indices = archive["frame_indices"].astype(np.int64, copy=False)
            ids = archive["observed_track_ids"].astype(np.int64, copy=False)
            reps = archive["rep_counts"].astype(np.int64, copy=False)
            width = int(archive["camera_width"].item())
            height = int(archive["camera_height"].item())
            duration = float(archive["session_duration"].item())
        if poses.ndim != 3 or poses.shape[1:] != (17, 3):
            raise ValueError("Expected poses with shape [N,17,3]")
        count = len(poses)
        if not count:
            raise ValueError("This recording contains no pose frames")
        if any(
            array.shape != (count,)
            for array in (timestamps, present, indices, ids, reps)
        ):
            raise ValueError("All per-frame arrays must have the same length as poses")
        if (
            not np.isfinite(timestamps).all()
            or timestamps[0] < 0
            or (np.diff(timestamps) < 0).any()
        ):
            raise ValueError("Timestamps must be finite, nonnegative and monotonic")
        if width <= 0 or height <= 0:
            raise ValueError("Recorded camera width and height must be positive")
        if not math.isfinite(duration) or duration < timestamps[-1]:
            raise ValueError("Session duration must be finite and cover all timestamps")
        return cls(
            path,
            poses,
            timestamps,
            present,
            indices,
            ids,
            reps,
            width,
            height,
            duration,
        )

    def frame_at(self, seconds: float) -> int:
        """Show the last observation at this time; do not interpolate missing poses."""
        return max(
            0,
            min(
                len(self.poses) - 1,
                int(np.searchsorted(self.timestamps, seconds, side="right")) - 1,
            ),
        )


class PlaybackClock:
    """Elapsed-time playback independent of the UI refresh rate or nominal FPS."""

    def __init__(self, duration: float, speed: float = 1.0, loop: bool = False):
        self.duration = duration
        self.speed = speed
        self.loop = loop
        self.position = 0.0
        self.playing = True
        self.last_clock = time.perf_counter()

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self.playing:
            self.position += max(0.0, now - self.last_clock) * self.speed
            if self.position >= self.duration:
                if self.loop and self.duration > 0:
                    self.position %= self.duration
                else:
                    self.position = self.duration
                    self.playing = False
        self.last_clock = now
        return self.position

    def seek(self, seconds: float, now: float | None = None) -> None:
        self.position = max(0.0, min(self.duration, seconds))
        self.last_clock = time.perf_counter() if now is None else now


def resolve_recording(filename: Path | None) -> Path:
    directory = Path(__file__).resolve().parent
    if filename is not None:
        if filename.is_file():
            return filename.resolve()
        sibling = directory / filename
        if sibling.is_file():
            return sibling.resolve()
        raise FileNotFoundError(f"Recording not found: {filename}")
    paths = list(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ recordings found in {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def display_size(
    recording: Recording, max_width: int, max_height: int
) -> tuple[int, int]:
    scale = min(
        max_width / recording.camera_width,
        (max_height - HEADER_HEIGHT - FOOTER_HEIGHT) / recording.camera_height,
    )
    return max(640, round(recording.camera_width * scale)), round(
        recording.camera_height * scale
    ) + HEADER_HEIGHT + FOOTER_HEIGHT


def projected_points(
    recording: Recording, index: int, size: tuple[int, int], confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Restore the source aspect ratio, center the pose, and mask unreliable points."""
    width, height = size
    view_height = height - HEADER_HEIGHT - FOOTER_HEIGHT
    scale = min(width / recording.camera_width, view_height / recording.camera_height)
    source_size = np.array([recording.camera_width, recording.camera_height])
    offset = np.array(
        [
            (width - recording.camera_width * scale) / 2,
            HEADER_HEIGHT + (view_height - recording.camera_height * scale) / 2,
        ]
    )
    pose = recording.poses[index]
    valid = (
        np.isfinite(pose).all(axis=1)
        & (pose[:, 2] >= confidence)
        & (pose[:, :2] >= 0).all(axis=1)
        & (pose[:, :2] <= 1).all(axis=1)
    )
    if not recording.present[index]:
        valid[:] = False
    points = np.zeros((17, 2), dtype=np.int32)
    points[valid] = np.rint(pose[valid, :2] * source_size * scale + offset).astype(
        np.int32
    )
    return points, valid


def draw_text(
    image, message: str, x: int, y: int, color=TEXT, scale: float = 0.55
) -> None:
    """Elide long filenames/control text instead of drawing past the window edge."""
    available = image.shape[1] - x - 16
    while (
        message
        and cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        > available
    ):
        message = message[:-4] + "..." if len(message) > 4 else ""
    cv2.putText(
        image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def time_label(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{seconds % 60:05.2f}"


def render_frame(
    recording: Recording,
    index: int,
    clock: PlaybackClock,
    size: tuple[int, int],
    confidence: float = 0.5,
    show_indices: bool = False,
) -> np.ndarray:
    """Draw one observation on a neutral canvas; never modify the saved pose."""
    width, height = size
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    points, valid = projected_points(recording, index, size, confidence)
    bottom = height - FOOTER_HEIGHT
    cv2.line(image, (0, HEADER_HEIGHT), (width, HEADER_HEIGHT), (60, 54, 48), 1)
    cv2.line(image, (0, bottom), (width, bottom), (60, 54, 48), 1)
    for first, second in SKELETON:
        if valid[first] and valid[second]:
            color = (
                LEFT_COLOR
                if first % 2 and second % 2
                else RIGHT_COLOR
                if not first % 2 and not second % 2
                else MUTED
            )
            cv2.line(
                image,
                tuple(points[first]),
                tuple(points[second]),
                color,
                3,
                cv2.LINE_AA,
            )
    for point in np.flatnonzero(valid):
        color = MUTED if point == 0 else LEFT_COLOR if point % 2 else RIGHT_COLOR
        cv2.circle(image, tuple(points[point]), 5, color, -1, cv2.LINE_AA)
        if show_indices:
            x, y = points[point]
            cv2.putText(
                image,
                str(point),
                (int(x) + 7, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                TEXT,
                1,
                cv2.LINE_AA,
            )
    state = (
        "PLAYING"
        if clock.playing
        else "ENDED"
        if clock.position >= recording.duration
        else "PAUSED"
    )
    draw_text(image, recording.path.name, 18, 26)
    draw_text(
        image,
        f"{state}  {time_label(clock.position)} / {time_label(recording.duration)}"
        f"   {clock.speed:g}x   {'LOOP' if clock.loop else ''}",
        18,
        55,
    )
    if not valid.any():
        missing = not recording.present[index]
        message = (
            "PERSON MISSING - no pose recorded"
            if missing
            else "No confident keypoints in this frame"
        )
        draw_text(
            image,
            message,
            30,
            HEADER_HEIGHT + (bottom - HEADER_HEIGHT) // 2,
            MUTED,
            0.65,
        )
    draw_text(image, "L: cyan    R: orange", max(18, width - 230), 55, MUTED, 0.45)
    draw_text(
        image,
        f"Sample {index + 1}/{len(recording.poses)}  |  Source frame {recording.frame_indices[index]}"
        f"  |  ID {recording.track_ids[index]}  |  Reps {recording.rep_counts[index]}"
        f"  |  Valid joints {int(valid.sum())}/17",
        18,
        bottom + 23,
        MUTED,
        0.48,
    )
    progress = clock.position / recording.duration if recording.duration else 0
    cv2.line(image, (20, bottom + 42), (width - 20, bottom + 42), (70, 65, 58), 5)
    cv2.line(
        image,
        (20, bottom + 42),
        (20 + round((width - 40) * progress), bottom + 42),
        LEFT_COLOR,
        5,
    )
    draw_text(
        image,
        "Space: pause/play   A/D: -/+5s   [/]: step   R: restart   Q/Esc: quit",
        18,
        bottom + 68,
        scale=0.45,
    )
    draw_text(
        image,
        "-/+: speed   L: loop   I: joint indices   Click timeline to seek",
        18,
        bottom + 89,
        MUTED,
        0.45,
    )
    return image


def play(recording: Recording, args: argparse.Namespace) -> None:
    size = display_size(recording, args.max_width, args.max_height)
    clock = PlaybackClock(recording.duration, args.speed, args.loop)
    clock.seek(args.start)
    manual_frame: int | None = None
    index = recording.frame_at(clock.position)
    show_indices = args.indices

    def mouse(event, x, y, _flags, _parameter):
        nonlocal manual_frame
        timeline_y = size[1] - FOOTER_HEIGHT + 42
        if event == cv2.EVENT_LBUTTONDOWN and abs(y - timeline_y) <= 14:
            clock.seek((x - 20) / (size[0] - 40) * recording.duration)
            manual_frame = None

    print(f"Playing: {recording.path}")
    print(
        f"{len(recording.poses)} samples | {recording.duration:.2f} seconds | "
        f"{int((~recording.present).sum())} missing-person samples"
    )
    print(
        "Space: pause | A/D: seek | [/]: step | +/-: speed | R: restart | L: loop | I: indices | Q/Esc: quit"
    )
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, mouse)
        clock.seek(args.start)  # window startup should not advance the timeline
        while True:
            clock.tick()
            index = (
                manual_frame
                if manual_frame is not None
                else recording.frame_at(clock.position)
            )
            cv2.imshow(
                WINDOW_NAME,
                render_frame(
                    recording, index, clock, size, args.confidence, show_indices
                ),
            )
            key = cv2.waitKeyEx(10)
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            clock.tick()  # charge GUI/event time to the old speed/pause state
            if key == ord(" "):
                if not clock.playing and clock.position >= recording.duration:
                    clock.seek(0)
                clock.playing = not clock.playing
                manual_frame = None
            elif key in (ord("r"), ord("R")):
                clock.seek(0)
                clock.playing = True
                manual_frame = None
            elif key in (ord("a"), ord("A"), ord("d"), ord("D")):
                clock.seek(clock.position + (-5 if key in (ord("a"), ord("A")) else 5))
                manual_frame = None
            elif key in (ord("["), ord("]")):
                manual_frame = max(
                    0,
                    min(
                        len(recording.poses) - 1, index + (-1 if key == ord("[") else 1)
                    ),
                )
                clock.seek(float(recording.timestamps[manual_frame]))
                clock.playing = False
            elif key in (ord("+"), ord("="), ord("-"), ord("_")):
                factor = 0.5 if key in (ord("-"), ord("_")) else 2
                clock.speed = min(8.0, max(0.125, clock.speed * factor))
            elif key in (ord("l"), ord("L")):
                clock.loop = not clock.loop
            elif key in (ord("i"), ord("I")):
                show_indices = not show_indices
    finally:
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Play a workout NPZ as animated COCO keypoints and skeleton."
    )
    parser.add_argument(
        "recording",
        nargs="?",
        type=Path,
        help="NPZ path; default: newest NPZ beside this script",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback multiplier, 0.125 to 8 (default: 1)",
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start at this many session seconds"
    )
    parser.add_argument(
        "--loop", action="store_true", help="Repeat automatically at the end"
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="Label joints with their zero-based COCO index",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum joint confidence (default: 0.5)",
    )
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or not 0.125 <= args.speed <= 8:
        parser.error("--speed must be between 0.125 and 8")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.max_width < 640 or args.max_height < 400:
        parser.error("Minimum display bounds are 640 wide and 400 high")
    try:
        recording = Recording.load(resolve_recording(args.recording))
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.error(f"Cannot load recording: {error}")
    try:
        play(recording, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
