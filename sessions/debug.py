r"""Inspect jumping-jack gates without changing the workout app or recordings.

From the project folder:
    .venv\Scripts\python.exe sessions\debug.py
    .venv\Scripts\python.exe sessions\debug.py session_name.npz --start 10 --paused
    .venv\Scripts\python.exe sessions\debug.py --summary
    .venv\Scripts\python.exe sessions\debug.py --start 10 --snapshot inspection.png
    .venv\Scripts\python.exe sessions\debug.py --self-test

Space: pause; [/]: one sample; A/D: seek 5 seconds; N/P: next/previous event;
-/+: speed; R: restart; L: loop; I: joint indices; T: spatial guides; H: help;
F: choose an NPZ in the current recording's folder. Up/Down or W/S selects,
Enter loads paused at zero, Esc cancels; Q quits. Threshold edits are retained.
click the timeline to seek. Click a threshold row and use ,/. or its -/+ buttons
to experiment. B restores the current source defaults. All edits are in memory.
CLI overrides use the config names with hyphens, e.g. --open-ankle-ratio 1.6.

The real JumpingJackCounter processes EVERY saved sample before playback, so
seeking, slow playback, and skipped display frames cannot alter the result.
Replay carries its own accepted repetition total across recorded generations.
It is a hypothetical all-samples replay, not a reconstruction of live mailbox
delivery/status acceptance. The NPZ does not store that per-packet history or a
snapshot of the original analysis configuration. Defaults come from the CURRENT
source. Saved live counts remain separate. NaN coordinates cannot be recovered
by lowering confidence. This tool never loads torch/YOLO or opens a camera.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

# Support both `python sessions/debug.py` and `python -m sessions.debug`, and
# running from inside sessions. Only CPU-only analysis modules are imported.
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from sessions.playback import (
    LEFT_COLOR,
    RIGHT_COLOR,
    SKELETON,
    PlaybackClock,
    Recording,
    resolve_recording,
    time_label,
)
from workout.analysis.jumping_jack import (
    ExerciseState,
    JumpingJackConfig,
    JumpingJackCounter,
)
from workout.gesture import (
    MISSING_GRACE_SECONDS,
    REACQUIRE_HOLD_SECONDS,
    START_HOLD_SECONDS,
)
from workout.packets import KEYPOINT_CONFIDENCE, PosePacket
from workout.session import SESSION_SECONDS

WINDOW = "PoseSense - Jumping Jack Debug (read-only replay)"
WIDTH, HEIGHT = 1500, 960
BG, PANEL = (22, 19, 17), (34, 30, 27)
TEXT, MUTED = (238, 236, 231), (174, 166, 156)
GREEN, BLUE, RED = (139, 220, 123), (246, 183, 91), (108, 106, 249)
YELLOW, BORDER = (102, 218, 247), (69, 60, 52)
REQUIRED = (5, 6, 9, 10, 15, 16)
JOINT_NAMES = (
    "L shoulder [5]",
    "R shoulder [6]",
    "L wrist [9]",
    "R wrist [10]",
    "L ankle [15]",
    "R ankle [16]",
)


@dataclass(frozen=True)
class Control:
    field: str
    label: str
    explanation: str
    step: float
    minimum: float
    maximum: float


CONTROLS = (
    Control("confidence", "Joint confidence", "all six joints >=", 0.05, 0, 1),
    Control(
        "closed_ankle_ratio",
        "Closed feet ratio",
        "ankle / shoulder width <=",
        0.05,
        0,
        10,
    ),
    Control(
        "open_ankle_ratio", "Open feet ratio", "ankle / shoulder width >=", 0.05, 0, 10
    ),
    Control("wrist_margin", "Wrist margin", "fraction of body span", 0.01, 0, 2),
    Control(
        "confirm_seconds",
        "Confirmation (s)",
        "matching posture duration >=",
        0.01,
        0,
        5,
    ),
    Control(
        "max_sample_gap", "Max sample gap (s)", "larger gap resets phase", 0.05, 0, 10
    ),
    Control(
        "minimum_shoulder_width",
        "Min shoulder width",
        "normalized x width >=",
        0.001,
        0.000001,
        1,
    ),
)


@dataclass
class Dataset:
    recording: Recording
    generations: np.ndarray
    selected_ids: np.ndarray
    metadata: dict
    inferred_generations: bool = False

    @classmethod
    def load(cls, path: Path) -> Dataset:
        recording = Recording.load(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = (
                json.loads(str(archive["metadata_json"].item()))
                if "metadata_json" in archive
                else {}
            )
            if not isinstance(metadata, dict):
                raise TypeError("metadata_json must contain a JSON object")
            generations = (
                archive["generations"].astype(np.int64)
                if "generations" in archive
                else None
            )
            if "final_rep_count" in archive:
                metadata.setdefault(
                    "final_rep_count", int(archive["final_rep_count"].item())
                )
        n = len(recording.poses)
        if np.any(np.diff(recording.frame_indices) <= 0):
            raise ValueError("Source frame indices must be strictly increasing")
        if np.any(recording.present & (recording.track_ids < 0)):
            raise ValueError(
                "Present samples must have a nonnegative observed track ID"
            )
        selected = np.empty(n, dtype=np.int64)
        # The recorder uses -1 for absent observations, not for the selected ID.
        tid = metadata.get("selected_track_id_at_start")
        if tid is None:
            observed = recording.track_ids[recording.present]
            tid = int(observed[0]) if len(observed) else -1
        for i in range(n):
            if recording.present[i]:
                tid = int(recording.track_ids[i])
            selected[i] = tid
        inferred = generations is None
        if inferred:
            changed = (recording.present[1:] != recording.present[:-1]) | (
                selected[1:] != selected[:-1]
            )
            generations = np.r_[1, 1 + np.cumsum(changed)]
        if generations.shape != (n,):
            raise ValueError("generations must have one value per sample")
        if np.any(generations < 0) or np.any(np.diff(generations) < 0):
            raise ValueError("generations must be nonnegative and monotonic")
        return cls(recording, generations, selected, metadata, inferred)


@dataclass
class Metrics:
    valid: np.ndarray
    width: float
    span: float
    ratio: float
    left_delta: float
    right_delta: float
    margin: float
    invalid: str

    @property
    def geometry_ok(self) -> bool:
        return not self.invalid


def measure(pose: np.ndarray, present: bool, cfg: JumpingJackConfig) -> Metrics:
    """Explain exactly the operands used by classify; do not add an xy-range gate."""
    valid = np.isfinite(pose).all(axis=1) & (pose[:, 2] >= cfg.confidence)
    # Keep float32 expression ordering identical to the analysis implementation.
    width = abs(float(pose[5, 0] - pose[6, 0]))
    span = float((pose[15, 1] + pose[16, 1] - pose[5, 1] - pose[6, 1]) / 2)
    ratio = abs(float(pose[15, 0] - pose[16, 0])) / width if width > 0 else math.nan
    left = float(pose[9, 1] - pose[5, 1])
    right = float(pose[10, 1] - pose[6, 1])
    if not present:
        invalid = "Selected person missing"
    elif not valid[list(REQUIRED)].all():
        bad = [name for name, k in zip(JOINT_NAMES, REQUIRED) if not valid[k]]
        invalid = "Invalid required joints: " + ", ".join(bad)
    elif width < cfg.minimum_shoulder_width:
        invalid = "Shoulder width below minimum"
    elif span <= 0:
        invalid = "Vertical shoulder-to-ankle span must be > 0"
    else:
        invalid = ""
    return Metrics(
        valid, width, span, ratio, left, right, cfg.wrist_margin * span, invalid
    )


def gates(
    m: Metrics, cfg: JumpingJackConfig
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Left wrist, right wrist, feet; both sets require all geometry gates too."""
    closed = (
        m.left_delta > m.margin,
        m.right_delta > m.margin,
        m.ratio <= cfg.closed_ankle_ratio,
    )
    opened = (
        m.left_delta < -m.margin,
        m.right_delta < -m.margin,
        m.ratio >= cfg.open_ankle_ratio,
    )
    return closed, opened


class InspectCounter(JumpingJackCounter):
    """Capture the actual classifier return without replacing its logic."""

    posture: ExerciseState | None = None

    def classify(self, pose):
        self.posture = super().classify(pose)
        return self.posture


@dataclass
class Diagnostic:
    metrics: Metrics
    candidate: str
    state: str
    pending: str
    held: float
    gap: float
    reps: int
    counted: bool
    reset: str
    reason: str


def replay(data: Dataset, cfg: JumpingJackConfig) -> list[Diagnostic]:
    """Deterministic all-samples replay, independent of the playback clock."""
    r = data.recording
    counter = InspectCounter(cfg)
    result = []
    for i, pose in enumerate(r.poses):
        timestamp = float(r.timestamps[i])
        gap = (
            timestamp - counter.last_timestamp
            if counter.last_timestamp is not None
            else 0.0
        )
        changed = (
            int(data.generations[i]) != counter.generation
            or int(data.selected_ids[i]) != counter.track_id
        )
        m = measure(pose, bool(r.present[i]), cfg)
        resets = []
        if i == 0:
            resets.append("Session start")
        elif changed:
            resets.append("Generation / track changed")
        elif not 0 <= gap <= cfg.max_sample_gap:
            resets.append("Sample gap exceeded")
        if m.invalid:
            resets.append(m.invalid)
        previous_reps, previous_state = counter.rep_count, counter.state.value
        counter.posture = None
        counter.update(
            PosePacket(
                int(r.frame_indices[i]),
                timestamp,
                int(data.selected_ids[i]),
                pose,
                bool(r.present[i]),
                int(data.generations[i]),
                counter.rep_count,
            )
        )
        candidate = (
            counter.posture.value
            if counter.posture
            else "INVALID"
            if m.invalid
            else "DEAD BAND"
        )
        pending = counter.pending.value if counter.pending else "NONE"
        held = timestamp - counter.pending_since if counter.pending else 0.0
        counted = counter.rep_count > previous_reps
        if m.invalid:
            reason = m.invalid + "; phase reset to UNKNOWN."
        elif candidate == "DEAD BAND":
            reason = "Neither posture passes all 3 gates. Stable phase kept; confirmation cancelled."
        elif held < cfg.confirm_seconds:
            reason = f"Hold {candidate} for {cfg.confirm_seconds - held:.3f}s more without a failed sample."
        elif counter.state == ExerciseState.UNKNOWN:
            reason = "OPEN cannot arm a new cycle. Confirm CLOSED first."
        elif counted:
            reason = "Confirmed OPEN -> CLOSED: one repetition added."
        elif counter.state.value != previous_state or changed:
            reason = f"Confirmed {counter.state.value}; next confirm {'OPEN' if counter.state == ExerciseState.CLOSED else 'CLOSED to count'}."
        else:
            reason = f"Still {counter.state.value}; waiting for {'OPEN' if counter.state == ExerciseState.CLOSED else 'CLOSED to count'}."
        result.append(
            Diagnostic(
                m,
                candidate,
                counter.state.value,
                pending,
                held,
                gap,
                counter.rep_count,
                counted,
                "; ".join(resets),
                reason,
            )
        )
    return result


def event_indices(rows: list[Diagnostic]) -> np.ndarray:
    """Navigate phase/validity/continuity changes, rather than every failed sample."""
    events = [0]
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if (
            b.counted
            or b.state != a.state
            or b.candidate != a.candidate
            or b.reset != a.reset
        ):
            events.append(i)
    return np.asarray(events, dtype=np.int64)


def summarize(data: Dataset, rows: list[Diagnostic], cfg: JumpingJackConfig) -> dict:
    r = data.recording
    valid = [row for row in rows if row.metrics.geometry_ok]
    near_open = [row for row in valid if row.metrics.ratio >= cfg.open_ankle_ratio]
    raised_arms = [row for row in valid if all(gates(row.metrics, cfg)[1][:2])]
    return {
        "recording": str(r.path),
        "config_from_current_source_with_in_memory_overrides": asdict(cfg),
        "saved_live_final_reps": data.metadata.get(
            "final_rep_count", int(r.rep_counts[-1])
        ),
        "offline_all_samples_reps": rows[-1].reps,
        "samples": len(rows),
        "present_samples": int(r.present.sum()),
        "missing_person_samples": int((~r.present).sum()),
        "candidate_samples": dict(Counter(row.candidate for row in rows)),
        "invalid_required_joint_samples_when_present": {
            name: sum(
                bool(r.present[i]) and not row.metrics.valid[k]
                for i, row in enumerate(rows)
            )
            for name, k in zip(JOINT_NAMES, REQUIRED)
        },
        "open_feet_samples_with_valid_geometry": len(near_open),
        "of_those_samples_failing_open_left_wrist": sum(
            not gates(row.metrics, cfg)[1][0] for row in near_open
        ),
        "of_those_samples_failing_open_right_wrist": sum(
            not gates(row.metrics, cfg)[1][1] for row in near_open
        ),
        "both_wrists_open_samples_with_valid_geometry": len(raised_arms),
        "of_those_samples_failing_open_feet": sum(
            not gates(row.metrics, cfg)[1][2] for row in raised_arms
        ),
        "longest_continuous_candidate_hold_seconds": {
            posture: round(
                max((row.held for row in rows if row.pending == posture), default=0.0),
                4,
            )
            for posture in ("CLOSED", "OPEN")
        },
        "sample_gaps_over_limit": sum(row.gap > cfg.max_sample_gap for row in rows[1:]),
        "replay_rep_times_seconds": [
            round(float(r.timestamps[i]), 4)
            for i, row in enumerate(rows)
            if row.counted
        ],
        "live_analysis_dropped_packets": data.metadata.get(
            "analysis_dropped_packets", "not recorded"
        ),
        "live_analysis_processed_packets": data.metadata.get(
            "analysis_processed_packets", "not recorded"
        ),
        "live_analysis_error": data.metadata.get("analysis_error", "not recorded"),
        "live_analysis_forced_shutdown": data.metadata.get(
            "analysis_forced_shutdown", "not recorded"
        ),
        "generations_inferred": data.inferred_generations,
        "caveat": "Offline replay uses every saved sample, current source thresholds and its own accepted total. Live per-packet delivery/status history was not saved. Failure counts overlap; they are not counts of missed reps.",
    }


def label(image, message, x, y, color=TEXT, scale=0.49, max_width=None):
    message = str(message)
    limit = max_width if max_width is not None else image.shape[1] - x - 12
    if cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > limit:
        while (
            message
            and cv2.getTextSize(message + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][
                0
            ]
            > limit
        ):
            message = message[:-1]
        message += "..."
    cv2.putText(
        image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def wrapped(image, message, x, y, width, color=TEXT, scale=0.49, lines=3):
    words, line, row = str(message).split(), "", 0
    for word in words:
        trial = f"{line} {word}".strip()
        if (
            cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width
            and line
        ):
            label(image, line, x, y + row * 21, color, scale, width)
            row += 1
            line = word
            if row >= lines:
                return
        else:
            line = trial
    label(image, line, x, y + row * 21, color, scale, width)


def panel(image, x, y, w, h, title):
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL, -1)
    label(image, title, x + 16, y + 25, MUTED, 0.51, w - 32)


def number(value, places=3):
    return f"{value:.{places}f}" if math.isfinite(value) else "N/A"


def state_color(state):
    return {"CLOSED": GREEN, "OPEN": BLUE, "DEAD BAND": YELLOW, "INVALID": RED}.get(
        state, MUTED
    )


def dashed(image, a, b, color):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    for start in np.arange(0, length, 12):
        p = np.rint(a + (b - a) * start / length).astype(int)
        q = np.rint(a + (b - a) * min(start + 6, length) / length).astype(int)
        cv2.line(image, tuple(p), tuple(q), color, 1, cv2.LINE_AA)


def draw_pose(image, data, index, row, cfg, guides, indices):
    r, m = data.recording, row.metrics
    panel(
        image,
        16,
        82,
        744,
        406,
        "SKELETON   |   anatomical L: cyan   R: orange   |   y grows downward",
    )
    # Draw into a clipped viewport: finite off-camera points are not invalidated
    # in analysis, but must never spill into other panels or overflow OpenCV ints.
    view = image[118:452, 32:744]
    vh, vw = view.shape[:2]
    scale = min(vw / r.camera_width, vh / r.camera_height)
    sx, sy = r.camera_width * scale, r.camera_height * scale
    ox, oy = (vw - sx) / 2, (vh - sy) / 2

    def xy(x, y):
        return (
            int(np.clip(round(x * sx + ox), -10000, 10000)),
            int(np.clip(round(y * sy + oy), -10000, 10000)),
        )

    cv2.rectangle(view, xy(0, 0), xy(1, 1), BORDER, 1)
    pose = r.poses[index]
    finite = np.isfinite(pose[:, :2]).all(axis=1) & bool(r.present[index])
    points = [xy(*pose[k, :2]) if finite[k] else None for k in range(17)]
    if guides and r.present[index]:
        if math.isfinite(m.span) and m.span > 0:
            for shoulder, side in ((5, "L"), (6, "R")):
                if not finite[shoulder]:
                    continue
                x, y = map(float, pose[shoulder, :2])
                for sign, color, target in (
                    (-1, BLUE, "OPEN <"),
                    (1, GREEN, "CLOSED >"),
                ):
                    level = y + sign * m.margin
                    dashed(view, xy(x - 0.12, level), xy(x + 0.12, level), color)
                    other = 6 if shoulder == 5 else 5
                    on_left = not finite[other] or x <= float(pose[other, 0])
                    px, py = xy(x - 0.12 if on_left else x + 0.12, level)
                    caption_x = max(0, px - 76) if on_left else min(vw - 85, px + 5)
                    label(view, f"{side} {target}", caption_x, py - 5, color, 0.34)
        if math.isfinite(m.width):
            y = float((pose[5, 1] + pose[6, 1]) / 2)
            if math.isfinite(y):
                cv2.line(
                    view, xy(float(pose[5, 0]), y), xy(float(pose[6, 0]), y), YELLOW, 2
                )
        if math.isfinite(m.ratio) and m.width > 0:
            mid = float((pose[15, 0] + pose[16, 0]) / 2)
            for ratio, color, yy in (
                (cfg.closed_ankle_ratio, GREEN, 0.94),
                (cfg.open_ankle_ratio, BLUE, 0.99),
            ):
                a, b = (
                    xy(mid - ratio * m.width / 2, yy),
                    xy(mid + ratio * m.width / 2, yy),
                )
                cv2.line(view, a, b, color, 2)
                for px, py in (a, b):
                    cv2.line(view, (px, py - 4), (px, py + 4), color, 1)
    for a, b in SKELETON:
        if points[a] is not None and points[b] is not None:
            color = (
                LEFT_COLOR
                if a % 2 and b % 2
                else RIGHT_COLOR
                if not a % 2 and not b % 2
                else MUTED
            )
            if not m.valid[a] or not m.valid[b]:
                color = BORDER
            cv2.line(view, points[a], points[b], color, 2, cv2.LINE_AA)
    for k, point in enumerate(points):
        if point is None:
            continue
        color = LEFT_COLOR if k % 2 else RIGHT_COLOR
        cv2.circle(
            view,
            point,
            5 if k in REQUIRED else 3,
            color if m.valid[k] else RED,
            -1 if m.valid[k] else 1,
            cv2.LINE_AA,
        )
        if indices or k in REQUIRED:
            label(view, str(k), point[0] + 7, point[1] - 5, TEXT, 0.36)
    if not r.present[index]:
        label(view, "SELECTED PERSON MISSING", 130, 160, RED, 0.75)
        label(view, "No interpolated or invented skeleton", 130, 188, MUTED, 0.5)
    label(
        image,
        "Dashed: per-shoulder wrist limits. Bottom bars: closed/open target foot spans.",
        32,
        475,
        MUTED,
        0.45,
        710,
    )


def gauge(image, x, y, width, low, high, limits, markers):
    def px(value):
        return x + round(width * np.clip((value - low) / (high - low), 0, 1))

    cv2.line(image, (x, y), (x + width, y), BORDER, 5)
    for j, (value, color, caption) in enumerate(limits):
        location = px(value)
        cv2.line(image, (location, y - 8), (location, y + 8), color, 2)
        label(
            image,
            caption,
            x if j == 0 else x + width - 112,
            y + 25,
            color,
            0.39,
            130,
        )
    for j, (value, color, caption) in enumerate(markers):
        if math.isfinite(value):
            location = px(value)
            cv2.circle(image, (location, y), 5 if j == 0 else 3, color, -1)
            label(
                image,
                caption,
                location - 4,
                y - 12 if j == 0 else y + 13,
                color,
                0.4,
                80,
            )


def draw_metrics(image, data, index, row, cfg):
    m, pose = row.metrics, data.recording.poses[index]
    panel(
        image,
        16,
        498,
        744,
        128,
        "BODY-RELATIVE MEASUREMENTS   |   thresholds follow the current pose",
    )
    left = m.left_delta / m.span if m.span > 0 else math.nan
    right = m.right_delta / m.span if m.span > 0 else math.nan
    label(
        image,
        f"Wrist dY / span:   L {number(left)}   R {number(right)}",
        32,
        546,
        TEXT,
        0.46,
    )
    extent = max(1.0, cfg.wrist_margin * 1.4)
    gauge(
        image,
        405,
        545,
        325,
        -extent,
        extent,
        [
            (-cfg.wrist_margin, BLUE, f"OPEN < {-cfg.wrist_margin:g}"),
            (cfg.wrist_margin, GREEN, f"CLOSED > {cfg.wrist_margin:g}"),
        ],
        [(left, LEFT_COLOR, "L"), (right, RIGHT_COLOR, "R")],
    )
    label(image, f"Feet / shoulder width:   {number(m.ratio)}", 32, 596, TEXT, 0.46)
    gauge(
        image,
        405,
        595,
        325,
        0,
        max(4.0, cfg.open_ankle_ratio * 1.4, cfg.closed_ankle_ratio * 1.4),
        [
            (cfg.closed_ankle_ratio, GREEN, f"C <= {cfg.closed_ankle_ratio:g}"),
            (cfg.open_ankle_ratio, BLUE, f"O >= {cfg.open_ankle_ratio:g}"),
        ],
        [(m.ratio, TEXT, "F")],
    )
    panel(
        image,
        16,
        636,
        744,
        200,
        "REQUIRED JOINTS   |   low confidence or NaN in ANY of these resets the phase",
    )
    label(image, "Joint", 32, 683, MUTED, 0.43)
    label(image, "Confidence", 211, 683, MUTED, 0.43)
    label(image, "Saved normalized x, y", 366, 683, MUTED, 0.43)
    label(image, "Gate", 601, 683, MUTED, 0.43)
    for j, (k, name) in enumerate(zip(REQUIRED, JOINT_NAMES)):
        y = 705 + j * 22
        color = LEFT_COLOR if k % 2 else RIGHT_COLOR
        label(image, name, 32, y, color, 0.44)
        confidence = float(pose[k, 2])
        label(image, number(confidence), 211, y, TEXT, 0.44)
        cv2.line(image, (273, y - 5), (343, y - 5), BORDER, 4)
        if math.isfinite(confidence):
            cv2.line(
                image,
                (273, y - 5),
                (273 + round(70 * np.clip(confidence, 0, 1)), y - 5),
                GREEN if m.valid[k] else RED,
                4,
            )
        threshold_x = 273 + round(70 * cfg.confidence)
        cv2.line(image, (threshold_x, y - 11), (threshold_x, y + 1), TEXT, 1)
        label(
            image,
            f"{number(float(pose[k, 0]))}, {number(float(pose[k, 1]))}",
            366,
            y,
            TEXT,
            0.44,
        )
        ok = bool(m.valid[k]) and bool(data.recording.present[index])
        status = (
            "PASS"
            if ok
            else "NaN xy"
            if not np.isfinite(pose[k, :2]).all()
            else "LOW CONF"
        )
        label(image, status, 601, y, GREEN if ok else RED, 0.44)
    label(
        image,
        "Elbows, knees and nose are NOT gates for jumping-jack counting.",
        32,
        830,
        MUTED,
        0.4,
    )


def draw_controls(image, cfg, selected):
    panel(
        image,
        780,
        82,
        704,
        274,
        "THRESHOLDS   |   loaded from workout/analysis/jumping_jack.py",
    )
    for j, control in enumerate(CONTROLS):
        top, y = 126 + j * 28, 146 + j * 28
        if j == selected:
            cv2.rectangle(image, (792, top), (1474, top + 26), (62, 52, 41), -1)
        label(image, control.label, 796, y, TEXT, 0.46, 208)
        label(image, f"{getattr(cfg, control.field):.3f}", 1012, y, YELLOW, 0.49, 77)
        label(image, control.explanation, 1100, y, MUTED, 0.4, 290)
        for x, symbol in ((1406, "-"), (1443, "+")):
            cv2.rectangle(image, (x, top + 2), (x + 29, top + 24), BORDER, 1)
            label(image, symbol, x + 8, y - 2, TEXT, 0.47)
    label(
        image,
        "Click row + ,/. or -/+ buttons: change     B: restore source defaults",
        796,
        346,
        MUTED,
        0.42,
    )


def draw_gates(image, row, cfg):
    panel(
        image,
        780,
        366,
        704,
        179,
        "POSTURE GATES   |   all 3 in a column must pass simultaneously",
    )
    label(image, "Measurement", 796, 411, MUTED, 0.44)
    label(image, "CLOSED", 1160, 411, GREEN, 0.46)
    label(image, "OPEN", 1340, 411, BLUE, 0.46)
    closed, opened = gates(row.metrics, cfg)
    names = (
        "Left wrist vs LEFT shoulder",
        "Right wrist vs RIGHT shoulder",
        "Ankle horizontal spread",
    )
    for j, name in enumerate(names):
        y = 435 + j * 25
        label(image, name, 796, y, TEXT, 0.48, 351)
        for x, passed, target in (
            (1160, closed[j], "down" if j < 2 else "narrow"),
            (1340, opened[j], "up" if j < 2 else "wide"),
        ):
            # Independent comparisons aren't evidence of a valid full posture.
            status = (
                "N/A" if not row.metrics.geometry_ok else "PASS" if passed else "FAIL"
            )
            label(
                image,
                f"{status} {target}",
                x,
                y,
                MUTED if status == "N/A" else GREEN if passed else RED,
                0.44,
                136,
            )
    label(
        image,
        f"Raw posture: {row.candidate}",
        796,
        516,
        state_color(row.candidate),
        0.56,
    )
    label(
        image,
        "Strict wrist inequalities; equality does not pass.",
        796,
        536,
        MUTED,
        0.41,
    )


def draw_state(image, data, index, row, cfg):
    r, meta, m = data.recording, data.metadata, row.metrics
    panel(
        image,
        780,
        555,
        704,
        281,
        "COUNTER / TIMING   |   CLOSED -> OPEN -> CLOSED = +1 rep",
    )
    label(image, f"Stable: {row.state}", 796, 610, state_color(row.state), 0.67)
    label(image, f"Pending: {row.pending}", 1140, 610, TEXT, 0.55)
    held = min(row.held, cfg.confirm_seconds)
    fraction = (
        min(1.0, row.held / cfg.confirm_seconds)
        if cfg.confirm_seconds > 0
        else float(row.pending != "NONE")
    )
    label(
        image,
        f"Confirmation: {held:.3f} / {cfg.confirm_seconds:.3f} s",
        796,
        639,
        TEXT,
        0.47,
    )
    cv2.rectangle(image, (1160, 627), (1462, 641), BORDER, -1)
    cv2.rectangle(
        image,
        (1160, 627),
        (1160 + round(302 * fraction), 641),
        state_color(row.pending),
        -1,
    )
    label(
        image,
        f"Sample gap: {row.gap:.3f}s <= {cfg.max_sample_gap:.3f}s",
        796,
        666,
        GREEN if row.gap <= cfg.max_sample_gap else RED,
        0.47,
    )
    label(
        image,
        f"Generation {int(data.generations[index])}   Track {int(data.selected_ids[index])}",
        1160,
        666,
        MUTED,
        0.46,
    )
    label(
        image,
        f"Shoulder width {number(m.width)} >= {cfg.minimum_shoulder_width:.3f}  |  Body span {number(m.span)} > 0",
        796,
        691,
        TEXT,
        0.46,
    )
    wrapped(image, row.reason, 796, 720, 668, GREEN if row.counted else YELLOW, 0.5, 2)
    reset = row.reset or "none (dead band only cancels pending confirmation)"
    label(image, "Reset: " + reset, 796, 767, RED if row.reset else MUTED, 0.43, 668)
    label(
        image,
        f"At this sample: saved live {r.rep_counts[index]} reps  |  replay {row.reps} reps",
        796,
        791,
        TEXT,
        0.47,
    )
    dropped, processed = (
        meta.get("analysis_dropped_packets", "?"),
        meta.get("analysis_processed_packets", "?"),
    )
    health = (
        "ERROR"
        if meta.get("analysis_error")
        else "FORCED STOP"
        if meta.get("analysis_forced_shutdown")
        else "no error reported"
    )
    label(
        image,
        f"Live worker (whole file): drops {dropped}; processed {processed}; {health}",
        796,
        818,
        MUTED,
        0.42,
        668,
    )


def timeline_image(data, rows):
    """Cache the full time-based strips instead of redrawing N samples per tick."""
    width = 1296
    duration = data.recording.duration
    seconds = np.linspace(0, duration, width)
    indices = np.clip(
        np.searchsorted(data.recording.timestamps, seconds, side="right") - 1,
        0,
        len(rows) - 1,
    )
    strip = np.full((44, width, 3), BG, np.uint8)
    for x, i in enumerate(indices):
        strip[0:10, x] = state_color(rows[i].candidate)
        strip[15:25, x] = state_color(rows[i].state)
    for i, row in enumerate(rows):
        x = (
            min(
                width - 1,
                round(float(data.recording.timestamps[i]) / duration * (width - 1)),
            )
            if duration
            else 0
        )
        if row.counted:
            cv2.line(strip, (x, 29), (x, 43), GREEN, 2)
        elif row.reset and (i == 0 or row.reset != rows[i - 1].reset):
            cv2.line(strip, (x, 30), (x, 38), RED, 1)
    return strip


def draw_help(image, data):
    cv2.rectangle(image, (110, 92), (1390, 837), (37, 33, 29), -1)
    cv2.rectangle(image, (110, 92), (1390, 837), BLUE, 1)
    label(image, "HOW TO READ THIS DEBUGGER   |   H closes help", 140, 132, TEXT, 0.72)
    threshold = data.metadata.get("keypoint_confidence_threshold", "not recorded")
    lines = [
        "1. Green = CLOSED; blue = OPEN; yellow = dead band; red = invalid / reset.",
        "2. Six joints must be finite and meet confidence: both shoulders, wrists and ankles.",
        "3. Width = abs(left shoulder x - right shoulder x), in normalized image x.",
        "   Span = mean ankle y - mean shoulder y. It must be > 0 (no other span threshold).",
        "4. Feet ratio = abs(left ankle x - right ankle x) / shoulder width (NOT Euclidean distance).",
        "5. Wrist dY = wrist y - its SAME-SIDE shoulder y. Margin = wrist_margin * span.",
        "   CLOSED: BOTH dY > +margin and feet ratio <= closed limit.",
        "   OPEN: BOTH dY < -margin and feet ratio >= open limit. Equality fails for wrists.",
        "6. The same raw posture must persist for confirm_seconds. A dead-band sample cancels",
        "   pending confirmation but keeps stable state. An invalid joint / loss resets to UNKNOWN.",
        "7. UNKNOWN must first confirm CLOSED. OPEN alone cannot start a repetition.",
        "   Complete CLOSED -> OPEN -> CLOSED. Track/generation changes and large gaps reset phase.",
        "8. Dashed shoulder-relative lines show wrist limits. Bottom foot bars are target SPANS",
        "   centered at the current ankle midpoint, not fixed left/right foot destinations.",
        "",
        f"Selection only (workout/gesture.py): confidence >= {KEYPOINT_CONFIDENCE:g}; RIGHT wrist y < nose y;",
        f"RIGHT elbow y < RIGHT shoulder y. Start hold {START_HOLD_SECONDS:g}s; reacquire hold {REACQUIRE_HOLD_SECONDS:g}s; missing grace {MISSING_GRACE_SECONDS:g}s.",
        f"Session target: {SESSION_SECONDS:g}s. These are NOT jumping-jack gates. Pre-start poses and other",
        "candidate people are not saved, so their selection/reacquisition holds cannot be reconstructed.",
        "",
        f"Recording xy masking confidence: {threshold}. Lowering it here cannot restore NaN coordinates.",
        "Replay uses the actual current counter with every saved sample and its own accepted total.",
        "Original config and live per-packet delivery/acknowledgments were NOT saved; totals may differ.",
        "All seven threshold edits are WHAT-IF ONLY, in memory. B restores current source defaults.",
        "F: files in this folder; Up/Down or W/S: select; Enter: load paused; Esc: cancel. Edits retained.",
        "The console summary lists overlapping blocker sample counts, NOT a count of missed exercises.",
    ]
    for i, line in enumerate(lines):
        label(
            image, line, 140, 171 + i * 24, MUTED if i in (14, 19) else TEXT, 0.51, 1216
        )


def render(
    data,
    rows,
    cfg,
    index,
    clock,
    selected=0,
    guides=True,
    indices=False,
    help_open=False,
    strip=None,
):
    image = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    r, row = data.recording, rows[index]
    custom = cfg != JumpingJackConfig()
    label(image, "JUMPING JACK DEBUG", 20, 28, TEXT, 0.72)
    label(
        image,
        "WHAT-IF CONFIG (memory only)" if custom else "CURRENT SOURCE DEFAULTS",
        325,
        28,
        YELLOW if custom else GREEN,
        0.51,
    )
    final = data.metadata.get("final_rep_count", int(r.rep_counts[-1]))
    label(
        image,
        f"Saved live final: {final}   |   Offline replay final: {rows[-1].reps}",
        900,
        28,
        TEXT,
        0.58,
    )
    state = "PLAYING" if clock.playing else "PAUSED"
    label(
        image,
        f"{state}  {time_label(clock.position)} / {time_label(r.duration)}   {clock.speed:g}x"
        f"   sample {index + 1}/{len(rows)}   frame {r.frame_indices[index]}   {'LOOP' if clock.loop else ''}",
        20,
        53,
        TEXT,
        0.48,
    )
    label(image, r.path.name, 900, 53, MUTED, 0.44, 575)
    label(
        image,
        "Offline = every saved sample; live worker timing and original config were not recorded. H: formulas / limitations",
        20,
        72,
        MUTED,
        0.4,
    )
    draw_pose(image, data, index, row, cfg, guides, indices)
    draw_metrics(image, data, index, row, cfg)
    draw_controls(image, cfg, selected)
    draw_gates(image, row, cfg)
    draw_state(image, data, index, row, cfg)
    if strip is None:
        strip = timeline_image(data, rows)
    image[850:894, 180:1476] = strip
    label(image, "Raw posture", 20, 859, MUTED, 0.41)
    label(image, "Stable phase", 20, 875, MUTED, 0.41)
    label(image, "Rep / reset", 20, 891, MUTED, 0.41)
    progress = clock.position / r.duration if r.duration else 0
    x = 180 + round(1295 * np.clip(progress, 0, 1))
    cv2.line(image, (x, 847), (x, 897), TEXT, 2)
    label(
        image,
        "Space: play/pause   [/]: step   A/D: +/-5s   N/P: events   -/+: speed   R: restart   L: loop   Click strips: seek",
        20,
        921,
        TEXT,
        0.48,
    )
    label(
        image,
        "F: files   T: guides   I: indices   H: help   Click threshold + ,/.: tune   B: defaults   Q/Esc: quit   Read-only",
        20,
        946,
        MUTED,
        0.46,
    )
    if help_open:
        draw_help(image, data)
    return image


class DebugPlayer:
    def __init__(self, data, cfg, args):
        self.data, self.cfg, self.args = data, cfg, args
        self.rows = replay(data, cfg)
        self.strip = timeline_image(data, self.rows)
        self.events = event_indices(self.rows)
        self.clock = PlaybackClock(data.recording.duration, args.speed, args.loop)
        self.clock.seek(args.start)
        self.clock.playing = not args.paused
        self.manual_frame = None
        self.selected, self.guides, self.indices, self.help_open = (
            0,
            True,
            args.indices,
            False,
        )
        scale = min(1.0, args.max_width / WIDTH, args.max_height / HEIGHT)
        self.size = (round(WIDTH * scale), round(HEIGHT * scale))
        self.picker_open = False
        self.picker_files = []
        self.picker_index = 0
        self.picker_error = ""
        self.picker_resume = False
        self.loading_path = None

    def open_picker(self):
        self.picker_resume = self.clock.playing
        self.clock.playing = False
        self.help_open = False
        self.picker_open = True
        self.picker_error = ""
        self.picker_index = 0
        current = self.data.recording.path.resolve()
        self.picker_folder = current.parent
        self.picker_files = []
        try:
            self.picker_files = sorted(
                (
                    p
                    for p in self.picker_folder.iterdir()
                    if p.is_file() and p.suffix.lower() == ".npz"
                ),
                key=lambda p: (p.name.casefold(), p.name),
            )
            if current in self.picker_files:
                self.picker_index = self.picker_files.index(current)
        except OSError as error:
            self.picker_error = f"Cannot list folder: {error}"

    def cancel_picker(self):
        self.picker_open = False
        self.loading_path = None
        # Rebase the wall clock so time spent choosing never advances playback.
        self.clock.seek(self.clock.position)
        self.clock.playing = self.picker_resume

    def picker_key(self, key):
        if key == 27:
            self.cancel_picker()
        elif key in (ord("w"), 2490368, 65362, 63232):  # Win32, Qt/X11, macOS
            self.picker_index = max(0, self.picker_index - 1)
        elif key in (ord("s"), 2621440, 65364, 63233):
            self.picker_index = min(
                max(0, len(self.picker_files) - 1), self.picker_index + 1
            )
        elif key in (10, 13) and self.picker_files:
            self.loading_path = self.picker_files[self.picker_index]
            self.picker_error = ""
        return True

    def load_pending(self):
        """Build all recording-dependent state before replacing anything visible."""
        path = self.loading_path
        if path is None:
            return
        self.loading_path = None
        try:
            data = Dataset.load(path)
            rows = replay(data, self.cfg)
            strip = timeline_image(data, rows)
            events = event_indices(rows)
            clock = PlaybackClock(
                data.recording.duration, self.clock.speed, self.clock.loop
            )
            clock.seek(0)
            clock.playing = False
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            EOFError,
            zipfile.BadZipFile,
            zlib.error,
            cv2.error,
        ) as error:
            self.picker_error = f"Cannot load {path.name}: {error}"
            return
        self.data, self.rows, self.strip, self.events, self.clock = (
            data,
            rows,
            strip,
            events,
            clock,
        )
        self.manual_frame = None
        self.picker_open = False
        self.picker_error = ""
        print(
            f"Loaded {path} -> {rows[-1].reps} offline reps; threshold edits retained (files unchanged)"
        )

    def draw_picker(self, image):
        cv2.rectangle(image, (90, 95), (1410, 840), BG, -1)
        cv2.rectangle(image, (90, 95), (1410, 840), BORDER, 2)
        label(image, "OPEN RECORDING", 120, 136, TEXT, 0.8)
        label(image, str(self.picker_folder), 120, 170, MUTED, 0.5, 1250)
        label(
            image,
            "Up/Down or W/S: select   Enter: load   Esc: cancel   Q: quit",
            120,
            207,
            TEXT,
            0.54,
        )
        label(
            image,
            "Starts paused at zero. Current thresholds, speed and display settings are kept.",
            120,
            237,
            MUTED,
            0.5,
        )
        page_size = 12
        start = (self.picker_index // page_size) * page_size
        current = self.data.recording.path.resolve()
        for row, path in enumerate(self.picker_files[start : start + page_size]):
            index = start + row
            y = 278 + row * 35
            selected = index == self.picker_index
            if selected:
                cv2.rectangle(image, (112, y - 23), (1385, y + 8), PANEL, -1)
            suffix = "  [current]" if path == current else ""
            label(
                image,
                f"{'>' if selected else ' '} {index + 1}. {path.name}{suffix}",
                124,
                y,
                GREEN if selected else TEXT,
                0.53,
                1240,
            )
        if not self.picker_files:
            label(
                image,
                "No NPZ files found. Esc returns to the current recording.",
                124,
                285,
                YELLOW,
                0.56,
            )
        label(
            image,
            f"{len(self.picker_files)} files | alphabetical | refreshed each time F opens this picker",
            120,
            718,
            MUTED,
            0.46,
        )
        if self.loading_path is not None:
            label(
                image,
                f"Loading and replaying: {self.loading_path.name}",
                120,
                758,
                YELLOW,
                0.56,
                1240,
            )
        elif self.picker_error:
            label(image, self.picker_error, 120, 758, RED, 0.49, 1240)
            label(
                image,
                "Previous recording is intact. Choose another file or press Esc.",
                120,
                795,
                MUTED,
                0.49,
            )

    @property
    def index(self):
        return (
            self.manual_frame
            if self.manual_frame is not None
            else self.data.recording.frame_at(self.clock.position)
        )

    def rebuild(self, cfg):
        self.clock.playing = False
        self.cfg = cfg
        self.rows = replay(self.data, cfg)
        self.strip, self.events = (
            timeline_image(self.data, self.rows),
            event_indices(self.rows),
        )
        print(
            f"WHAT-IF {asdict(cfg)} -> {self.rows[-1].reps} offline reps (files unchanged)"
        )

    def adjust(self, direction):
        control = CONTROLS[self.selected]
        value = float(
            np.clip(
                round(getattr(self.cfg, control.field) + direction * control.step, 6),
                control.minimum,
                control.maximum,
            )
        )
        self.rebuild(replace(self.cfg, **{control.field: value}))

    def seek_frame(self, index):
        self.manual_frame = max(0, min(len(self.rows) - 1, index))
        self.clock.seek(float(self.data.recording.timestamps[self.manual_frame]))
        self.clock.playing = False

    def mouse(self, event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN or self.help_open or self.picker_open:
            return
        x, y = x * WIDTH / self.size[0], y * HEIGHT / self.size[1]
        if 180 <= x <= 1476 and 845 <= y <= 898:
            self.clock.seek((x - 180) / 1295 * self.data.recording.duration)
            self.manual_frame = None
        elif 792 <= x <= 1474 and 126 <= y < 322:
            self.selected = int((y - 126) // 28)
            if 1406 <= x <= 1435:
                self.adjust(-1)
            elif 1443 <= x <= 1472:
                self.adjust(1)

    def key(self, key):
        if key in (ord("q"), ord("Q")):
            return False
        if 65 <= key <= 90:
            key += 32
        if self.picker_open:
            return self.picker_key(key)
        if key == 27:
            return False
        if key == ord("f"):
            self.open_picker()
        elif key == ord("h"):
            self.help_open = not self.help_open
            self.clock.playing = False
        elif self.help_open:
            return True
        elif key == ord(" "):
            if (
                not self.clock.playing
                and self.clock.position >= self.data.recording.duration
            ):
                self.clock.seek(0)
            self.clock.playing = not self.clock.playing
            self.manual_frame = None
        elif key in (ord("["), ord("]")):
            self.seek_frame(self.index + (-1 if key == ord("[") else 1))
        elif key in (ord("n"), ord("p")):
            found = (
                self.events[self.events > self.index]
                if key == ord("n")
                else self.events[self.events < self.index]
            )
            if len(found):
                self.seek_frame(int(found[0] if key == ord("n") else found[-1]))
        elif key in (ord("a"), ord("d")):
            self.clock.seek(self.clock.position + (-5 if key == ord("a") else 5))
            self.manual_frame = None
        elif key == ord("r"):
            self.clock.seek(0)
            self.clock.playing, self.manual_frame = True, None
        elif key in (ord("+"), ord("="), ord("-"), ord("_")):
            factor = 0.5 if key in (ord("-"), ord("_")) else 2
            self.clock.speed = min(8.0, max(0.125, self.clock.speed * factor))
        elif key == ord("l"):
            self.clock.loop = not self.clock.loop
        elif key == ord("t"):
            self.guides = not self.guides
        elif key == ord("i"):
            self.indices = not self.indices
        elif key in (ord(","), ord(".")):
            self.adjust(-1 if key == ord(",") else 1)
        elif key == ord("b"):
            self.rebuild(JumpingJackConfig())
        return True

    def frame(self):
        frame = render(
            self.data,
            self.rows,
            self.cfg,
            self.index,
            self.clock,
            self.selected,
            self.guides,
            self.indices,
            self.help_open,
            self.strip,
        )
        if self.picker_open:
            self.draw_picker(frame)
        return (
            cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            if self.size != (WIDTH, HEIGHT)
            else frame
        )

    def play(self):
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, self.mouse)
            self.clock.seek(self.args.start)
            while True:
                self.clock.tick()
                cv2.imshow(WINDOW, self.frame())
                key = cv2.waitKeyEx(10)
                self.clock.tick()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                # Enter schedules loading. Render/flush its indicator on the next
                # iteration before the synchronous archive validation and replay.
                if self.loading_path is not None:
                    if key in (ord("q"), ord("Q")):
                        break
                    if key == 27:
                        self.cancel_picker()
                    else:
                        self.load_pending()
                elif not self.key(key):
                    break
        finally:
            cv2.destroyAllWindows()


def self_test():
    """Headless regression checks, kept here so no other project file is added."""
    cfg = JumpingJackConfig()

    def pose(opened=False):
        p = np.full((17, 3), 0.5, dtype=np.float32)
        p[:, 2] = 0.99
        p[5, :2], p[6, :2] = (0.4, 0.4), (0.6, 0.4)
        p[9, :2], p[10, :2] = (
            (0.4, 0.1 if opened else 0.65),
            (0.6, 0.1 if opened else 0.65),
        )
        p[15, :2], p[16, :2] = (
            ((0.25, 0.9), (0.75, 0.9)) if opened else ((0.45, 0.9), (0.55, 0.9))
        )
        return p

    def dataset(poses, timestamps=None, generations=None, present=None):
        n = len(poses)
        timestamps = (
            np.arange(n, dtype=float) * 0.04
            if timestamps is None
            else np.asarray(timestamps, dtype=float)
        )
        present = np.ones(n, bool) if present is None else np.asarray(present, bool)
        r = Recording(
            Path("synthetic.npz"),
            np.asarray(poses, np.float32),
            timestamps,
            present,
            np.arange(n),
            np.where(present, 7, -1),
            np.zeros(n, np.int64),
            1280,
            720,
            float(timestamps[-1]),
        )
        return Dataset(
            r,
            np.ones(n, np.int64) if generations is None else np.asarray(generations),
            np.full(n, 7),
            {},
        )

    def check(condition, message):
        if not condition:
            raise AssertionError(message)

    sequence = [pose()] * 3 + [pose(True)] * 3 + [pose()] * 3
    data = dataset(sequence)
    rows = replay(data, cfg)
    check(rows[-1].reps == 1 and rows[-1].counted, "closed-open-closed must count once")
    check(
        rows[0].state == "UNKNOWN" and rows[2].state == "CLOSED",
        "confirmation duration",
    )
    check(
        replay(dataset([pose(True)] * 3 + [pose()] * 3), cfg)[-1].reps == 0,
        "open start cannot arm",
    )
    check(
        replay(dataset([pose()] * 3 + [pose(True)] + [pose()] * 3), cfg)[-1].reps == 0,
        "brief open rejected",
    )
    dead = pose()
    dead[9, 1] = dead[10, 1] = 0.4
    dead_rows = replay(dataset(sequence[:6] + [dead]), cfg)
    check(
        dead_rows[-1].state == "OPEN" and dead_rows[-1].pending == "NONE",
        "dead band preserves stable phase",
    )
    bad = pose()
    bad[10, 2] = 0.49
    check(
        replay(dataset(sequence[:6] + [bad]), cfg)[-1].state == "UNKNOWN",
        "low-confidence joint resets",
    )
    bad[10, :2] = np.nan
    check(
        replay(dataset([bad]), replace(cfg, confidence=0))[-1].candidate == "INVALID",
        "NaNs cannot be restored",
    )
    bad_pose = pose()
    bad_pose[6, 0] = bad_pose[5, 0]
    check(
        replay(dataset([bad_pose]), cfg)[-1].candidate == "INVALID",
        "zero shoulder width",
    )
    # Use binary-exact coordinates for boundary tests, avoiding float32 rounding.
    boundary = pose()
    boundary[5, :2], boundary[6, :2] = (0.375, 0.25), (0.625, 0.25)
    boundary[15, :2], boundary[16, :2] = (0.25, 0.75), (0.75, 0.75)
    boundary[9, 1] = boundary[10, 1] = 0.125
    exact = replace(
        cfg, wrist_margin=0.25, open_ankle_ratio=2.0, minimum_shoulder_width=0.25
    )
    check(
        replay(dataset([boundary]), exact)[0].candidate == "DEAD BAND",
        "wrist equality must fail",
    )
    boundary[9, 1] = boundary[10, 1] = 0.0
    check(
        replay(dataset([boundary]), exact)[0].candidate == "OPEN",
        "ankle and shoulder-width equality must pass",
    )
    instant = replay(dataset(sequence), replace(cfg, confirm_seconds=0))
    check(
        instant[0].state == "CLOSED" and instant[-1].reps == 1,
        "zero confirmation supported",
    )
    cancelled = replay(dataset([pose(), pose(), dead, pose(), pose()]), cfg)
    check(cancelled[-1].state == "UNKNOWN", "dead band restarts confirmation duration")
    bad_span = pose()
    bad_span[15:17, 1] = 0.3
    check(replay(dataset([bad_span]), cfg)[-1].candidate == "INVALID", "negative span")
    changed = replay(dataset(sequence, generations=[1] * 6 + [3] * 3), cfg)
    check(changed[-1].reps == 0 and changed[6].reset, "generation resets phase")
    gap = replay(
        dataset(
            sequence, timestamps=[0, 0.04, 0.08, 0.12, 0.16, 0.20, 1.0, 1.04, 1.08]
        ),
        cfg,
    )
    check(gap[-1].reps == 0 and "gap" in gap[6].reset, "large gap resets phase")
    missing = replay(
        dataset(
            sequence[:6] + [np.full((17, 3), np.nan)], present=[True] * 6 + [False]
        ),
        cfg,
    )
    check(missing[-1].state == "UNKNOWN", "absent person resets")
    seeded = replay(dataset(sequence + sequence, generations=[1] * 9 + [2] * 9), cfg)
    check(seeded[-1].reps == 2, "offline accepted total preserved across generations")
    # Verify the explanatory gates against the actual classifier on randomized
    # poses/configs; a display-only validity check must not change semantics.
    rng = np.random.default_rng(17)
    for _ in range(250):
        p = rng.uniform(-0.1, 1.1, (17, 3)).astype(np.float32)
        p[:, 2] = 0.99
        if rng.random() < 0.1:
            p[rng.choice(REQUIRED), 0] = np.nan
        config = replace(
            cfg,
            closed_ankle_ratio=float(rng.uniform(0.5, 3)),
            open_ankle_ratio=float(rng.uniform(0.5, 3)),
            wrist_margin=float(rng.uniform(0, 0.5)),
        )
        m = measure(p, True, config)
        closed, opened = gates(m, config)
        expected = (
            "INVALID"
            if m.invalid
            else "CLOSED"
            if all(closed)
            else "OPEN"
            if all(opened)
            else "DEAD BAND"
        )
        actual = replay(dataset([p]), config)[0].candidate
        check(expected == actual, "display/classifier gate mismatch")
    original = data.recording.poses.copy()
    clock = PlaybackClock(data.recording.duration)
    for i in (0, 5, 8):
        frame = render(data, rows, cfg, i, clock)
        check(frame.shape == (HEIGHT, WIDTH, 3), "render dimensions")
    check(
        np.array_equal(original, data.recording.poses), "render must not mutate poses"
    )
    args = argparse.Namespace(
        speed=1.0,
        loop=False,
        start=0.0,
        paused=True,
        indices=False,
        max_width=1500,
        max_height=960,
    )
    player = DebugPlayer(data, cfg, args)
    player.seek_frame(8)
    player.key(ord("["))
    check(
        player.index == 7 and player.rows[-1].reps == 1,
        "stepping independent of replay",
    )
    player.selected = 4
    player.adjust(1)
    check(
        player.cfg.confirm_seconds == 0.07 and player.rows[-1].reps == 1,
        "in-memory tuning",
    )
    player.key(ord("b"))
    check(player.cfg == cfg, "source defaults restored")
    check(
        "torch" not in sys.modules and "ultralytics" not in sys.modules,
        "no inference dependencies",
    )
    print(
        "Self-test passed: sequence, timing, confidence, NaNs, scale, loss/generation, gates, rendering and controls."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "recording", nargs="?", type=Path, help="Default: newest NPZ beside this script"
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--indices", action="store_true")
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=960)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print diagnostic JSON and exit without a window",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Render --start to a NEW PNG and exit without a window",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic headless checks; no recording needed",
    )
    defaults = JumpingJackConfig()
    for control in CONTROLS:
        parser.add_argument(
            "--" + control.field.replace("_", "-"),
            type=float,
            default=getattr(defaults, control.field),
            help=f"What-if {control.label.lower()}; current source default: {getattr(defaults, control.field)}",
        )
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if not math.isfinite(args.speed) or not 0.125 <= args.speed <= 8:
        parser.error("--speed must be between 0.125 and 8")
    if args.max_width < 1000 or args.max_height < 640:
        parser.error("Display bounds must be at least 1000 x 640")
    for control in CONTROLS:
        value = getattr(args, control.field)
        if not math.isfinite(value) or not control.minimum <= value <= control.maximum:
            parser.error(
                f"--{control.field.replace('_', '-')} must be between {control.minimum:g} and {control.maximum:g}"
            )
    cfg = JumpingJackConfig(**{c.field: getattr(args, c.field) for c in CONTROLS})
    try:
        data = Dataset.load(resolve_recording(args.recording))
        player = DebugPlayer(data, cfg, args)
        print(json.dumps(summarize(data, player.rows, cfg), indent=2, allow_nan=False))
        if data.inferred_generations:
            print(
                "WARNING: generations not saved; inferred from presence and track changes."
            )
        if args.snapshot:
            if args.snapshot.suffix.lower() != ".png":
                parser.error("--snapshot must use a .png extension")
            player.clock.playing = False
            success, encoded = cv2.imencode(".png", player.frame())
            if not success:
                raise OSError("Could not encode snapshot")
            # Exclusive creation avoids accidentally overwriting any user file.
            with args.snapshot.open("xb") as stream:
                stream.write(encoded.tobytes())
            print(f"Snapshot: {args.snapshot.resolve()}")
        elif not args.summary:
            player.play()
    except (OSError, ValueError, TypeError, KeyError, EOFError, cv2.error) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
