"""Read-only live skeleton/threshold UI, imported only by the main runtime.

Current-pose checks are separate from the worker's accepted phase and rep count.
This view never updates a counter, selects a person or changes a recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from workout.analysis.jumping_jack import JumpingJackConfig, JumpingJackCounter
from workout.packets import TrackedPerson, clean_pose
from workout.session import SessionController, SessionState

WIDTH, HEIGHT = 1280, 860
BG, PANEL = (22, 19, 17), (34, 30, 27)
TEXT, MUTED, BORDER = (238, 236, 231), (174, 166, 156), (69, 60, 52)
GREEN, BLUE, RED, YELLOW = (
    (139, 220, 123),
    (246, 183, 91),
    (108, 106, 249),
    (102, 218, 247),
)
LEFT, RIGHT = (200, 220, 50), (80, 165, 250)
REQUIRED = (5, 6, 9, 10, 15, 16)
JOINT_NAMES = ("L shoulder", "R shoulder", "L wrist", "R wrist", "L ankle", "R ankle")
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


@dataclass(frozen=True)
class PoseChecks:
    pose: np.ndarray
    valid: np.ndarray
    width: float
    span: float
    left: float
    right: float
    ratio: float
    closed: tuple[bool, bool, bool]
    opened: tuple[bool, bool, bool]
    posture: str
    invalid: str


def inspect_pose(pose: np.ndarray | None, cfg: JumpingJackConfig) -> PoseChecks:
    """Explain the actual classifier, using its float32 operation order."""
    points = np.full((17, 3), np.nan, np.float32) if pose is None else pose
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= cfg.confidence)
    width = abs(float(points[5, 0] - points[6, 0]))
    span = float((points[15, 1] + points[16, 1] - points[5, 1] - points[6, 1]) / 2)
    left = float(points[9, 1] - points[5, 1])
    right = float(points[10, 1] - points[6, 1])
    ratio = abs(float(points[15, 0] - points[16, 0])) / width if width > 0 else math.nan
    margin = cfg.wrist_margin * span
    if pose is None:
        invalid = "No current pose"
    elif not valid[list(REQUIRED)].all():
        bad = [name for k, name in zip(REQUIRED, JOINT_NAMES) if not valid[k]]
        invalid = "Missing / low confidence: " + ", ".join(bad)
    elif width < cfg.minimum_shoulder_width:
        invalid = "Shoulder width is below the minimum"
    elif span <= 0:
        invalid = "Shoulder-to-ankle vertical span must be > 0"
    else:
        invalid = ""
    closed = (left > margin, right > margin, ratio <= cfg.closed_ankle_ratio)
    opened = (left < -margin, right < -margin, ratio >= cfg.open_ankle_ratio)
    # classify() may reset its own phase. This local instance never calls
    # update(), counts reps or touches the independent background worker.
    classified = JumpingJackCounter(cfg).classify(points)
    posture = "INVALID" if invalid else classified.value if classified else "TRANSITION"
    return PoseChecks(
        points, valid, width, span, left, right, ratio, closed, opened, posture, invalid
    )


def display_person(session: SessionController, people: list[TrackedPerson]):
    """Use the selected pose; finished previews use current tracking without recording."""
    packet = session.last_packet
    if session.state == SessionState.FINISHED:
        person = next(
            (p for p in people if p.track_id == session.selected_track_id), None
        )
        if person is None:
            return None, f"Selected ID {session.selected_track_id} missing", True
        return clean_pose(person.pose), f"Live preview ID {person.track_id}", True
    if session.state == SessionState.ACTIVE_TRACKED:
        if packet is not None and packet.present:
            return packet.pose, f"Selected ID {packet.selected_track_id}", False
        return None, "Selected person missing", False
    gesture = (
        session.start_gesture
        if session.state == SessionState.WAITING
        else session.reacquire_gesture
    )
    person = next((p for p in people if p.track_id == gesture.candidate_id), None)
    if (
        person is None
        and session.state == SessionState.WAITING
        and gesture.candidate_id is None
        and people
    ):
        person = min(people, key=lambda p: p.track_id)
    if person is None:
        caption = (
            "No gesture candidate"
            if session.state == SessionState.WAITING
            else f"Selected ID {session.selected_track_id} LOST"
        )
        return None, caption, True
    return clean_pose(person.pose), f"Preview ID {person.track_id} - NOT selected", True


def target_posture(session: SessionController) -> str | None:
    if session.state != SessionState.ACTIVE_TRACKED:
        return None
    return "OPEN" if session.exercise_state == "CLOSED" else "CLOSED"


def text(image, message, x, y, color=TEXT, scale=0.5, width=None):
    message = str(message)
    limit = image.shape[1] - x - 10 if width is None else width
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


def wrap(image, message, x, y, width, color=TEXT, scale=0.5, max_lines=3):
    words, line, row = str(message).split(), "", 0
    for i, word in enumerate(words):
        candidate = f"{line} {word}".strip()
        if (
            line
            and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            > width
        ):
            text(image, line, x, y + row * 22, color, scale, width)
            row += 1
            if row == max_lines - 1:
                text(image, " ".join(words[i:]), x, y + row * 22, color, scale, width)
                return
            line = word
        else:
            line = candidate
    text(image, line, x, y + row * 22, color, scale, width)


def panel(image, rect, title):
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL, -1)
    text(image, title, x + 16, y + 26, MUTED, 0.49, w - 32)


def number(value):
    return f"{value:.3f}" if math.isfinite(value) else "N/A"


def color_for(posture):
    return {"OPEN": BLUE, "CLOSED": GREEN, "INVALID": RED, "TRANSITION": YELLOW}.get(
        posture, MUTED
    )


def project(pose, valid, camera_width, camera_height, viewport):
    """Preserve camera aspect ratio. Clip only drawing, not analysis validity."""
    width, height = viewport
    scale = min(width / camera_width, height / camera_height)
    source = np.array([camera_width, camera_height], dtype=np.float64) * scale
    origin = (np.array([width, height]) - source) / 2
    points = np.zeros((17, 2), dtype=np.int32)
    finite = np.isfinite(pose[:, :2]).all(axis=1) & valid
    points[finite] = np.clip(
        np.rint(pose[finite, :2] * source + origin), -10000, 10000
    ).astype(np.int32)
    return points, finite, source, origin


def dashed(image, a, b, color):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    for start in np.arange(0, length, 12):
        p = np.rint(a + (b - a) * start / length).astype(int)
        q = np.rint(a + (b - a) * min(start + 6, length) / length).astype(int)
        cv2.line(image, tuple(p), tuple(q), color, 1, cv2.LINE_AA)


class LiveSkeletonView:
    def __init__(self):
        self.config = JumpingJackConfig()

    def draw_pose(self, image, checks, camera_width, camera_height, caption):
        panel(image, (16, 90, 644, 444), "SKELETON  |  anatomical L: cyan   R: orange")
        view = image[133:496, 32:644]
        p = checks.pose
        points, valid, source, origin = project(
            p, checks.valid, camera_width, camera_height, (view.shape[1], view.shape[0])
        )

        def xy(x, y):
            return tuple(
                np.clip(
                    np.rint(np.array([x, y]) * source + origin), -10000, 10000
                ).astype(int)
            )

        cv2.rectangle(view, xy(0, 0), xy(1, 1), BORDER, 1)
        margin = self.config.wrist_margin * checks.span
        if math.isfinite(margin) and checks.span > 0:
            for shoulder, side in ((5, "L"), (6, "R")):
                if not valid[shoulder]:
                    continue
                x, y = map(float, p[shoulder, :2])
                for sign, color, title in (
                    (-1, BLUE, "OPEN <"),
                    (1, GREEN, "CLOSED >"),
                ):
                    level = y + sign * margin
                    dashed(view, xy(x - 0.12, level), xy(x + 0.12, level), color)
                    other = 6 if shoulder == 5 else 5
                    on_left = not valid[other] or x <= float(p[other, 0])
                    px, py = xy(x - 0.12 if on_left else x + 0.12, level)
                    label_x = (
                        max(0, px - 69) if on_left else min(view.shape[1] - 78, px + 4)
                    )
                    text(view, f"{side} {title}", label_x, py - 5, color, 0.33)
        if valid[5] and valid[6]:
            y = float((p[5, 1] + p[6, 1]) / 2)
            cv2.line(view, xy(float(p[5, 0]), y), xy(float(p[6, 0]), y), YELLOW, 2)
        if math.isfinite(checks.ratio) and checks.width > 0:
            middle = float((p[15, 0] + p[16, 0]) / 2)
            for ratio, color, y in (
                (self.config.closed_ankle_ratio, GREEN, 0.94),
                (self.config.open_ankle_ratio, BLUE, 0.99),
            ):
                a, b = (
                    xy(middle - ratio * checks.width / 2, y),
                    xy(middle + ratio * checks.width / 2, y),
                )
                cv2.line(view, a, b, color, 2)
                for px, py in (a, b):
                    cv2.line(view, (px, py - 4), (px, py + 4), color, 1)
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                color = (
                    LEFT
                    if a % 2 and b % 2
                    else RIGHT
                    if not a % 2 and not b % 2
                    else MUTED
                )
                cv2.line(
                    view, tuple(points[a]), tuple(points[b]), color, 3, cv2.LINE_AA
                )
        for k in np.flatnonzero(valid):
            color = LEFT if k % 2 else RIGHT
            cv2.circle(
                view,
                tuple(points[k]),
                5 if k in REQUIRED else 3,
                color,
                -1,
                cv2.LINE_AA,
            )
            if k in REQUIRED:
                text(
                    view,
                    str(k),
                    int(points[k, 0]) + 7,
                    int(points[k, 1]) - 5,
                    TEXT,
                    0.35,
                )
        if not valid.any():
            text(view, "NO CURRENT SKELETON", 126, 165, MUTED, 0.7)
            text(view, "Keep your full body in the camera view", 92, 194, MUTED, 0.5)
        text(image, caption, 32, 518, TEXT, 0.46, 610)

    def draw_checks(self, image, checks, preview):
        cfg = self.config
        panel(
            image,
            (676, 90, 588, 346),
            "CURRENT-POSE CHECKS  |  all 3 gates in one column must pass",
        )
        text(
            image,
            "Preview only" if preview else "Current geometry (not a confirmed rep)",
            692,
            143,
            MUTED,
            0.43,
        )
        text(image, "CLOSED", 940, 171, GREEN, 0.52)
        text(image, "OPEN", 1103, 171, BLUE, 0.52)
        values = (
            checks.left / checks.span if checks.span > 0 else math.nan,
            checks.right / checks.span if checks.span > 0 else math.nan,
            checks.ratio,
        )
        for row, (name, value) in enumerate(
            zip(("L wrist dY / span", "R wrist dY / span", "Feet / shoulders"), values)
        ):
            y = 205 + row * 48
            text(image, name, 692, y, TEXT, 0.48, 235)
            text(image, f"Measured: {number(value)}", 692, y + 19, MUTED, 0.43, 235)
            rules = (
                (f"> +{cfg.wrist_margin:g}", f"< -{cfg.wrist_margin:g}")
                if row < 2
                else (f"<= {cfg.closed_ankle_ratio:g}", f">= {cfg.open_ankle_ratio:g}")
            )
            for x, passed, rule in (
                (940, checks.closed[row], rules[0]),
                (1103, checks.opened[row], rules[1]),
            ):
                status = "N/A" if checks.invalid else "PASS" if passed else "FAIL"
                color = MUTED if checks.invalid else GREEN if passed else RED
                text(image, f"{status}  {rule}", x, y + 3, color, 0.46, 144)
        text(
            image, f"Pose: {checks.posture}", 692, 361, color_for(checks.posture), 0.67
        )
        reason = checks.invalid or (
            "Rule: transition keeps stable phase, but cancels pending confirmation."
            if checks.posture == "TRANSITION"
            else "Spatial gates met; timing and the confirmed cycle still apply."
        )
        wrap(image, reason, 692, 394, 552, RED if checks.invalid else MUTED, 0.46, 2)

    def draw_confidence(self, image, checks):
        cfg = self.config
        panel(
            image,
            (16, 548, 644, 224),
            "REQUIRED JOINTS  |  all six finite and confident",
        )
        for row, (k, name) in enumerate(zip(REQUIRED, JOINT_NAMES)):
            y = 602 + row * 27
            text(image, f"{name} [{k}]", 32, y, LEFT if k % 2 else RIGHT, 0.48, 176)
            confidence = float(checks.pose[k, 2])
            text(image, number(confidence), 213, y, TEXT, 0.48)
            cv2.line(image, (289, y - 6), (453, y - 6), BORDER, 5)
            if math.isfinite(confidence):
                cv2.line(
                    image,
                    (289, y - 6),
                    (289 + round(164 * np.clip(confidence, 0, 1)), y - 6),
                    GREEN if checks.valid[k] else RED,
                    5,
                )
            bound = 289 + round(164 * cfg.confidence)
            cv2.line(image, (bound, y - 14), (bound, y + 2), TEXT, 1)
            status = (
                "PASS"
                if checks.valid[k]
                else "NaN xy"
                if not np.isfinite(checks.pose[k, :2]).all()
                else "LOW CONF"
            )
            text(image, status, 484, y, GREEN if checks.valid[k] else RED, 0.46, 148)
        text(
            image,
            f"Confidence >= {cfg.confidence:g}. A bad required joint resets the cycle.",
            32,
            762,
            MUTED,
            0.42,
            610,
        )

    def draw_progress(self, image, session, checks, analysis_note, saved):
        cfg = self.config
        panel(
            image,
            (676, 450, 588, 322),
            "WORKOUT PROGRESS  |  authoritative accepted worker result",
        )
        if session.state in (SessionState.WAITING, SessionState.ACTIVE_LOST):
            gesture = (
                session.start_gesture
                if session.state == SessionState.WAITING
                else session.reacquire_gesture
            )
            title = (
                "START SESSION"
                if session.state == SessionState.WAITING
                else "TRACK LOST - count frozen"
            )
            text(image, title, 692, 510, YELLOW, 0.66, 552)
            wrap(
                image,
                f"Raise anatomical RIGHT hand above your head for {gesture.hold_seconds:g}s.",
                692,
                547,
                548,
                TEXT,
                0.52,
                2,
            )
            text(
                image,
                "Wrist above nose AND elbow above right shoulder.",
                692,
                599,
                MUTED,
                0.46,
                548,
            )
            fraction = min(1.0, max(0.0, gesture.progress))
            cv2.rectangle(image, (692, 625), (1244, 641), BORDER, -1)
            cv2.rectangle(
                image, (692, 625), (692 + round(552 * fraction), 641), YELLOW, -1
            )
            text(
                image,
                f"Gesture ID {gesture.candidate_id if gesture.candidate_id is not None else '-'}: "
                f"{fraction * 100:.0f}% / {gesture.hold_seconds:g}s",
                692,
                670,
                TEXT,
                0.5,
                552,
            )
            text(
                image,
                "Joint preview does not select or count a different person.",
                692,
                703,
                MUTED,
                0.43,
                552,
            )
        elif session.state == SessionState.FINISHED:
            text(
                image,
                f"FINISHED - {session.rep_count} jumping jacks",
                692,
                515,
                GREEN,
                0.71,
                552,
            )
            wrap(
                image,
                f"Saved: {saved.name}" if saved else "Saving session...",
                692,
                554,
                552,
                TEXT,
                0.52,
                3,
            )
            text(
                image,
                "Live tracking preview. Recording and rep count have stopped.",
                692,
                640,
                MUTED,
                0.45,
                552,
            )
        else:
            target = target_posture(session)
            text(
                image,
                f"Confirmed phase: {session.exercise_state}",
                692,
                507,
                color_for(session.exercise_state),
                0.62,
                552,
            )
            text(
                image,
                "CLOSED  ->  OPEN  ->  CLOSED  =  +1 rep",
                692,
                540,
                TEXT,
                0.55,
                552,
            )
            message = (
                "Establish CLOSED first"
                if session.exercise_state == "UNKNOWN"
                else f"Next target: {target}"
            )
            text(image, message, 692, 578, color_for(target), 0.65, 552)
            if checks.invalid:
                advice = "Restore all six required joints and body scale."
            elif checks.posture == target:
                advice = (
                    "Target pose met. Await worker confirmation of the full sequence."
                )
            else:
                passed = checks.opened if target == "OPEN" else checks.closed
                actions = (
                    ("Raise LEFT wrist", "Raise RIGHT wrist", "Spread feet wider")
                    if target == "OPEN"
                    else ("Lower LEFT wrist", "Lower RIGHT wrist", "Bring feet closer")
                )
                advice = (
                    "; ".join(action for action, ok in zip(actions, passed) if not ok)
                    + "."
                )
            wrap(image, advice, 692, 610, 552, YELLOW, 0.53, 2)
            text(
                image,
                f"Same posture must persist >= {cfg.confirm_seconds:.3f}s in the worker.",
                692,
                674,
                MUTED,
                0.45,
                552,
            )
            text(
                image,
                f"Worker packet gaps > {cfg.max_sample_gap:g}s reset the phase.",
                692,
                698,
                MUTED,
                0.45,
                552,
            )
            text(
                image,
                f"Accepted analysis frame: {session.last_status_frame if session.last_status_frame >= 0 else 'waiting'}",
                692,
                722,
                MUTED,
                0.45,
                552,
            )
        text(
            image,
            analysis_note
            or "Pose checks are read-only; thresholds and rep counting are unchanged.",
            692,
            757,
            RED if analysis_note else MUTED,
            0.4,
            552,
        )

    def render(
        self,
        session,
        people,
        *,
        camera_width,
        camera_height,
        now,
        frame_index,
        dropped=0,
        analysis_note="",
        saved=None,
    ):
        image = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        pose, caption, preview = display_person(session, people)
        checks = inspect_pose(pose, self.config)
        elapsed = session.elapsed(now)
        text(image, "JUMPING JACK / LIVE SKELETON", 20, 31, TEXT, 0.7)
        text(image, f"Reps: {session.rep_count}", 713, 37, GREEN, 1.0)
        text(
            image,
            f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d} / {int(session.duration_seconds) // 60:02d}:{int(session.duration_seconds) % 60:02d}",
            1015,
            37,
            TEXT,
            0.8,
        )
        phase = (
            "FINISHED / live preview"
            if session.state == SessionState.FINISHED
            else session.state.value
        )
        text(
            image,
            f"{phase}   |   {caption}   |   Frame {frame_index}",
            20,
            66,
            MUTED,
            0.47,
            870,
        )
        text(image, "V: video overlay   Q/Esc: quit", 957, 68, TEXT, 0.45, 305)
        self.draw_pose(
            image,
            checks,
            camera_width,
            camera_height,
            "Dashed: wrist limits. Bottom bars: closed / open foot spans.",
        )
        self.draw_checks(image, checks, preview)
        self.draw_confidence(image, checks)
        self.draw_progress(image, session, checks, analysis_note, saved)
        width_ok = (
            math.isfinite(checks.width)
            and checks.width >= self.config.minimum_shoulder_width
        )
        span_ok = math.isfinite(checks.span) and checks.span > 0
        text(
            image,
            f"Shoulder width: {number(checks.width)} >= {self.config.minimum_shoulder_width:g}",
            24,
            803,
            GREEN if width_ok else RED,
            0.49,
            456,
        )
        text(
            image,
            f"Body span: {number(checks.span)} > 0",
            470,
            803,
            GREEN if span_ok else RED,
            0.49,
            400,
        )
        text(image, f"Analysis drops: {dropped}", 1010, 803, MUTED, 0.46, 250)
        return image

    @staticmethod
    def draw_performance(image, values):
        text(
            image,
            f"Loop {values['loop_fps']:.1f} FPS   |   Inference {values['inference_ms']:.1f} ms"
            f"   |   View {values['plot_ms']:.1f} ms   |   Camera {values['camera_width']}x{values['camera_height']}"
            f"   |   {values['device_name']}",
            24,
            840,
            MUTED,
            0.45,
            1232,
        )
