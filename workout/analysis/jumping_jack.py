"""Experimental body-relative jumping-jack heuristics; tune the config below."""

from dataclasses import dataclass
from enum import Enum

from workout.packets import (
    KEYPOINT_CONFIDENCE,
    LEFT_ANKLE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PosePacket,
    valid_keypoints,
)


class ExerciseState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CLOSED = "CLOSED"
    OPEN = "OPEN"


@dataclass(frozen=True)
class JumpingJackConfig:
    """Experimental thresholds for a frontal, full-body camera view."""

    confidence: float = KEYPOINT_CONFIDENCE
    closed_ankle_ratio: float = 1.0
    open_ankle_ratio: float = 1.4
    wrist_margin: float = 0.10  # fraction of vertical shoulder-to-ankle span
    confirm_seconds: float = 0.06
    max_sample_gap: float = 0.5
    minimum_shoulder_width: float = 0.01  # reject a degenerate scale


class JumpingJackCounter:
    def __init__(self, config: JumpingJackConfig | None = None):
        self.config = config or JumpingJackConfig()
        self.state = ExerciseState.UNKNOWN
        self.rep_count = 0
        self.generation: int | None = None
        self.track_id: int | None = None
        self.last_timestamp: float | None = None
        self.last_frame = -1
        self.pending: ExerciseState | None = None
        self.pending_since = 0.0

    def reset_phase(self) -> None:
        self.state = ExerciseState.UNKNOWN
        self.pending = None

    def classify(self, pose) -> ExerciseState | None:
        cfg = self.config
        needed = (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        )
        if not valid_keypoints(pose, needed, cfg.confidence):
            self.reset_phase()
            return None
        width = abs(float(pose[LEFT_SHOULDER, 0] - pose[RIGHT_SHOULDER, 0]))
        height = float(
            (
                pose[LEFT_ANKLE, 1]
                + pose[RIGHT_ANKLE, 1]
                - pose[LEFT_SHOULDER, 1]
                - pose[RIGHT_SHOULDER, 1]
            )
            / 2
        )
        if width < cfg.minimum_shoulder_width or height <= 0:
            self.reset_phase()
            return None
        ratio = abs(float(pose[LEFT_ANKLE, 0] - pose[RIGHT_ANKLE, 0])) / width
        left = float(pose[LEFT_WRIST, 1] - pose[LEFT_SHOULDER, 1])
        right = float(pose[RIGHT_WRIST, 1] - pose[RIGHT_SHOULDER, 1])
        margin = cfg.wrist_margin * height
        if left > margin and right > margin and ratio <= cfg.closed_ankle_ratio:
            return ExerciseState.CLOSED
        if left < -margin and right < -margin and ratio >= cfg.open_ankle_ratio:
            return ExerciseState.OPEN
        return None  # hysteresis dead band: keep stable phase, cancel confirmation

    def update(self, packet: PosePacket) -> int:
        if packet.frame_index <= self.last_frame:
            return self.rep_count
        changed = (
            packet.generation != self.generation
            or packet.selected_track_id != self.track_id
        )
        if changed:
            self.reset_phase()
            self.rep_count = packet.accepted_reps
        elif (
            self.last_timestamp is not None
            and not 0
            <= packet.timestamp - self.last_timestamp
            <= self.config.max_sample_gap
        ):
            self.reset_phase()
        self.generation = packet.generation
        self.track_id = packet.selected_track_id
        self.last_timestamp = packet.timestamp
        self.last_frame = packet.frame_index
        if not packet.present:
            self.reset_phase()
            return self.rep_count
        posture = self.classify(packet.pose)
        if posture is None:
            self.pending = None
            return self.rep_count
        if posture != self.pending:
            self.pending, self.pending_since = posture, packet.timestamp
        if packet.timestamp - self.pending_since < self.config.confirm_seconds:
            return self.rep_count
        if self.state == ExerciseState.UNKNOWN:
            if posture == ExerciseState.CLOSED:
                self.state = ExerciseState.CLOSED
        elif posture != self.state:
            if self.state == ExerciseState.OPEN and posture == ExerciseState.CLOSED:
                self.rep_count += 1
            self.state = posture
        return self.rep_count
