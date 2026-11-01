"""Anatomical RIGHT-hand selection, measured with monotonic seconds."""

from dataclasses import dataclass

from workout.packets import (
    KEYPOINT_CONFIDENCE,
    NOSE,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    TrackedPerson,
    valid_keypoints,
)

START_HOLD_SECONDS = 2.0
REACQUIRE_HOLD_SECONDS = 1.0
MISSING_GRACE_SECONDS = 0.15


def right_hand_raised(
    person: TrackedPerson, confidence: float = KEYPOINT_CONFIDENCE
) -> bool | None:
    """None means uncertain; False means the hand was confidently lowered."""
    pose = person.pose
    if not valid_keypoints(
        pose, (NOSE, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), confidence
    ):
        return None
    return bool(
        pose[RIGHT_WRIST, 1] < pose[NOSE, 1]
        and pose[RIGHT_ELBOW, 1] < pose[RIGHT_SHOULDER, 1]
    )


@dataclass
class Hold:
    started: float
    last_valid: float


class GestureDetector:
    def __init__(
        self,
        hold_seconds: float = START_HOLD_SECONDS,
        grace_seconds: float = MISSING_GRACE_SECONDS,
        confidence: float = KEYPOINT_CONFIDENCE,
    ):
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.confidence = confidence
        self.holds: dict[int, Hold] = {}
        self.candidate_id: int | None = None
        self.progress = 0.0

    def reset(self) -> None:
        self.holds.clear()
        self.candidate_id = None
        self.progress = 0.0

    def update(self, people: list[TrackedPerson], now: float) -> int | None:
        # Expire before processing new observations: a long camera stall cannot
        # masquerade as a continuously observed raised hand.
        self.holds = {
            tid: hold
            for tid, hold in self.holds.items()
            if now - hold.last_valid <= self.grace_seconds
        }
        raised_ids = set()
        for person in people:
            raised = right_hand_raised(person, self.confidence)
            if raised is False:
                self.holds.pop(person.track_id, None)
            elif raised is True:
                hold = self.holds.setdefault(person.track_id, Hold(now, now))
                hold.last_valid = now
                raised_ids.add(person.track_id)
        self.candidate_id = None
        self.progress = 0.0
        if not self.holds:
            return None
        # No bounding-box-size preference; ties have a deterministic outcome.
        ordered = sorted(self.holds, key=lambda tid: (self.holds[tid].started, tid))
        self.candidate_id = ordered[0]
        self.progress = min(
            1.0, (now - self.holds[ordered[0]].started) / self.hold_seconds
        )
        for tid in ordered:
            if tid in raised_ids and now - self.holds[tid].started >= self.hold_seconds:
                return tid
        return None
