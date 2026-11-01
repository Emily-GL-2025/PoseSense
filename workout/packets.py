"""Small CPU-only data contracts; never pass video frames to analysis."""

from dataclasses import dataclass

import numpy as np

KEYPOINT_CONFIDENCE = 0.5
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 8, 9, 10
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def clean_pose(
    pose: np.ndarray | None, confidence: float = KEYPOINT_CONFIDENCE
) -> np.ndarray:
    """Own a float32 pose, masking unreliable x/y while retaining confidence."""
    if pose is None:
        return np.full((17, 3), np.nan, dtype=np.float32)
    result = np.array(pose, dtype=np.float32, copy=True)
    if result.shape != (17, 3):
        raise ValueError(f"Expected a (17, 3) COCO pose, got {result.shape}")
    valid = np.isfinite(result).all(axis=1) & (result[:, 2] >= confidence)
    result[~valid, :2] = np.nan
    return result


def valid_keypoints(
    pose: np.ndarray, indices: tuple[int, ...], confidence: float
) -> bool:
    points = pose[list(indices)]
    return bool(np.isfinite(points).all() and (points[:, 2] >= confidence).all())


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    pose: np.ndarray  # normalized COCO [17,3], current frame only


@dataclass(frozen=True)
class PosePacket:
    frame_index: int
    timestamp: float  # seconds since session start
    selected_track_id: int
    pose: np.ndarray
    present: bool
    generation: int  # survives overwritten loss packets
    accepted_reps: int = 0  # conservative baseline after a discontinuity


@dataclass(frozen=True)
class AnalysisStatus:
    rep_count: int
    exercise_state: str
    last_processed_frame: int
    generation: int
    processed_packets: int = 0
    error: str = ""
