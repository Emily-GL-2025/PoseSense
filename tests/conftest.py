import numpy as np
import pytest

from workout.packets import TrackedPerson


@pytest.fixture
def pose_factory():
    def make(posture="closed", track_id=7):
        pose = np.full((17, 3), 0.5, dtype=np.float32)
        pose[:, 2] = 0.99
        pose[0, :2] = (0.5, 0.2)
        pose[5, :2], pose[6, :2] = (0.4, 0.4), (0.6, 0.4)
        pose[9, :2], pose[10, :2] = (0.4, 0.65), (0.6, 0.65)
        pose[15, :2], pose[16, :2] = (0.45, 0.9), (0.55, 0.9)
        if posture == "raised":
            pose[10, 1], pose[8, 1] = 0.1, 0.3
        elif posture == "open":
            pose[9, 1] = pose[10, 1] = 0.1
            pose[15, 0], pose[16, 0] = 0.25, 0.75
        elif posture == "jitter":
            pose[15, 0], pose[16, 0] = 0.38, 0.62
            pose[9, 1] = pose[10, 1] = 0.4
        return TrackedPerson(track_id, pose)

    return make
