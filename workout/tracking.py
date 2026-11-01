"""Installed Ultralytics integration; only small pose/box tensors leave the GPU."""

import importlib.util
import time
from pathlib import Path

import numpy as np

from workout.packets import TrackedPerson


def select_tracker(requested: str = "auto") -> str:
    import ultralytics
    from ultralytics.utils import YAML

    # Prevent Ultralytics' import-time automatic installer from changing the env.
    if importlib.util.find_spec("lap") is None:
        raise RuntimeError(
            "Tracking requires lap>=0.5.12. Install only that optional dependency: "
            "uv pip install --python .venv/Scripts/python.exe --no-deps 'lap>=0.5.12'"
        )
    from ultralytics.trackers.track import TRACKER_MAP

    directory = Path(ultralytics.__file__).parent / "cfg" / "trackers"
    candidates = ("fasttrack", "bytetrack") if requested == "auto" else (requested,)
    for name in candidates:
        path = directory / f"{name}.yaml"
        if not path.is_file():
            continue
        config = YAML.load(path)
        if (
            config.get("tracker_type") == name
            and name in TRACKER_MAP
            and not config.get("with_reid", False)
        ):
            return str(path)
    raise RuntimeError(f"No supported non-ReID tracker config found for {requested!r}")


def tracked_people(result, selected_track_id: int | None = None) -> list[TrackedPerson]:
    """Extract aligned current-frame IDs/poses; unconfirmed boxes aren't people.

    While the selected ID is visible we only transfer that person's keypoints.
    While waiting/lost, candidates are also needed for the deliberate gesture.
    """
    if result.boxes is None or result.keypoints is None or result.boxes.id is None:
        return []
    ids = result.boxes.id.detach().cpu().numpy().astype(np.int64, copy=False)
    indices = (
        np.flatnonzero(ids == selected_track_id)
        if selected_track_id is not None
        else np.arange(len(ids))
    )
    if not len(indices):
        indices = np.arange(len(ids))
    if not len(indices):
        return []
    data = result.keypoints.data
    if data.shape[1:] != (17, 3):
        raise RuntimeError(
            f"Expected COCO pose with confidence [N,17,3], got {tuple(data.shape)}"
        )
    # One combined CPU transfer, normalized there instead of two GPU transfers
    # for xyn and conf. All arithmetic stays off the large image tensors.
    poses = data[indices.tolist()].detach().cpu().numpy().astype(np.float32, copy=True)
    height, width = result.orig_shape
    poses[:, :, 0] /= width
    poses[:, :, 1] /= height
    return [TrackedPerson(int(ids[index]), pose) for index, pose in zip(indices, poses)]


class TrackerTimer:
    """Bracket the installed tracking callback, including its box transfers."""

    def __init__(self, model):
        from ultralytics.trackers.track import (
            on_predict_postprocess_end,
            register_tracker,
        )

        self.started = 0.0
        self.last_ms = 0.0
        register_tracker(model, persist=True)
        callbacks = model.callbacks["on_predict_postprocess_end"]
        index = next(
            i
            for i, cb in enumerate(callbacks)
            if getattr(cb, "func", None) is on_predict_postprocess_end
        )
        callbacks.insert(index, self.begin)
        callbacks.insert(index + 2, self.end)

    def begin(self, _predictor) -> None:
        self.started = time.perf_counter()

    def end(self, _predictor) -> None:
        self.last_ms = (time.perf_counter() - self.started) * 1000
