"""Authoritative in-memory history, owned exclusively by the main process."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from workout.packets import PosePacket, clean_pose


class SessionRecorder:
    def __init__(self) -> None:
        self.poses: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.frame_indices: list[int] = []
        self.present: list[bool] = []
        self.track_ids: list[int] = []
        self.rep_counts: list[int] = []
        self.generations: list[int] = []
        self.saved_path: Path | None = None

    def __len__(self) -> int:
        return len(self.poses)

    def append(self, packet: PosePacket, rep_count: int) -> None:
        if self.saved_path is not None:
            raise RuntimeError("Cannot append to a saved session")
        if not np.isfinite(packet.timestamp) or packet.timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if self.timestamps and packet.timestamp < self.timestamps[-1]:
            raise ValueError("Session timestamps must be monotonic")
        if self.frame_indices and packet.frame_index <= self.frame_indices[-1]:
            raise ValueError(
                "Each processed frame may be recorded exactly once, in order"
            )
        # This tiny owned copy is independent of mailbox overwrites and callers.
        self.poses.append(clean_pose(packet.pose if packet.present else None))
        self.timestamps.append(packet.timestamp)
        self.frame_indices.append(packet.frame_index)
        self.present.append(packet.present)
        self.track_ids.append(packet.selected_track_id if packet.present else -1)
        self.rep_counts.append(rep_count)
        self.generations.append(packet.generation)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "poses": np.stack(self.poses)
            if self.poses
            else np.empty((0, 17, 3), np.float32),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int64),
            "present": np.asarray(self.present, dtype=np.bool_),
            "observed_track_ids": np.asarray(self.track_ids, dtype=np.int64),
            "rep_counts": np.asarray(self.rep_counts, dtype=np.int64),
            "generations": np.asarray(self.generations, dtype=np.int64),
        }

    def save(self, output_dir: str | Path, metadata: dict) -> Path:
        """Save once outside the live loop. No object arrays or pickle needed."""
        if self.saved_path is not None:
            return self.saved_path
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        target = directory / f"workout_{stamp}.npz"
        arrays = self.arrays()
        span = self.timestamps[-1] - self.timestamps[0] if len(self) > 1 else 0
        metadata = {
            **metadata,
            "schema_version": 1,
            "recorded_frames": len(self),
            "observed_source_fps": (len(self) - 1) / span if span > 0 else 0.0,
            "coordinate_format": "normalized_x,normalized_y,confidence",
            "keypoint_confidence_threshold": 0.5,
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
        # Convenient scalars plus complete metadata in a single JSON string.
        for key in (
            "camera_width",
            "camera_height",
            "nominal_source_fps",
            "observed_source_fps",
            "selected_track_id_at_start",
            "selected_track_id_at_end",
            "session_duration",
            "target_duration",
            "completed",
            "final_rep_count",
            "stop_reason",
        ):
            arrays[key] = np.asarray(metadata[key])
        arrays["events_json"] = np.asarray(json.dumps(metadata.get("events", [])))
        fd, temporary = tempfile.mkstemp(
            prefix=".workout_", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            # Remove only this exact temporary file; the in-memory history remains.
            Path(temporary).unlink(missing_ok=True)
            raise
        self.saved_path = target
        return target
