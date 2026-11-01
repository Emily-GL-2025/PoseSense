"""Small optional timing collector. JSON is written only after capture ends."""

import json
import time
from pathlib import Path

import numpy as np


class FrameMetrics:
    def __init__(self, warmup: int = 30):
        self.warmup = warmup
        self.seen = 0
        self.rows: list[dict] = []
        self.wall_start: float | None = None
        self.cpu_start: float | None = None
        self.cpu_seconds = 0.0

    def add(self, **row) -> None:
        self.seen += 1
        if self.seen <= self.warmup:
            return
        if self.wall_start is None:
            self.wall_start = time.perf_counter()
            self.cpu_start = time.process_time()
        self.rows.append(row)
        self.cpu_seconds = time.process_time() - self.cpu_start

    def summary(self, **metadata) -> dict:
        means = (
            {
                key: float(np.mean([row[key] for row in self.rows]))
                for key in self.rows[0]
            }
            if self.rows
            else {}
        )
        loop_ms = means.get("loop_ms", 0)
        wall_seconds = sum(row["loop_ms"] for row in self.rows) / 1000
        return {
            **metadata,
            "warmup_frames": self.warmup,
            "measured_frames": len(self.rows),
            "total_processed_frames": self.seen,
            "average_loop_fps": 1000 / loop_ms if loop_ms else 0,
            "mean_ms": means,
            "loop_p95_ms": float(np.percentile([r["loop_ms"] for r in self.rows], 95))
            if self.rows
            else 0,
            "main_process_cpu_percent_one_core": 100 * self.cpu_seconds / wall_seconds
            if wall_seconds
            else 0,
            "cpu_note": "Main-process CPU time only; 100% means one core; excludes analysis process.",
        }

    def save(self, path: str | Path, **metadata) -> dict:
        result = self.summary(**metadata)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return result
