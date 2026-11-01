"""Import local jumping-jack demo videos; never modify live analysis thresholds.

Run --help for options. See standard_import.md for interpretation and labels.
Heavy inference/video imports occur only when importing a video, not on --help.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from workout.analysis.jumping_jack import JumpingJackConfig, JumpingJackCounter
from workout.packets import PosePacket, clean_pose, valid_keypoints
from workout_session import MODEL_PATH
from workout_session import parse_args as workout_args

ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NEEDED = (5, 6, 9, 10, 15, 16)
METRICS = (
    "shoulder_width",
    "shoulder_ankle_span",
    "ankle_ratio",
    "left_wrist_delta",
    "right_wrist_delta",
    "left_wrist_ratio",
    "right_wrist_ratio",
    "minimum_required_confidence",
)
GATES = (
    "required_keypoints",
    "geometry",
    "closed_feet",
    "closed_left_wrist",
    "closed_right_wrist",
    "open_feet",
    "open_left_wrist",
    "open_right_wrist",
    "closed_all",
    "open_all",
)


def parse_args(argv=None):
    defaults = workout_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos", nargs="+", type=Path, help="Local files or folders (non-recursive)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("standards"))
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / MODEL_PATH,
        help="Existing local COCO pose weights",
    )
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=defaults.device
    )
    parser.add_argument(
        "--tracker",
        choices=("auto", "fasttrack", "bytetrack"),
        default=defaults.tracker,
    )
    parser.add_argument(
        "--track-id",
        type=int,
        help="Lock this tracker ID in each clip; no automatic reassignment",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Force constant-rate timestamps; useful if video timing is broken",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional independent OPEN/CLOSED intervals; see standard_import.md",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Per-video cap; 0 imports all decoded frames",
    )
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.max_frames < 0:
        parser.error("imgsz must be positive and max-frames nonnegative")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and positive")
    if args.track_id is not None and args.track_id < 0:
        parser.error("track-id must be nonnegative")
    return args


def resolve_videos(paths):
    videos = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()
            )
            if not found:
                raise ValueError(f"No supported videos in {path}")
            videos.extend(found)
        elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos.append(path)
        else:
            raise ValueError(f"Not a supported local video: {path}")
    return list(dict.fromkeys(p.resolve() for p in videos))


def load_labels(path, videos):
    """Labels are filename-keyed, nonoverlapping [start,end) seconds from clip start."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Labels must be an object keyed by video filename")
    names = Counter(p.name for p in videos)
    result = {}
    for name, intervals in data.items():
        if names[name] != 1:
            raise ValueError(f"Label filename must match exactly one input: {name}")
        if not isinstance(intervals, list):
            raise TypeError(f"Expected interval list for {name}")
        validated = []
        for item in intervals:
            if not isinstance(item, dict):
                raise TypeError(f"Invalid interval for {name}")
            start, end = float(item["start"]), float(item["end"])
            phase = item["phase"]
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f"Invalid label time range for {name}")
            if phase not in ("OPEN", "CLOSED"):
                raise ValueError("Label phase must be OPEN or CLOSED")
            validated.append({"start": start, "end": end, "phase": phase})
        validated.sort(key=lambda item: item["start"])
        if any(a["end"] > b["start"] for a, b in pairwise(validated)):
            raise ValueError(f"Overlapping labels for {name}")
        result[name] = validated
    return result


def video_times(milliseconds, fps, override=None):
    """Prefer strictly increasing decoder times; use one clock for the whole clip."""
    times = np.asarray(milliseconds, dtype=np.float64) / 1000
    if not len(times):
        raise ValueError("Video contains no decoded frames")
    if override is None and np.isfinite(times).all() and (np.diff(times) > 0).all():
        return times - times[0], "decoder_POS_MSEC"
    rate = override if override is not None else fps
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Decoder timestamps and FPS are unusable; provide --fps")
    return np.arange(len(times), dtype=np.float64) / rate, (
        "forced_constant_fps" if override is not None else "fallback_metadata_fps"
    )


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    quantiles = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    # Frame gaps often differ only by floating-point roundoff. NumPy cannot
    # construct 24 distinct edges in such a tiny range; pad the plotting range
    # without altering the measurements or their quantiles.
    lower, upper = float(values.min()), float(values.max())
    padding = max(abs(lower), abs(upper), 1.0) * 1e-6
    histogram_range = (
        (lower - padding, upper + padding)
        if upper - lower < padding
        else (lower, upper)
    )
    counts, edges = np.histogram(values, bins=24, range=histogram_range)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        **dict(
            zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                map(float, quantiles),
            )
        ),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }


def measure(pose, cfg):
    """Match the analyzer's float32 subtraction order and strict wrist comparisons."""
    metrics = np.full(len(METRICS), np.nan)
    gates = np.full(len(GATES), -1, dtype=np.int8)  # -1 means not evaluable
    confidence = pose[list(NEEDED), 2]
    if np.isfinite(confidence).all():
        metrics[7] = float(confidence.min())
    valid = valid_keypoints(pose, NEEDED, cfg.confidence)
    gates[0] = int(valid)
    if not valid:
        return metrics, gates, "INVALID"
    width = abs(float(pose[5, 0] - pose[6, 0]))
    span = float((pose[15, 1] + pose[16, 1] - pose[5, 1] - pose[6, 1]) / 2)
    metrics[:2] = width, span
    gates[1] = int(width >= cfg.minimum_shoulder_width and span > 0)
    if not gates[1]:
        return metrics, gates, "INVALID"
    ratio = abs(float(pose[15, 0] - pose[16, 0])) / width
    left = float(pose[9, 1] - pose[5, 1])
    right = float(pose[10, 1] - pose[6, 1])
    margin = cfg.wrist_margin * span
    metrics[2:7] = ratio, left, right, left / span, right / span
    gates[2:5] = ratio <= cfg.closed_ankle_ratio, left > margin, right > margin
    gates[5:8] = ratio >= cfg.open_ankle_ratio, left < -margin, right < -margin
    gates[8:] = bool(gates[2:5].all()), bool(gates[5:8].all())
    return metrics, gates, "CLOSED" if gates[8] else "OPEN" if gates[9] else "DEAD_BAND"


def analyze(raw_poses, times, track_ids, labels, cfg):
    raw = np.asarray(raw_poses, dtype=np.float32)
    poses = np.stack([clean_pose(p) for p in raw])
    present = np.asarray(track_ids) >= 0
    rows, checks, candidates, states, reps, pending, held, generations = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    reference = np.full(len(times), "UNLABELED", dtype="<U9")
    for item in labels:
        reference[(times >= item["start"]) & (times < item["end"])] = item["phase"]
    counter = JumpingJackCounter(cfg)
    generation, previous_id = 0, -1
    runs, active_run = [], None
    for i, (pose, timestamp, track_id) in enumerate(zip(poses, times, track_ids)):
        if int(track_id) != previous_id:
            generation += 1
        previous_id = int(track_id)
        metrics, gates, candidate = measure(pose, cfg)
        count_before = counter.rep_count
        counter.update(
            PosePacket(
                i,
                float(timestamp),
                int(track_id),
                pose,
                bool(present[i]),
                generation,
                count_before,
            )
        )
        rows.append(metrics)
        checks.append(gates)
        candidates.append(candidate)
        states.append(counter.state.value)
        reps.append(counter.rep_count)
        pending.append(counter.pending.value if counter.pending else "NONE")
        held.append(
            float(timestamp - counter.pending_since) if counter.pending else 0.0
        )
        generations.append(generation)
        continuity = (
            i > 0
            and 0 <= times[i] - times[i - 1] <= cfg.max_sample_gap
            and generations[i] == generations[i - 1]
        )
        if active_run is not None and (
            candidate != active_run["phase"] or not continuity
        ):
            runs.append(active_run)
            active_run = None
        if candidate in ("OPEN", "CLOSED"):
            if active_run is None:
                active_run = {
                    "phase": candidate,
                    "start": float(timestamp),
                    "end": float(timestamp),
                    "samples": 0,
                }
            active_run["end"] = float(timestamp)
            active_run["samples"] += 1
    if active_run is not None:
        runs.append(active_run)
    for run in runs:
        run["observed_span_seconds"] = run["end"] - run["start"]
        run["meets_confirmation"] = run["observed_span_seconds"] >= cfg.confirm_seconds
    arrays = {
        "raw_poses": raw,
        "poses": poses,
        "timestamps": np.asarray(times),
        "present": present,
        "observed_track_ids": np.asarray(track_ids, dtype=np.int64),
        "frame_indices": np.arange(len(times)),
        "generations": np.asarray(generations),
        "rep_counts": np.asarray(reps),
        "metrics": np.asarray(rows),
        "gates": np.asarray(checks),
        "candidate": np.asarray(candidates),
        "state": np.asarray(states),
        "pending": np.asarray(pending),
        "pending_held_seconds": np.asarray(held),
        "reference_phase": reference,
        "metric_names": np.asarray(METRICS),
        "gate_names": np.asarray(GATES),
    }
    return arrays, runs


def summarize(arrays, cfg):
    metrics, gates = arrays["metrics"], arrays["gates"]
    groups = {}
    # Reference labels and rule-selected frames are deliberately distinct.
    for name, mask in {
        "all_frames": np.ones(len(metrics), dtype=bool),
        "reference_OPEN": arrays["reference_phase"] == "OPEN",
        "reference_CLOSED": arrays["reference_phase"] == "CLOSED",
        "rule_OPEN": arrays["candidate"] == "OPEN",
        "rule_CLOSED": arrays["candidate"] == "CLOSED",
        "both_wrists_above_open_margin": (gates[:, 6:8] == 1).all(axis=1),
    }.items():
        groups[name] = {
            "frames": int(mask.sum()),
            "distributions": {
                metric: distribution(metrics[mask, i])
                for i, metric in enumerate(METRICS)
            },
            "gates": {
                gate: {
                    "pass": int((gates[mask, i] == 1).sum()),
                    "fail": int((gates[mask, i] == 0).sum()),
                    "not_evaluable": int((gates[mask, i] < 0).sum()),
                }
                for i, gate in enumerate(GATES)
            },
        }
    joints = {}
    for j, name in enumerate(JOINTS):
        raw = arrays["raw_poses"][:, j]
        good = (
            np.isfinite(raw).all(axis=1)
            & (raw[:, 2] >= cfg.confidence)
            & arrays["present"]
        )
        joints[name] = {
            "reliable_frames": int(good.sum()),
            "x": distribution(raw[good, 0]),
            "y": distribution(raw[good, 1]),
            "confidence": distribution(raw[arrays["present"], 2]),
        }
    return {
        "frames": len(metrics),
        "present_frames": int(arrays["present"].sum()),
        "missing_frames": int((~arrays["present"]).sum()),
        "candidates": dict(Counter(map(str, arrays["candidate"]))),
        "keypoint_distributions": joints,
        "groups": groups,
        "reference_recognition": {
            phase: {
                "labeled_frames": int((arrays["reference_phase"] == phase).sum()),
                "recognized_frames": int(
                    (
                        (arrays["reference_phase"] == phase)
                        & (arrays["candidate"] == phase)
                    ).sum()
                ),
                "missed_including_invalid": int(
                    (
                        (arrays["reference_phase"] == phase)
                        & (arrays["candidate"] != phase)
                    ).sum()
                ),
            }
            for phase in ("OPEN", "CLOSED")
        },
    }


def import_video(path, args, intervals, cfg):
    import cv2
    from ultralytics import YOLO

    from smoke_pose5 import choose_device
    from workout.tracking import select_tracker, tracked_people

    tracker = select_tracker(args.tracker)
    device = choose_device(args.device)
    capture = cv2.VideoCapture(str(path))
    raw, ids, milliseconds, people_counts = [], [], [], []
    selected = args.track_id
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        # A fresh predictor/tracker per clip prevents identity state leaking across videos.
        model = YOLO(str(args.model))
        if model.task != "pose":
            raise ValueError("Model must be a COCO 17-keypoint pose model")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        estimated_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        dimensions = None
        while not args.max_frames or len(raw) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if dimensions is None:
                dimensions = (width, height)
            elif dimensions != (width, height):
                raise ValueError(
                    "Video resolution changed; split it into consistent clips"
                )
            milliseconds.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            result = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                device=device,
                imgsz=args.imgsz,
                quantize=16,
                compile=False,
                verbose=False,
            )[0]
            people = tracked_people(result)
            people_counts.append(len(people))
            if selected is None and people:
                if len(people) != 1:
                    raise ValueError(
                        f"{path.name}: multiple initial tracked people {[p.track_id for p in people]}; "
                        "crop to a single demonstrator or choose --track-id"
                    )
                selected = people[0].track_id
            person = next((p for p in people if p.track_id == selected), None)
            raw.append(
                person.pose.copy()
                if person is not None
                else np.full((17, 3), np.nan, np.float32)
            )
            ids.append(person.track_id if person is not None else -1)
            if len(raw) % 150 == 0:
                print(f"  {path.name}: {len(raw)} frames", flush=True)
    finally:
        capture.release()
    times, clock = video_times(milliseconds, fps, args.fps)
    arrays, runs = analyze(raw, times, ids, intervals, cfg)
    arrays["decoder_timestamps_ms"] = np.asarray(milliseconds)
    arrays["tracked_people_count"] = np.asarray(people_counts)
    arrays["camera_width"], arrays["camera_height"] = map(np.asarray, dimensions)
    duration = float(times[-1])
    arrays["session_duration"] = np.asarray(duration)
    warnings = []
    if clock != "decoder_POS_MSEC":
        warnings.append(
            "Constant-FPS timing used; variable-rate temporal conclusions may be inaccurate."
        )
    if not arrays["present"].any():
        warnings.append(
            "No selected person was tracked; this clip cannot calibrate pose thresholds."
        )
    if np.any(np.asarray(people_counts) > 1):
        warnings.append(
            "Multiple people visible; inspect playback for tracker identity switches."
        )
    if args.max_frames and len(raw) >= args.max_frames:
        warnings.append("Frame cap reached: this may be only a partial clip.")
    elif math.isfinite(estimated_frames) and estimated_frames > len(raw) + 1:
        warnings.append(
            "Decoded fewer frames than metadata advertises; EOF or decoder failure is indistinguishable."
        )
    if any(
        item["end"] > duration + (1 / fps if math.isfinite(fps) and fps > 0 else 0)
        for item in intervals
    ):
        warnings.append("Some label intervals extend beyond decoded video coverage.")
    metadata = {
        "source_video": str(path),
        "source_bytes": path.stat().st_size,
        "source_mtime_ns": path.stat().st_mtime_ns,
        "selected_track_id": selected,
        "camera_width": dimensions[0],
        "camera_height": dimensions[1],
        "source_fps": fps if math.isfinite(fps) else None,
        "decoded_frames": len(raw),
        "observed_span_seconds": duration,
        "clock": clock,
        "inference": {
            "model": str(args.model.resolve()),
            "imgsz": args.imgsz,
            "device": str(device),
            "tracker": tracker,
            "quantize": 16,
            "compile": False,
        },
        "analysis_config": asdict(cfg),
        "final_rep_count": int(arrays["rep_counts"][-1]),
        "analysis_mode": "offline_all_decoded_frames",
        "labels": intervals,
        "warnings": warnings,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    arrays["events_json"] = np.asarray("[]")
    report = summarize(arrays, cfg)
    report.update(metadata)
    report["candidate_runs"] = runs
    report["timing"] = {
        "sample_gap_seconds": distribution(np.diff(times)),
        "gaps_exceeding_max": int((np.diff(times) > cfg.max_sample_gap).sum()),
        "runs": {
            phase: {
                "count": sum(r["phase"] == phase for r in runs),
                "shorter_than_confirmation": sum(
                    r["phase"] == phase and not r["meets_confirmation"] for r in runs
                ),
                "observed_span_seconds": distribution(
                    [r["observed_span_seconds"] for r in runs if r["phase"] == phase]
                ),
            }
            for phase in ("OPEN", "CLOSED")
        },
    }
    return arrays, report


def histogram_svg(stats, thresholds=()):
    if not stats["n"]:
        return "<p>No evaluable samples.</p>"
    counts, edges = stats["histogram"]["counts"], stats["histogram"]["edges"]
    lo, hi = (
        min(edges[0], *thresholds) if thresholds else edges[0],
        max(edges[-1], *thresholds) if thresholds else edges[-1],
    )

    def x(value):
        return 35 + (value - lo) / (hi - lo) * 470

    parts = [
        '<svg viewBox="0 0 540 150" role="img" aria-label="Frame distribution; red lines mark current thresholds">'
    ]
    for i, count in enumerate(counts):
        height = 100 * count / max(counts)
        parts.append(
            f'<rect x="{x(edges[i]):.2f}" y="{115 - height:.2f}" width="{max(0.3, x(edges[i + 1]) - x(edges[i]) - 0.5):.2f}" height="{height:.2f}" fill="#48c9bd"/>'
        )
    for threshold in thresholds:
        parts.append(
            f'<path d="M{x(threshold):.2f} 12V118" stroke="#ff847b" stroke-dasharray="4 3"/><text x="{x(threshold):.2f}" y="10" text-anchor="middle">{threshold:g}</text>'
        )
    parts.append(
        f'<text x="35" y="140">{lo:.3g}</text><text x="505" y="140" text-anchor="end">{hi:.3g}</text></svg>'
    )
    return "".join(parts)


def render_html(report):
    cfg = report["analysis_config"]
    esc = lambda value: html.escape(str(value))
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jumping-jack reference import</title><style>body{font:16px system-ui;background:#111c29;color:#e6eef7;max-width:1100px;margin:auto;padding:28px}h1,h2,h3{color:#75ddd0}section{background:#1c2b3b;padding:20px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #3b4a5c}svg{width:100%;max-width:540px}svg text{fill:#e6eef7;font-size:11px}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}code{overflow-wrap:anywhere}a{color:#75ddd0}</style><h1>Jumping-jack reference import</h1>'
    ]
    parts.append(
        "<p>Reference demos are model observations, not ground-truth joint positions. Histograms are frame-weighted: longer clips and held poses contribute more samples. No thresholds were changed.</p>"
    )
    parts.append(
        "<p>Independent reference_OPEN / reference_CLOSED groups come only from your optional labels. Rule groups are selected by current thresholds and cannot independently validate them. Unlabeled footage may include transitions, poor form, or unrelated activity.</p>"
    )
    parts.append(
        f"<section><h2>Current analyzer configuration</h2><pre>{esc(json.dumps(cfg, indent=2))}</pre><p>Feet: CLOSED ratio &le; closed_ankle_ratio; OPEN ratio &ge; open_ankle_ratio. Wrists: both delta/span &gt; +wrist_margin for CLOSED, both &lt; -wrist_margin for OPEN. y increases downward. All six required joints must pass confidence; shoulder width must meet its minimum and vertical span must be positive.</p><p>A rep requires confirmed CLOSED → OPEN → CLOSED. Candidate holds must span confirm_seconds. Dead band cancels the pending hold but preserves stable phase; invalid/missing poses or a gap above max_sample_gap reset phase. Replay consumes all decoded frames, not the live worker's delivered subset. Candidate-run spans are observed endpoint differences, not interpolated true pose durations.</p></section>"
    )
    entries = [
        ("Pooled distributions (not a continuous session)", report["pooled"])
    ] + [(r["source_video"], r) for r in report["clips"]]
    for name, entry in entries:
        parts.append(
            f"<section><h2>{esc(name)}</h2><p>{entry['frames']} frames · {entry['present_frames']} present · {entry['missing_frames']} missing</p>"
        )
        if "final_rep_count" in entry:
            parts.append(
                f"<p>Offline count: {entry['final_rep_count']} · clock: {esc(entry['clock'])} · selected tracker ID: {entry['selected_track_id']}</p>"
            )
            parts.extend(f"<p>Warning: {esc(w)}</p>" for w in entry["warnings"])
            parts.append(
                "<h3>Timing checks</h3><pre>"
                + esc(
                    json.dumps(
                        {
                            p: {
                                k: v
                                for k, v in d.items()
                                if k != "observed_span_seconds"
                            }
                            for p, d in entry["timing"]["runs"].items()
                        },
                        indent=2,
                    )
                )
                + f"</pre><p>Gaps above {cfg['max_sample_gap']}s: {entry['timing']['gaps_exceeding_max']}</p>"
            )
        parts.append(
            "<h3>Independent label comparison</h3><table><tr><th>Reference</th><th>Labeled frames</th><th>Recognized</th><th>Missed (includes invalid)</th></tr>"
        )
        for phase, values in entry["reference_recognition"].items():
            parts.append(
                f"<tr><td>{phase}</td><td>{values['labeled_frames']}</td><td>{values['recognized_frames']}</td><td>{values['missed_including_invalid']}</td></tr>"
            )
        parts.append(
            "</table><p>Zero labeled frames means this reference test was not performed, not that the thresholds are correct.</p>"
        )
        for group, data in entry["groups"].items():
            if not data["frames"]:
                continue
            parts.append(
                f"<h3>{esc(group)} · {data['frames']} frames</h3><div class=charts>"
            )
            for metric, marks in (
                ("ankle_ratio", (cfg["closed_ankle_ratio"], cfg["open_ankle_ratio"])),
                ("left_wrist_ratio", (-cfg["wrist_margin"], cfg["wrist_margin"])),
                ("right_wrist_ratio", (-cfg["wrist_margin"], cfg["wrist_margin"])),
            ):
                stats = data["distributions"][metric]
                detail = f"n={stats['n']}"
                if stats["n"]:
                    detail += f" · P05 {stats['p05']:.3f} · median {stats['median']:.3f} · P95 {stats['p95']:.3f}"
                parts.append(
                    f"<div><p>{metric}<br>{detail}</p>{histogram_svg(stats, marks)}</div>"
                )
            parts.append(
                "</div><table><tr><th>Check</th><th>Pass</th><th>Fail</th><th>N/A</th><th>Pass / evaluable</th></tr>"
            )
            for gate, values in data["gates"].items():
                denominator = values["pass"] + values["fail"]
                rate = f"{values['pass'] / denominator:.1%}" if denominator else "N/A"
                parts.append(
                    f"<tr><td>{gate}</td><td>{values['pass']}</td><td>{values['fail']}</td><td>{values['not_evaluable']}</td><td>{rate}</td></tr>"
                )
            parts.append("</table>")
            parts.append(
                "<details><summary>All measurement distributions</summary><table><tr><th>Metric</th><th>n</th><th>P05</th><th>Median</th><th>P95</th></tr>"
            )
            for metric, stats in data["distributions"].items():
                cells = "".join(
                    f"<td>{stats[k]:.4g}</td>" if stats["n"] else "<td>N/A</td>"
                    for k in ("p05", "median", "p95")
                )
                parts.append(f"<tr><td>{metric}</td><td>{stats['n']}</td>{cells}</tr>")
            parts.append("</table></details>")
        parts.append("</section>")
    parts.append(
        "<p>JSON includes all 17 joints' reliable x/y and raw confidence distributions, all geometry metrics, timing runs, provenance and explicit denominators. NPZ retains every frame for playback/debugging. Spatial N/A rows do not imply a pass: missing/invalid frames fail reference pose recognition.</p></html>"
    )
    return "".join(parts)


def main(argv=None):
    args = parse_args(argv)
    try:
        videos = resolve_videos(args.videos)
        labels = load_labels(args.labels, videos)
        if not args.model.is_file():
            raise ValueError(
                f"Local model not found: {args.model}; no automatic download is performed"
            )
        cfg = JumpingJackConfig()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / datetime.now(UTC).strftime(
            "import_%Y%m%dT%H%M%S_%fZ"
        )
        output.mkdir()  # never reuse or overwrite an existing import
        print(f"Output: {output.resolve()}", flush=True)
        reports, pooled = [], []
        for index, path in enumerate(videos, 1):
            print(f"[{index}/{len(videos)}] {path.name}", flush=True)
            arrays, report = import_video(path, args, labels.get(path.name, []), cfg)
            filename = f"{index:03d}_{path.stem}.npz"
            with (output / filename).open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            report["keypoints_file"] = filename
            (output / f"{index:03d}_{path.stem}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            reports.append(report)
            pooled.append(
                {
                    key: arrays[key]
                    for key in (
                        "raw_poses",
                        "present",
                        "metrics",
                        "gates",
                        "reference_phase",
                        "candidate",
                    )
                }
            )
            print(
                f"  Offline reps: {report['final_rep_count']}; missing: {report['missing_frames']}/{report['frames']}",
                flush=True,
            )
            for warning in report["warnings"]:
                print(f"  Warning: {warning}", flush=True)
        import importlib.metadata

        with args.model.open("rb") as model_file:
            model_hash = hashlib.file_digest(model_file, "sha256").hexdigest()
        report = {
            "schema_version": 1,
            "analysis_config": asdict(cfg),
            "command": sys.argv if argv is None else argv,
            "provenance": {
                "source_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in (
                        "workout_session.py",
                        "workout/analysis/jumping_jack.py",
                        "workout/packets.py",
                        "workout/tracking.py",
                        "standard_import.py",
                    )
                },
                "model_sha256": model_hash,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "opencv-python", "ultralytics", "torch")
                },
            },
            "clips": reports,
            "pooled": summarize(
                {key: np.concatenate([a[key] for a in pooled]) for key in pooled[0]},
                cfg,
            ),
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "report.html").write_text(render_html(report), encoding="utf-8")
        print(f"Open {output.resolve() / 'report.html'}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(
            f"Import failed: {exc}. Any completed clip archives remain in the printed output folder.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
