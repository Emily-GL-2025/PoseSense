# Import jumping-jack reference videos

From the project folder:

```powershell
uv run --locked python standard_import.py "D:\videos\demo.mp4"
uv run --locked python standard_import.py "D:\videos\demo1.mp4" "D:\videos\demo2.mov"
uv run --locked python standard_import.py "D:\videos\jumping_jacks" --device cuda
```

This is a separate command-line program in this project, not a new mode in the workout app. It reuses the project's actual pose conversion, confidence cleaning, configuration, and jumping-jack counter. It never opens the camera, changes thresholds, or writes to your source videos or workout recordings. Existing project dependencies and local model weights are required.

## Inputs and identity

Use frontal, full-body, single-demonstrator clips with the wrists and ankles visible throughout. Trim unrelated activity and camera cuts first. File and folder inputs can be combined; folder discovery is non-recursive. Supported extensions: MP4, MOV, AVI, MKV, WEBM, M4V, subject to installed decoder support.

The first **single tracked person** becomes the selected demonstrator. Multiple tracked people before selection cause an error displaying their IDs: crop the video or explicitly select `--track-id 7`. This option applies separately to every input clip; tracker IDs are not persistent human identities. Use separate runs if clips need different IDs.

Once selected, the importer never silently changes to another ID. Missing poses remain missing; the same ID can return. A fresh model/tracker is created for each clip. Tracking can still swap identities behind the same ID: visually review the saved skeletons, especially when other people are visible. Cuts should be separate clips. No gesture or live-session duration limit is imposed.

Defaults follow `workout_session.py`: `yolo26m-pose.pt`, `imgsz=640`, auto device and tracker, and `quantize=16`. Offline inference disables compilation to avoid its startup overhead. Use `--model existing-pose.pt`, `--imgsz 960`, `--device cpu|cuda|auto`, or `--tracker auto|fasttrack|bytetrack` to override. Model files must already exist; the script does not request a download.

## Outputs

Each invocation creates a new `standards/import_<UTC timestamp>/` folder:

- `report.html`: offline report with inline SVG histograms, current threshold markers, pass/fail/N/A tables, timing diagnostics and independent-label comparisons. Open it in a browser.
- `report.json`: full per-clip and pooled measurements, configuration, software versions, model/source hashes, and source video path/size/mtime.
- `001_<video>.npz` and `001_<video>.json`: per-video arrays and report, also retained if a later video fails.

NPZ arrays use no pickle:

| Array | Meaning |
| --- | --- |
| `raw_poses[N,17,3]` | Normalized model output x, y, confidence, before the application's confidence mask. Not ground-truth annotations; Ultralytics may already have processed its output. |
| `poses[N,17,3]` | Exact live `clean_pose` behavior: unreliable coordinates become NaN; confidence is retained. |
| `timestamps`, `decoder_timestamps_ms` | Chosen clip-relative analysis clock and original decoder times. |
| `present`, `observed_track_ids`, `tracked_people_count` | Tracking quality; missing selected ID is -1 and its entire raw pose is NaN. |
| `metrics`, `metric_names` | Shoulder width, vertical shoulder-to-ankle span, ankle ratio, signed wrist offsets and body-relative ratios, minimum required confidence. |
| `gates`, `gate_names` | Exact current checks: 1 pass, 0 fail, -1 not evaluable. |
| `candidate`, `state`, `rep_counts` | Instantaneous posture, confirmed state and offline counter results. |
| `pending`, `pending_held_seconds` | Actual replay counter's current confirmation candidate and observed hold time. |
| `reference_phase` | Your independent OPEN/CLOSED labels, otherwise UNLABELED. |

The archives are compatible with the existing viewers:

```powershell
uv run --locked python sessions/playback.py "standards\import_<timestamp>\001_demo.npz"
uv run --locked python sessions/debug.py "standards\import_<timestamp>\001_demo.npz"
```

Reports provide count, mean, population standard deviation, min, P05/P25/median/P75/P95/max and a 24-bin histogram. All 17 joints have reliable x/y distributions and confidence distributions for present frames, including below-threshold confidence values. A reliable joint has finite x/y/confidence and confidence at least the current threshold. Body-relative geometry requires all six analyzer joints to be valid; shoulder width and span are reported before the additional geometry rejection.

Coordinates are x/image width and y/image height, with y increasing downward. Raw x/y distributions depend on framing and should not be treated as universal form limits. Feet use ankle horizontal separation / shoulder horizontal separation. Wrist ratios use (wrist y - same-side shoulder y) / vertical shoulder-to-ankle span. These match the analyzer and are more comparable across video resolutions, but remain sensitive to perspective, occlusion and pose-estimation error.

## Independent standard-pose labels

Without labels, the script can describe distributions and current behavior, but **cannot determine which frames are actually correct OPEN/CLOSED poses**. In particular, `rule_OPEN` and `rule_CLOSED` groups are selected using the thresholds under examination. Their passing values are not independent validation.

For a meaningful reference comparison, manually inspect the original video and create a JSON file such as:

```json
{
  "demo.mp4": [
    {"start": 0.0, "end": 0.25, "phase": "CLOSED"},
    {"start": 0.50, "end": 0.70, "phase": "OPEN"},
    {"start": 0.95, "end": 1.20, "phase": "CLOSED"}
  ]
}
```

These are example times, not labels for your footage. Use intervals for clearly demonstrated endpoint poses, not entire cycles; leave transitions unlabeled. Times are seconds relative to the first decoded frame, intervals include start and exclude end, and overlapping intervals are rejected. Filename keys must uniquely match an input video. Other videos may remain unlabeled.

```powershell
uv run --locked python standard_import.py "D:\videos\demo.mp4" --labels "D:\videos\labels.json"
```

The `reference_OPEN` and `reference_CLOSED` groups then show how the current gates treat those independent samples. The recognition summary counts missing/invalid frames as misses. Individual spatial gate percentages exclude N/A from their denominator, with the N/A count shown explicitly. A missed joint cannot pass the overall posture check.

The `both_wrists_above_open_margin` group is a diagnostic cross-check: its ankle-ratio distribution can reveal an overly restrictive foot gate when both wrists meet the current arm rule. It is still threshold-conditioned, not independent ground truth.

## Spatial and temporal interpretation

All seven current `JumpingJackConfig` values are captured in every report:

- Six required joints must be finite and meet `confidence`; shoulder width must be at least `minimum_shoulder_width`; vertical span must be positive.
- CLOSED requires feet ratio at most `closed_ankle_ratio` and both wrist offsets strictly greater than `wrist_margin × span`.
- OPEN requires feet ratio at least `open_ankle_ratio` and both offsets strictly less than `-wrist_margin × span`.
- Each candidate must persist for `confirm_seconds`. Reports include candidate-run spans and the number too short to confirm.
- A sample gap greater than `max_sample_gap` resets phase; reports count those gaps.

A rep requires confirmed CLOSED → OPEN → CLOSED. Starting OPEN alone cannot arm counting. The dead band clears the pending confirmation but retains the stable phase; invalid/missing samples and identity changes reset the phase. The importer uses the real counter, not a separate approximation of its temporal logic.

Timing uses monotonically increasing decoder timestamps, rebased to clip start. If those are unusable, the entire clip uses frame index / metadata FPS, with a warning. `--fps 30` explicitly forces constant-FPS timing. For variable-frame-rate footage, fallback timing is not reliable; repair timestamps or use a suitable constant-rate reference clip before interpreting temporal thresholds. Playback speed edits and slow motion are measured as encoded, not as the original movement speed.

No processing-wall-clock timestamps, live-worker frame dropping, start gestures or session duration limit are simulated. Offline rep counts therefore need not match the live application. Run durations are last minus first matching sample time, not interpolated durations; a single-sample endpoint has zero observed hold time.

## Limits and practical review

Use multiple people, cameras and repetitions across separate well-framed clips before drawing calibration conclusions. Pooled distributions are **frame-weighted**, not person-, video-, time- or repetition-balanced; long holds and high-FPS clips dominate. Frames are correlated, so the report does not present statistical confidence intervals or automatic recommendations.

Check each video's missingness, tracker identity, reference labels, short candidate runs and failures before considering a threshold change. The current analyzer does not validate airtime, elbow/knee extension, or overall exercise safety. Positive endpoint demos alone cannot measure false-positive behavior on incorrect exercises; include separately reviewed negative examples before loosening rules. No automatic threshold updates are performed.

Use `--max-frames 120` for a short import check; capped clips are flagged. Long imports currently retain compact per-frame keypoints and diagnostic arrays in memory (not video frames), so split very long videos if memory becomes a concern. An unexpected decode stop may be indistinguishable from EOF; a discrepancy with reported frame count is flagged.
