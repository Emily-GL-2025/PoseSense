# PoseSense

PoseSense is a Python application for camera-based workout tracking and analysis that works offline after initial setup. It currently counts jumping jacks using a webcam, a selected person's body landmarks and a separate CPU analysis process.

Raise your anatomical right hand for two seconds to start a workout. The screen shows repetitions and time remaining in either video or skeleton view. Each workout saves pose data that you can inspect with the offline debugger.

## Requirements and setup

The current dependency configuration is for the tested **Windows / Python 3.12** environment, with PyTorch's CUDA 13.0 packages and Triton Windows. An NVIDIA GPU was used for development; the application's device selector also offers CPU inference. This configuration is not a general installation recipe for Linux or macOS.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the project root:

```powershell
uv sync --locked
```

uv creates a local `.venv` using the committed `pyproject.toml` and `uv.lock`. You do not need to activate it before `uv run`. The tracker dependency `lap==0.5.13` is included in this publishing copy's manifests.

The live application uses `yolo26m-pose.pt`. Ultralytics downloads missing official weights on first use and reuses the saved file afterward. Installing dependencies and downloading weights require internet access; normal local workout tracking can then run offline. Keep model weights out of Git.

## Start a workout

```powershell
uv run --locked python workout_session.py
```

1. Stand facing the camera with shoulders, wrists and ankles visible. The application does not mirror the image.
2. Hold your anatomical **right** hand above your head for two seconds. PoseSense selects that track and starts the workout timer.
3. Lower both hands and bring your feet together to establish CLOSED. A confirmed **CLOSED -> OPEN -> CLOSED** cycle adds one repetition on the final return to CLOSED.
4. The default workout lasts **30 seconds**. At the end it saves once and returns to ready mode. A fresh two-second right-hand hold starts another workout and creates a separate recording.

Choose a different duration or display:

```powershell
uv run --locked python workout_session.py --duration 120
uv run --locked python workout_session.py --view skeleton
uv run --locked python workout_session.py --fullscreen --ui-scale 1.25
uv run --locked python workout_session.py --backend dshow --device cpu
uv run --locked python workout_session.py --help
```

Defaults include camera 0, MSMF capture, a requested 1280 x 720 at 30 FPS, inference image size 640, automatic CUDA/CPU selection, FP16 where supported, and compilation. The first inference can take longer while compilation warms up. `--tracker bytetrack` selects the fallback tracker explicitly.

| Key | Action in the workout window |
| --- | --- |
| V | Switch video / skeleton view |
| D | Show or hide diagnostic details |
| F | Toggle fullscreen |
| + or = / - or _ | Increase / decrease text size |
| Q or Escape | Save any active partial workout and quit |

Closing the window also exits. Quitting while ready creates no recording. Display controls do not change the current exercise state or session timer.

## Tracking loss

The workout timer continues while the selected person is missing. Missing observations are recorded and the accepted live count freezes. The same track can return automatically. Selecting a different track while lost requires a deliberate one-second right-hand hold, which is logged as a reassignment. Tracker IDs are not biometric identities.

## Saved recordings

Recordings are written to `sessions/workout_<UTC timestamp>.npz` by default. Use `--output-dir` to choose another folder.

NPZ files contain normalized body landmarks, timestamps, presence flags, source frame indices, track IDs, continuity generations, saved repetition counts and session metadata. They contain **keypoints, not recorded camera video**. Unusable landmark coordinates are saved as NaN. The main process records each active processed frame before offering its pose to the analysis worker.

Workout recordings are excluded from the GitHub source files. After cloning, record a workout before using normal playback or the debugger. Generated recordings are ignored by Git.

## Offline debugger and playback

Inspect the newest recording next to the debugger:

```powershell
uv run --locked python sessions/debug.py
```

For a specific recording, replace the example filename with an existing NPZ:

```powershell
uv run --locked python sessions/debug.py "sessions/workout_example.npz" --paused
uv run --locked python sessions/debug.py "sessions/workout_example.npz" --summary
```

The debugger uses NumPy, OpenCV and the CPU counter. It does not open a camera or load YOLO. Its controls explain the current pose, required joints, threshold gates, confirmation timing, stable phase, resets and counts. Threshold changes are temporary what-if experiments; they do not edit source code or recordings.

| Key / action | Debugger behavior |
| --- | --- |
| Space | Play / pause |
| [ / ] | Previous / next saved sample and pause |
| A / D | Seek backward / forward five seconds |
| P / N | Previous / next diagnostic event and pause |
| - / + | Halve / double playback speed |
| R / L | Restart / toggle looping |
| T / I / H | Guides / extra joint IDs / help |
| F | Choose another NPZ from the current recording's folder |
| Click a threshold, then , / . | Decrease / increase the selected threshold |
| B | Restore current source defaults |
| Click the timeline | Seek to that time |
| Q / Escape | Quit; Escape cancels when the file picker is open |

The debugger reprocesses **every saved sample** using current source defaults and its own accumulated count. The live worker may have received fewer samples, so its saved count can differ from this offline replay. Seeking or changing playback speed does not change replay results.

Run a synthetic debugger self-test without any recording, model or camera:

```powershell
uv run --locked python sessions/debug.py --self-test
```

For a simpler player that displays saved live counts without reanalysis:

```powershell
uv run --locked python sessions/playback.py
```

See [sessions/README.md](sessions/README.md) for more details.

## Architecture

The main process owns the camera, one YOLO pose/tracking pass per processed frame, selection, workout timing, display and recording. A separate CPU process counts repetitions using small pose/status mailboxes. A slow analysis worker can miss parts of a movement while the saved pose history remains complete for the frames the main loop processed.

| File | Responsibility |
| --- | --- |
| `workout_session.py` | Live entry point and application coordination |
| `smoke_pose5.py` | Original camera demo and helpers imported by the live app |
| `workout/packets.py` | Pose/status objects, landmark indices and confidence cleaning |
| `workout/gesture.py` | Right-hand selection and reacquisition holds |
| `workout/session.py` | Session state, continuity and accepted live counts |
| `workout/recorder.py` | In-memory recording and atomic NPZ saving |
| `workout/tracking.py` | Installed tracker integration and pose extraction |
| `workout/live_display.py` | Live skeleton and spatial diagnostics |
| `workout/workout_display.py` | Workout layout, countdown, sizing and controls |
| `workout/analysis/jumping_jack.py` | Shared posture thresholds and repetition state machine |
| `workout/analysis/worker.py` | CPU process and latest-value communication |
| `workout/benchmark.py` | Timing collector required by the live app |
| `sessions/debug.py`, `sessions/playback.py` | Saved-data inspection and playback |

Both shoulders, wrists and ankles must be usable. The default rules compare horizontal ankle spread with shoulder width: CLOSED requires a ratio at most 1.0 and OPEN requires a ratio at least 1.4. Each wrist's vertical position is compared with its own shoulder. Postures must persist for 0.06 seconds. Invalid required landmarks, person/continuity changes and excessive sample gaps reset the partial cycle.

The counter contains experimental heuristics for a frontal, full-body view. Its counts are algorithm outputs, not independently verified exercise assessments.

## Learning guides

These HTML documents open directly in a browser and work offline:

- [Debugger screenshot explained simply](sessions/debug_screen_guide.html)
- [Jumping-jack analysis and interactive examples](sessions/jumping_jack_analysis.html)
- [Flowcharts and step-through example](sessions/jumping_jack_flowchart.html)
- [Plain-language pseudocode](sessions/jumping_jack_pseudocode.html)
- [Python explained line by line](sessions/jumping_jack_python_comments.html)
- [Workout application architecture](sessions/workout_session_architecture.html)

The screenshot guide embeds the updated screenshot and regenerated example replay data using the current CLOSED 1.0 / OPEN 1.4 settings. The example is independent of the recordings saved by your own workouts. Historical examples in the teaching guides may use different scenario durations; the current live default is 30 seconds.

## Video import and performance tools

`standard_import.py` analyzes local reference videos and writes new archives/reports under `standards/`. It requires the model file to exist locally before import and deliberately does not download it. See [standard_import.md](standard_import.md).

`benchmark_workout.py` runs finite comparisons of the original demo and workout runtime. Camera inference remains real; `--synthetic-session` supplies scripted exercise poses and is benchmark-only. It is not a human accuracy measurement.

```powershell
uv run --locked python benchmark_workout.py --variant workout --headless --frames 150 --warmup 30 --metrics benchmarks/workout.json
```

Generated imports, recordings and benchmark outputs are ignored by Git. The standalone smoke demo uses `yolo26n-pose.pt`, separate from the live workout's medium model.

## Development checks

```powershell
uv run --locked python -m pytest -q
uv run --locked ruff check workout workout_session.py benchmark_workout.py tests
```

Manage new dependencies with `uv add`, and keep `pyproject.toml` and `uv.lock` together in version control. Do not commit `.venv`, downloaded model weights, generated recordings or tool caches.

## Dependency licensing

The repository's own license does not replace dependency and model licenses. Ultralytics supplies its package and official models under AGPL-3.0 or separate enterprise terms; review [Ultralytics licensing](https://www.ultralytics.com/license) before describing the complete application as MIT-only.
