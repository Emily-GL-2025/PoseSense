# Recorded skeleton playback and debugging

Run these commands from the project root after `uv sync --locked`. The GitHub source includes viewers and teaching guides; workout NPZs are excluded. After cloning, run a workout first to create a recording:

```powershell
uv run --locked python workout_session.py
```

## Debugger

Inspect the newest NPZ beside the script:

```powershell
uv run --locked python sessions/debug.py
```

Supply an existing path to choose a file or start time. Replace the example filename with your recording:

```powershell
uv run --locked python sessions/debug.py "sessions/workout_example.npz" --start 10 --paused
uv run --locked python sessions/debug.py "sessions/workout_example.npz" --summary
```

The debugger replays every saved sample through the current jumping-jack counter. It explains joint confidence, posture gates, confirmation timing, phase resets and repetition counts. Saved live totals and offline all-sample totals are kept separate. Editing a threshold pauses and rebuilds the replay in memory; neither source files nor recordings are changed.

Space plays/pauses; `[` and `]` step one sample; A/D seek five seconds; P/N jump to diagnostic events; -/+ changes speed; T toggles guides; I toggles extra landmark IDs; H opens help; B restores current source defaults. Click a threshold and use comma/period or its on-screen -/+ boxes to adjust it. Click the timeline to seek. Q or Escape quits.

F opens the file picker for the current recording's folder. Use Up/Down or W/S to select, Enter to load paused at zero, and Escape to cancel. Threshold edits, playback speed and display settings are retained.

Run the self-test without any recording:

```powershell
uv run --locked python sessions/debug.py --self-test
```

## Simple playback

```powershell
uv run --locked python sessions/playback.py
uv run --locked python sessions/playback.py "sessions/workout_example.npz" --start 10 --speed 0.5 --indices
```

This player draws saved skeletons and saved live counts without rerunning exercise analysis. Anatomical left is cyan and right is orange. Missing people and missing landmarks remain gaps. The original source image's aspect ratio is preserved; no camera-video background is saved in the normal workout recording.

| Control | Action |
| --- | --- |
| Space | Pause/resume; restart when ended |
| A / D | Seek backward/forward five seconds |
| [ / ] | Previous/next sample and pause |
| Click timeline | Seek |
| - / + | Halve/double speed from 0.125x to 8x |
| R / L | Restart / toggle looping |
| I | Toggle extra joint IDs |
| Q / Escape / close window | Exit |

Both viewers use NumPy and OpenCV. They do not open a camera, load YOLO or change recordings. They use saved timestamps; ordinary display playback may skip observations to keep time. Pause and step to inspect every sample. Lowering confidence cannot recover coordinates already saved as NaN.

## Offline learning guides

- [Your debugger screenshot explained simply](debug_screen_guide.html)
- [Jumping-jack analysis](jumping_jack_analysis.html)
- [Flowcharts and step-through example](jumping_jack_flowchart.html)
- [Plain-language pseudocode](jumping_jack_pseudocode.html)
- [Python explained line by line](jumping_jack_python_comments.html)
- [Live application architecture](workout_session_architecture.html)

These HTML guides open directly in a browser. The screenshot guide embeds its own example screenshot and replay data. No separate example NPZ is needed to read it.
