import numpy as np
import pytest

from sessions.playback import (
    PlaybackClock,
    Recording,
    display_size,
    projected_points,
    render_frame,
)


def make_recording(tmp_path, timestamps=(0.0, 0.1, 1.2), duration=1.4):
    path = tmp_path / "sample.npz"
    poses = np.full((3, 17, 3), 0.5, np.float32)
    poses[:, :, 2] = 0.9
    poses[1] = np.nan
    np.savez(
        path,
        poses=poses,
        timestamps=timestamps,
        present=[True, False, True],
        frame_indices=[10, 11, 12],
        observed_track_ids=[7, -1, 7],
        rep_counts=[0, 0, 1],
        camera_width=1280,
        camera_height=720,
        session_duration=duration,
    )
    return Recording.load(path)


def test_real_timestamp_lookup_preserves_missing_gap(tmp_path):
    recording = make_recording(tmp_path)
    assert recording.frame_at(0.09) == 0
    assert recording.frame_at(0.1) == 1
    assert recording.frame_at(1.19) == 1
    assert not recording.present[recording.frame_at(1.19)]
    assert recording.frame_at(1.2) == 2
    assert recording.frame_at(10) == 2


def test_playback_clock_speed_pause_seek_end_and_loop():
    clock = PlaybackClock(10, speed=2)
    clock.seek(0, now=100)
    assert clock.tick(101) == 2
    clock.playing = False
    assert clock.tick(105) == 2
    clock.seek(6, now=110)
    clock.playing = True
    assert clock.tick(111) == 8
    assert clock.tick(112) == 10 and not clock.playing
    clock.loop = True
    clock.playing = True
    clock.seek(9, now=120)
    assert clock.tick(121) == 1


def test_normalized_projection_keeps_source_aspect_and_missing_points(tmp_path):
    recording = make_recording(tmp_path)
    size = display_size(recording, 1280, 900)
    recording.poses[0, 0, :2] = (0, 0)
    recording.poses[0, 1, :2] = (1, 1)
    recording.poses[0, 2, 2] = 0.1
    recording.poses[0, 3, 0] = np.nan
    points, valid = projected_points(recording, 0, size, 0.5)
    delta = points[1] - points[0]
    assert delta[0] / delta[1] == pytest.approx(1280 / 720, rel=0.002)
    assert not valid[2] and not valid[3]
    assert not projected_points(recording, 1, size, 0.5)[1].any()


def test_renderer_handles_present_missing_and_unconfident_frames(tmp_path):
    recording = make_recording(tmp_path)
    size = display_size(recording, 640, 500)
    clock = PlaybackClock(recording.duration)
    for index in range(3):
        clock.seek(float(recording.timestamps[index]))
        image = render_frame(recording, index, clock, size, show_indices=True)
        assert image.shape == (size[1], size[0], 3)
        assert image.dtype == np.uint8
    recording.poses[0] = np.nan
    render_frame(recording, 0, clock, size)


@pytest.mark.parametrize(
    "timestamps,duration", [((0, 1, 0.5), 2), ((0, 1, np.nan), 2), ((0, 1, 2), 1)]
)
def test_bad_timeline_has_clear_error(tmp_path, timestamps, duration):
    with pytest.raises(ValueError):
        make_recording(tmp_path, timestamps, duration)
