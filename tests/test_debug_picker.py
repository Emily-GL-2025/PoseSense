"""Keyboard file switching without a GUI, inference, or camera."""

from argparse import Namespace
from dataclasses import replace

import cv2
import numpy as np
import pytest

from sessions.debug import HEIGHT, WIDTH, Dataset, DebugPlayer
from workout.analysis.jumping_jack import JumpingJackConfig


def archive(path, pose_factory, count=6):
    poses = [
        pose_factory("open" if 2 <= i < 4 else "closed").pose for i in range(count)
    ]
    np.savez(
        path,
        poses=poses,
        timestamps=np.arange(count) * 0.1,
        present=np.ones(count, bool),
        frame_indices=np.arange(count),
        observed_track_ids=np.full(count, 7),
        rep_counts=np.zeros(count, int),
        camera_width=1280,
        camera_height=720,
        session_duration=count * 0.1,
    )
    return path


@pytest.fixture
def player(tmp_path, pose_factory):
    path = archive(tmp_path / "b.npz", pose_factory)
    args = Namespace(
        speed=2.0,
        loop=True,
        start=0.1,
        paused=False,
        indices=True,
        max_width=WIDTH,
        max_height=HEIGHT,
    )
    return DebugPlayer(Dataset.load(path), JumpingJackConfig(), args)


def test_navigation_cancel_and_refresh(player, tmp_path, pose_factory):
    archive(tmp_path / "a.npz", pose_factory)
    (tmp_path / "ignored.txt").touch()
    (tmp_path / "folder.npz").mkdir()
    assert player.key(ord("F"))
    assert [p.name for p in player.picker_files] == ["a.npz", "b.npz"]
    assert player.picker_index == 1 and not player.clock.playing
    position = player.clock.position
    player.key(2490368)
    assert player.picker_index == 0
    player.key(ord("W"))
    assert player.picker_index == 0
    player.key(2621440)
    assert player.picker_index == 1
    player.key(ord("s"))
    assert player.picker_index == 1
    assert player.key(27)
    assert not player.picker_open and player.clock.playing
    assert player.clock.position == position
    archive(tmp_path / "c.npz", pose_factory)
    player.key(ord("f"))
    assert len(player.picker_files) == 3
    player.key(27)
    assert player.key(27) is False


def test_switch_is_atomic_and_preserves_settings(player, tmp_path, pose_factory):
    other = archive(tmp_path / "a.npz", pose_factory, count=3)
    player.cfg = replace(player.cfg, open_ankle_ratio=1.6)
    player.guides = False
    player.selected = 4
    player.seek_frame(5)
    player.key(ord("f"))
    player.key(ord("w"))
    original = player.data
    player.key(13)
    assert player.data is original  # indicator is displayed before loading
    assert player.loading_path == other
    assert player.frame().shape == (HEIGHT, WIDTH, 3)
    player.load_pending()
    assert player.data.recording.path == other
    assert len(player.rows) == 3 and player.index == 0 and player.manual_frame is None
    assert player.clock.position == 0 and not player.clock.playing
    assert player.clock.speed == 2 and player.clock.loop
    assert player.cfg.open_ankle_ratio == 1.6
    assert player.indices and not player.guides and player.selected == 4
    assert not player.picker_open and player.loading_path is None


@pytest.mark.parametrize("kind", ["corrupt", "schema", "removed", "replay"])
def test_load_failure_keeps_previous_recording(player, tmp_path, monkeypatch, kind):
    path = tmp_path / "a.npz"
    if kind == "schema":
        np.savez(path, unrelated=[1, 2])
    else:
        path.write_bytes(b"not an archive")
    player.key(ord("f"))
    player.key(ord("w"))
    player.key(13)
    if kind == "removed":
        path.unlink()
    elif kind == "replay":
        monkeypatch.setattr(Dataset, "load", lambda _: player.data)

        def fail(*_args):
            raise ValueError("synthetic replay error")

        monkeypatch.setattr("sessions.debug.replay", fail)
    old = (player.data, player.rows, player.strip, player.events, player.clock)
    player.load_pending()
    assert all(
        a is b
        for a, b in zip(
            old, (player.data, player.rows, player.strip, player.events, player.clock)
        )
    )
    assert player.picker_open and "Cannot load" in player.picker_error
    assert player.loading_path is None and not player.clock.playing
    assert player.frame().shape == (HEIGHT, WIDTH, 3)
    player.key(27)
    assert player.clock.playing


def test_modal_controls_and_pagination(player, tmp_path, pose_factory):
    for index in range(16):
        archive(tmp_path / f"x{index:02}.npz", pose_factory)
    player.seek_frame(3)
    player.key(ord("f"))
    cfg = player.cfg
    for key in (ord(" "), ord("."), ord("r"), ord("l"), ord("h")):
        player.key(key)
    player.mouse(cv2.EVENT_LBUTTONDOWN, 1450, 140, 0, None)
    assert player.cfg is cfg and player.index == 3 and not player.clock.playing
    for _ in range(16):
        player.key(ord("s"))
    assert player.picker_index == 16
    assert player.frame().shape == (HEIGHT, WIDTH, 3)
    assert player.key(ord("Q")) is False
    player.key(27)
    assert player.index == 3 and not player.clock.playing


def test_empty_folder_and_listing_failure(player, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "iterdir", lambda _: iter(()))
    player.open_picker()
    player.key(13)
    assert player.loading_path is None
    assert player.frame().shape == (HEIGHT, WIDTH, 3)
    player.cancel_picker()

    def fail(_):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", fail)
    player.open_picker()
    assert "Cannot list folder" in player.picker_error
    assert player.frame().shape == (HEIGHT, WIDTH, 3)


def test_play_renders_loading_before_replay(
    player, tmp_path, pose_factory, monkeypatch
):
    from sessions import debug

    archive(tmp_path / "a.npz", pose_factory)
    keys = iter([ord("f"), ord("w"), 13, -1, ord("q")])
    shown = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *_: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_: shown.append(player.loading_path))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _: next(keys))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    original_replay = debug.replay

    def checked_replay(*args):
        assert shown[-1] == tmp_path / "a.npz"
        return original_replay(*args)

    monkeypatch.setattr(debug, "replay", checked_replay)
    player.play()
    assert player.data.recording.path.name == "a.npz"
    assert not player.clock.playing
