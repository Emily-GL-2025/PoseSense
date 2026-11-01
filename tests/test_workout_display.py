"""Workout readability, resize geometry and controls, without opening a GUI."""

import cv2
import numpy as np
import pytest

import workout.workout_display as ui
from workout.packets import PosePacket, clean_pose
from workout.session import SessionController, SessionState
from workout_session import parse_args


def active_session(person):
    session = SessionController()
    session.state = SessionState.ACTIVE_TRACKED
    session.start_time = 100.0
    session.selected_track_id = person.track_id
    session.rep_count = 19
    session.exercise_state = "CLOSED"
    session.last_packet = PosePacket(
        1, 0, person.track_id, clean_pose(person.pose), True, 1
    )
    return session


@pytest.mark.parametrize(
    "now,expected",
    [
        (100, "00:30"),
        (100.01, "00:30"),
        (129, "00:01"),
        (129.99, "00:01"),
        (130, "00:00"),
    ],
)
def test_countdown_never_shows_zero_before_the_deadline(pose_factory, now, expected):
    assert ui.remaining_label(active_session(pose_factory()), now) == expected


@pytest.mark.parametrize("size", [(2560, 1600), (1280, 800), (640, 480)])
@pytest.mark.parametrize("ui_scale", [0.75, 2.0])
@pytest.mark.parametrize(
    "state",
    [SessionState.WAITING, SessionState.ACTIVE_TRACKED, SessionState.ACTIVE_LOST],
)
def test_all_text_stays_inside_its_region_without_mutating_session(
    monkeypatch, pose_factory, size, ui_scale, state
):
    person = pose_factory()
    session = active_session(person)
    session.state = state
    if state == SessionState.WAITING:
        session.start_time = None
        session.start_gesture.progress = 0.65
    before = session.last_packet.pose.copy()
    snapshot = (
        session.rep_count,
        session.state,
        session.start_gesture.progress,
        len(session.recorder),
    )
    labels = []
    put_text = cv2.putText

    def checked_text(
        region, message, position, font, scale, color, thickness, line_type
    ):
        (width, height), baseline = cv2.getTextSize(message, font, scale, thickness)
        x, y = position
        assert 0 <= x and x + width <= region.shape[1]
        assert 0 <= y - height and y + baseline <= region.shape[0]
        labels.append(message)
        return put_text(
            region, message, position, font, scale, color, thickness, line_type
        )

    monkeypatch.setattr(cv2, "putText", checked_text)
    image = ui.WorkoutDisplay(ui_scale=ui_scale).render(
        session, [person], now=118, size=size, camera_width=1280, camera_height=720
    )
    assert image.shape == (size[1], size[0], 3)
    assert image.dtype == np.uint8
    assert "REPS" in labels and "TIME LEFT" in labels
    if state == SessionState.ACTIVE_LOST:
        assert "PERSON LOST - TIMER CONTINUES" in labels
    if state == SessionState.WAITING:
        assert "READY - RAISE RIGHT HAND FOR 2s" in labels
    assert not any("Analysis drops" in label for label in labels)
    assert snapshot == (
        session.rep_count,
        session.state,
        session.start_gesture.progress,
        len(session.recorder),
    )
    np.testing.assert_array_equal(session.last_packet.pose, before)


def test_laptop_digits_are_large_in_final_display_pixels(monkeypatch, pose_factory):
    heights = {}
    put_text = cv2.putText

    def measured(region, message, position, font, scale, color, thickness, line_type):
        heights[message] = cv2.getTextSize(message, font, scale, thickness)[0][1]
        return put_text(
            region, message, position, font, scale, color, thickness, line_type
        )

    monkeypatch.setattr(cv2, "putText", measured)
    person = pose_factory()
    ui.WorkoutDisplay().render(
        active_session(person),
        [person],
        now=118,
        size=(2560, 1600),
        camera_width=1280,
        camera_height=720,
    )
    assert heights["19"] >= 150 and heights["00:12"] >= 150
    assert heights["KEEP GOING"] >= 45


@pytest.mark.parametrize("source_size", [(1280, 720), (720, 1280)])
def test_camera_is_letterboxed_without_cropping_or_changing_source(
    pose_factory, source_size
):
    width, height = source_size
    color = (90, 110, 140)
    source = np.full((height, width, 3), color, dtype=np.uint8)
    person = pose_factory()
    image = ui.WorkoutDisplay().render(
        active_session(person),
        [person],
        now=118,
        size=(2560, 1600),
        camera_width=width,
        camera_height=height,
        video=source,
    )
    y, x = np.where(np.all(image == color, axis=2))
    assert y.min() == 0
    assert (x.max() - x.min() + 1) / (y.max() - y.min() + 1) == pytest.approx(
        width / height, abs=0.003
    )
    np.testing.assert_array_equal(source, np.full_like(source, color))


def test_fullscreen_restores_window_size_and_text_controls_are_bounded(monkeypatch):
    bounds = [2048, 1280]
    modes = []
    monkeypatch.setattr(ui, "work_area", lambda: (2560, 1600))
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(
        cv2,
        "resizeWindow",
        lambda name, width, height: bounds.__setitem__(slice(None), [width, height]),
    )
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *bounds))

    def set_property(name, prop, mode):
        modes.append(mode)
        if mode == cv2.WINDOW_FULLSCREEN:
            bounds[:] = [2560, 1600]

    monkeypatch.setattr(cv2, "setWindowProperty", set_property)
    display = ui.WorkoutDisplay()
    display.open("test")
    assert display.viewport_size("test") == (2048, 1280)
    display.handle_key(ord("F"), "test")
    assert display.viewport_size("test") == (2560, 1600)
    display.handle_key(ord("f"), "test")
    assert display.viewport_size("test") == (2048, 1280)
    assert modes == [cv2.WINDOW_FULLSCREEN, cv2.WINDOW_NORMAL]
    startup_fullscreen = ui.WorkoutDisplay(fullscreen=True)
    startup_fullscreen.open("test")
    assert startup_fullscreen.viewport_size("test") == (2560, 1600)
    startup_fullscreen.handle_key(ord("F"), "test")
    assert startup_fullscreen.viewport_size("test") == (2048, 1280)
    display.handle_key(ord("D"), "test")
    assert display.diagnostics
    for _ in range(30):
        display.handle_key(ord("+"), "test")
    assert display.ui_scale == ui.MAX_UI_SCALE
    for _ in range(30):
        display.handle_key(ord("-"), "test")
    assert display.ui_scale == ui.MIN_UI_SCALE


def test_text_scaling_enlarges_digits_at_the_same_window_size(
    monkeypatch, pose_factory
):
    digits = []
    put_text = cv2.putText

    def measure(region, message, position, font, scale, color, thickness, line_type):
        if message == "19":
            digits.append(cv2.getTextSize(message, font, scale, thickness)[0][1])
        return put_text(
            region, message, position, font, scale, color, thickness, line_type
        )

    monkeypatch.setattr(cv2, "putText", measure)
    person = pose_factory()
    session = active_session(person)
    for scale in (1.0, 1.5):
        image = ui.WorkoutDisplay(ui_scale=scale).render(
            session,
            [person],
            now=118,
            size=(2560, 1600),
            camera_width=1280,
            camera_height=720,
        )
        assert image.shape == (1600, 2560, 3)
    assert digits[1] > digits[0]


def test_ui_cli_defaults_and_overrides():
    defaults = parse_args([])
    assert (defaults.max_width, defaults.max_height) == (2560, 1600)
    assert (
        not defaults.fullscreen and not defaults.diagnostics and defaults.ui_scale == 1
    )
    args = parse_args(["--fullscreen", "--diagnostics", "--ui-scale", "1.5"])
    assert args.fullscreen and args.diagnostics and args.ui_scale == 1.5


@pytest.mark.parametrize("value", ["nan", "inf", "0", "0.7", "2.1"])
def test_ui_cli_rejects_invalid_scales(value):
    with pytest.raises(SystemExit) as error:
        parse_args(["--ui-scale", value])
    assert error.value.code == 2
