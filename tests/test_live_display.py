"""Live UI is read-only; its spatial checks agree with the actual analyzer."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from workout.analysis.jumping_jack import JumpingJackConfig, JumpingJackCounter
from workout.live_display import (
    HEIGHT,
    WIDTH,
    LiveSkeletonView,
    display_person,
    inspect_pose,
    project,
    target_posture,
)
from workout.packets import PosePacket, clean_pose
from workout.session import SessionController, SessionState
from workout_session import parse_args


def active_session(person):
    session = SessionController()
    session.state = SessionState.ACTIVE_TRACKED
    session.start_time = 10.0
    session.selected_track_id = person.track_id
    session.rep_count = 3
    session.generation = 1
    session.exercise_state = "CLOSED"
    session.last_packet = PosePacket(
        12, 2.0, person.track_id, clean_pose(person.pose), True, 1, 3
    )
    return session


@pytest.mark.parametrize(
    "name,expected", [("closed", "CLOSED"), ("open", "OPEN"), ("jitter", "TRANSITION")]
)
def test_spatial_checks_match_current_classifier(pose_factory, name, expected):
    cfg = JumpingJackConfig()
    pose = pose_factory(name).pose
    checks = inspect_pose(pose, cfg)
    assert checks.posture == expected
    actual = JumpingJackCounter(cfg).classify(pose)
    assert checks.posture == (actual.value if actual else "TRANSITION")
    assert all(checks.closed) is (expected == "CLOSED")
    assert all(checks.opened) is (expected == "OPEN")


def test_strict_wrist_and_inclusive_ankle_scale_confidence_boundaries(pose_factory):
    pose = pose_factory().pose
    pose[:, 2] = 0.5
    pose[5, :2], pose[6, :2] = (0.375, 0.25), (0.625, 0.25)
    pose[15, :2], pose[16, :2] = (0.25, 0.75), (0.75, 0.75)
    pose[9, 1] = pose[10, 1] = 0.125
    cfg = replace(
        JumpingJackConfig(),
        wrist_margin=0.25,
        open_ankle_ratio=2.0,
        minimum_shoulder_width=0.25,
    )
    assert inspect_pose(pose, cfg).posture == "TRANSITION"
    pose[9, 1] = pose[10, 1] = 0
    assert inspect_pose(pose, cfg).posture == "OPEN"
    pose[10, 2] = 0.499
    assert inspect_pose(pose, cfg).posture == "INVALID"
    assert inspect_pose(None, cfg).posture == "INVALID"


def test_randomized_checks_use_the_same_gates(pose_factory):
    rng = np.random.default_rng(8)
    cfg = JumpingJackConfig()
    for _ in range(300):
        pose = pose_factory().pose
        pose[:, :2] = rng.uniform(-0.1, 1.1, (17, 2)).astype(np.float32)
        if rng.random() < 0.2:
            pose[10, 0] = np.nan
        checks = inspect_pose(pose, cfg)
        classified = JumpingJackCounter(cfg).classify(pose)
        if checks.invalid:
            assert classified is None
        else:
            expected = (
                "CLOSED"
                if all(checks.closed)
                else "OPEN"
                if all(checks.opened)
                else "TRANSITION"
            )
            assert checks.posture == expected
            assert checks.posture == (classified.value if classified else "TRANSITION")


def test_selected_pose_never_substitutes_a_bystander(pose_factory):
    selected, other = pose_factory(track_id=7), pose_factory("open", track_id=9)
    session = active_session(selected)
    pose, caption, preview = display_person(session, [other])
    assert pose is session.last_packet.pose and "7" in caption and not preview
    session.state = SessionState.ACTIVE_LOST
    session.last_packet = replace(
        session.last_packet, pose=clean_pose(None), present=False, generation=2
    )
    assert display_person(session, [other])[0] is None
    session.reacquire_gesture.candidate_id = 9
    pose, caption, preview = display_person(session, [other])
    assert preview and "NOT selected" in caption
    assert session.selected_track_id == 7 and session.rep_count == 3


def test_waiting_previews_gesture_candidate_including_zero_id(pose_factory):
    session = SessionController()
    first, second = pose_factory(track_id=0), pose_factory(track_id=9)
    assert "0" in display_person(session, [second, first])[1]
    session.start_gesture.candidate_id = 9
    assert "9" in display_person(session, [first, second])[1]
    # During the gesture's missing grace period, do not paint a different
    # person's pose underneath the old candidate's hold progress.
    assert display_person(session, [first])[0] is None
    assert session.selected_track_id is None


def test_finished_preview_uses_current_selected_pose_without_changing_recording(
    pose_factory,
):
    session = active_session(pose_factory("closed"))
    session.state = SessionState.FINISHED
    session.end_time = 12.0
    recorded_pose = session.last_packet.pose.copy()
    current = pose_factory("open")
    pose, caption, preview = display_person(session, [current])
    np.testing.assert_array_equal(pose, clean_pose(current.pose))
    assert preview and "Live preview ID 7" == caption
    assert not np.array_equal(pose, recorded_pose, equal_nan=True)
    assert display_person(session, [pose_factory("open", track_id=9)])[0] is None
    assert display_person(session, [])[0] is None
    np.testing.assert_array_equal(session.last_packet.pose, recorded_pose)
    assert session.rep_count == 3 and session.selected_track_id == 7


@pytest.mark.parametrize(
    "phase,target", [("UNKNOWN", "CLOSED"), ("CLOSED", "OPEN"), ("OPEN", "CLOSED")]
)
def test_target_comes_from_accepted_worker_phase(pose_factory, phase, target):
    session = active_session(pose_factory("open"))
    session.exercise_state = phase
    assert target_posture(session) == target
    session.state = SessionState.ACTIVE_LOST
    assert target_posture(session) is None


@pytest.mark.parametrize("stage", list(SessionState))
def test_render_all_session_states_without_mutating_input(pose_factory, stage):
    person = pose_factory("open")
    session = active_session(person)
    session.state = stage
    if stage == SessionState.FINISHED:
        session.end_time = 12.0
    if stage == SessionState.ACTIVE_LOST:
        session.last_packet = replace(
            session.last_packet, pose=clean_pose(None), present=False
        )
    before = session.last_packet.pose.copy()
    pose_before = person.pose.copy()
    snapshot = (
        session.rep_count,
        session.generation,
        session.exercise_state,
        session.selected_track_id,
        len(session.recorder),
    )
    view = LiveSkeletonView()
    frame = view.render(
        session,
        [person],
        camera_width=1280,
        camera_height=720,
        now=12.0,
        frame_index=12,
        saved=Path("session.npz"),
    )
    assert frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8
    assert np.array_equal(before, session.last_packet.pose, equal_nan=True)
    assert np.array_equal(pose_before, person.pose)
    assert snapshot == (
        session.rep_count,
        session.generation,
        session.exercise_state,
        session.selected_track_id,
        len(session.recorder),
    )


def test_invalid_offscreen_and_portrait_projection(pose_factory):
    person = pose_factory()
    person.pose[10, 2] = 0.1
    person.pose[15, 0] = -2
    session = active_session(person)
    view = LiveSkeletonView()
    frame = view.render(
        session,
        [person],
        camera_width=720,
        camera_height=1280,
        now=12.0,
        frame_index=12,
    )
    assert frame.shape == (HEIGHT, WIDTH, 3)
    points, valid, scale, origin = project(
        person.pose, np.ones(17, bool), 1280, 720, (640, 360)
    )
    np.testing.assert_allclose(scale, [640, 360])
    np.testing.assert_allclose(origin, [0, 0])
    assert valid.all() and points[15, 0] < 0  # drawing clip, not an added validity rule


def test_view_cli_and_spawn_import_stays_light():
    assert parse_args([]).view == "video"
    assert parse_args(["--view", "skeleton"]).view == "skeleton"
    with pytest.raises(SystemExit):
        parse_args(["--view", "bogus"])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,workout_session; assert not any(m in sys.modules for m in ('torch','ultralytics','cv2','workout.live_display'))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
