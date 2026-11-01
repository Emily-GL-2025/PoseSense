import pytest

from workout.analysis.jumping_jack import ExerciseState, JumpingJackCounter
from workout.packets import PosePacket


def feed(counter, person, start, generation=1, present=True, accepted_reps=0):
    for time_value in (start, start + 0.04, start + 0.08):
        packet = PosePacket(
            counter.last_frame + 1,
            time_value,
            person.track_id,
            person.pose,
            present,
            generation,
            accepted_reps,
        )
        counter.update(packet)
    return counter.rep_count


@pytest.mark.parametrize(
    ("posture", "ankle_width", "expected"),
    [
        ("closed", 0.3125, ExerciseState.CLOSED),  # ratio 1.0: inclusive
        ("closed", 0.34375, None),  # ratio 1.1: too wide for CLOSED
        ("open", 0.40625, None),  # ratio 1.3: too narrow for OPEN
        ("open", 0.4375, ExerciseState.OPEN),  # ratio 1.4: inclusive
        ("open", 0.46875, ExerciseState.OPEN),  # ratio 1.5: wide enough
    ],
)
def test_default_feet_thresholds(pose_factory, posture, ankle_width, expected):
    person = pose_factory(posture)
    # Exact binary coordinates keep float32 rounding away from the foot bounds.
    person.pose[5, 0], person.pose[6, 0] = 0.34375, 0.65625
    person.pose[15, 0] = 0.5 - ankle_width / 2
    person.pose[16, 0] = 0.5 + ankle_width / 2
    assert JumpingJackCounter().classify(person.pose) == expected


def test_required_sequence_and_start_gesture(pose_factory):
    counter = JumpingJackCounter()
    assert feed(counter, pose_factory("raised"), 0) == 0
    assert feed(counter, pose_factory("open"), 0.1) == 0
    assert counter.state == ExerciseState.UNKNOWN
    assert feed(counter, pose_factory("closed"), 0.2) == 0
    assert counter.state == ExerciseState.CLOSED
    assert feed(counter, pose_factory("open"), 0.3) == 0
    assert counter.state == ExerciseState.OPEN
    assert feed(counter, pose_factory("closed"), 0.4) == 1
    assert feed(counter, pose_factory("closed"), 0.5) == 1


def test_jitter_and_brief_opposite_posture_do_not_count(pose_factory):
    counter = JumpingJackCounter()
    feed(counter, pose_factory(), 0)
    for i in range(5):
        feed(counter, pose_factory("jitter"), 0.1 + i * 0.1)
    assert counter.state == ExerciseState.CLOSED and counter.rep_count == 0
    counter.update(
        PosePacket(counter.last_frame + 1, 0.6, 7, pose_factory("open").pose, True, 1)
    )
    feed(counter, pose_factory(), 0.64)
    assert counter.rep_count == 0


@pytest.mark.parametrize("lost_packet_delivered", [True, False])
def test_loss_resets_even_when_loss_packet_is_overwritten(
    pose_factory, lost_packet_delivered
):
    counter = JumpingJackCounter()
    feed(counter, pose_factory(), 0)
    feed(counter, pose_factory("open"), 0.1)
    if lost_packet_delivered:
        feed(counter, pose_factory(), 0.2, generation=2, present=False)
        assert counter.state == ExerciseState.UNKNOWN
    # Main has already incremented continuity on loss AND return.
    assert feed(counter, pose_factory("open"), 0.3, generation=3) == 0
    assert counter.state == ExerciseState.UNKNOWN
    assert feed(counter, pose_factory(), 0.4, generation=3) == 0
    assert feed(counter, pose_factory("open"), 0.5, generation=3) == 0
    assert feed(counter, pose_factory(), 0.6, generation=3) == 1


def test_low_confidence_and_long_analysis_gap_reset(pose_factory):
    counter = JumpingJackCounter()
    feed(counter, pose_factory(), 0)
    feed(counter, pose_factory("open"), 0.1)
    noisy = pose_factory("closed")
    noisy.pose[15, 2] = 0.01
    feed(counter, noisy, 0.2)
    assert counter.state == ExerciseState.UNKNOWN
    assert feed(counter, pose_factory(), 0.3) == 0
    feed(counter, pose_factory("open"), 0.4)
    assert feed(counter, pose_factory(), 2.0) == 0


def test_degenerate_scale_and_duplicate_packet(pose_factory):
    counter = JumpingJackCounter()
    person = pose_factory()
    person.pose[6, 0] = person.pose[5, 0]
    feed(counter, person, 0)
    assert counter.state == ExerciseState.UNKNOWN
    prior = counter.last_frame
    counter.update(PosePacket(prior, 0.5, 7, pose_factory().pose, True, 1))
    assert counter.last_timestamp == 0.08
