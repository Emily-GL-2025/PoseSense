import multiprocessing as mp
import time

from workout.analysis.worker import AnalysisWorker, LatestMailbox
from workout.packets import PosePacket
from workout.recorder import SessionRecorder


def test_latest_mailbox_overwrites_and_never_waits_for_lock():
    mailbox = LatestMailbox(mp.get_context("spawn"))
    assert mailbox.publish("old")
    assert mailbox.publish("new")
    assert mailbox.dropped == 1
    assert mailbox.read() == "new"
    assert mailbox.read() is None
    mailbox.lock.acquire()
    try:
        started = time.perf_counter()
        assert mailbox.publish("contended") is False
        assert mailbox.read() is None
        assert time.perf_counter() - started < 0.1
    finally:
        mailbox.lock.release()


def test_spawned_slow_worker_drops_packets_but_recorder_is_complete(
    pose_factory, tmp_path
):
    worker = AnalysisWorker(delay_seconds=0.15)
    recorder = SessionRecorder()
    worker.start()
    try:
        person = pose_factory()
        started = time.perf_counter()
        for frame in range(1000):
            packet = PosePacket(
                frame, frame / 30, 7, person.pose, frame % 7 != 0, frame // 7
            )
            recorder.append(packet, 0)
            worker.publish(packet)
            worker.poll()
        elapsed = time.perf_counter() - started
        assert elapsed < 2.0  # serial analysis would require 150 seconds
        assert len(recorder) == 1000
        assert worker.poses.dropped > 900
        # Wait only in this test, not the application's producer path.
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            status = worker.poll()
            if status is not None and status.last_processed_frame == 999:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Worker never consumed the newest packet")
        assert status.processed_packets < 20 and not status.error
    finally:
        worker.close()
    assert not worker.process.is_alive() and not worker.forced_shutdown


def test_shutdown_drains_last_packet(pose_factory):
    worker = AnalysisWorker(delay_seconds=0.3)
    worker.start()
    person = pose_factory()
    worker.publish(PosePacket(123, 0, 7, person.pose, True, 1))
    status = worker.close()
    assert status.last_processed_frame == 123
    assert not worker.process.is_alive()
