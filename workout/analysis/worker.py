"""A latest-value mailbox and a Windows-spawn-safe, CPU-only exercise process."""

import ctypes
import multiprocessing as mp
import pickle
import traceback

from workout.analysis.jumping_jack import JumpingJackCounter
from workout.packets import AnalysisStatus, PosePacket


class LatestMailbox:
    """One tiny shared slot. Neither producer nor consumer waits for its lock.

    Unlike Queue(maxsize=1), there is no feeder thread racing a get/replace.
    Pickle is only used between our own trusted processes, never for disk data.
    All expensive work happens outside the lock; analysis cannot hold up capture.
    """

    def __init__(self, context):
        self.data = context.RawArray(ctypes.c_ubyte, 8192)
        self.size = context.RawValue(ctypes.c_int, 0)
        self.sequence = context.RawValue(ctypes.c_int64, 0)
        self.consumed = context.RawValue(ctypes.c_int64, 0)
        self.lock = context.Lock()
        self.dropped = 0  # producer-local count (overwrites plus lock contention)
        self.published = 0

    def publish(self, value) -> bool:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self.data):
            raise ValueError("Mailbox is for small pose/status packets only")
        if not self.lock.acquire(False):
            self.dropped += 1
            return False
        try:
            if self.sequence.value != self.consumed.value:
                self.dropped += 1
            ctypes.memmove(ctypes.addressof(self.data), payload, len(payload))
            self.size.value = len(payload)
            self.sequence.value += 1
            self.published += 1
        finally:
            self.lock.release()
        return True

    def read(self):
        if not self.lock.acquire(False):
            return None
        try:
            if self.sequence.value == self.consumed.value:
                return None
            payload = ctypes.string_at(ctypes.addressof(self.data), self.size.value)
            self.consumed.value = self.sequence.value
        finally:
            self.lock.release()
        return pickle.loads(payload)


def analysis_main(
    poses: LatestMailbox,
    statuses: LatestMailbox,
    stop,
    ready,
    delay_seconds: float = 0.0,
) -> None:
    """Students can replace/extend this counter without touching GPU inference."""
    counter = JumpingJackCounter()
    processed = 0
    ready.set()
    try:
        while True:
            packet = poses.read()
            if packet is None:
                if stop.is_set():
                    break
                stop.wait(0.002)  # only the idle CPU worker waits
                continue
            if delay_seconds:
                stop.wait(delay_seconds)  # test hook, interruptible on shutdown
            counter.update(packet)
            processed += 1
            statuses.publish(
                AnalysisStatus(
                    counter.rep_count,
                    counter.state.value,
                    packet.frame_index,
                    packet.generation,
                    processed,
                )
            )
    except Exception:
        statuses.publish(
            AnalysisStatus(
                counter.rep_count,
                counter.state.value,
                counter.last_frame,
                counter.generation or 0,
                processed,
                traceback.format_exc()[-4000:],
            )
        )
        raise


class AnalysisWorker:
    def __init__(self, delay_seconds: float = 0.0):
        context = mp.get_context("spawn")
        self.poses = LatestMailbox(context)
        self.statuses = LatestMailbox(context)
        self.stop = context.Event()
        self.ready = context.Event()
        self.process = context.Process(
            target=analysis_main,
            args=(self.poses, self.statuses, self.stop, self.ready, delay_seconds),
            name="workout-analysis",
            daemon=True,
        )
        self.last_status: AnalysisStatus | None = None
        self.forced_shutdown = False

    def start(self) -> None:
        self.process.start()
        # Startup only, before opening the camera/live loop.
        if not self.ready.wait(15):
            self.close()
            raise RuntimeError(
                "Workout analysis process did not start within 15 seconds"
            )

    def publish(self, packet: PosePacket) -> bool:
        return self.poses.publish(packet)

    def poll(self) -> AnalysisStatus | None:
        status = self.statuses.read()
        if status is not None:
            self.last_status = status
        return status

    def close(self) -> AnalysisStatus | None:
        """Bounded drain/join, called only AFTER leaving the live loop."""
        if self.process.pid is None:
            return self.last_status
        self.stop.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.forced_shutdown = True
            self.process.terminate()
            self.process.join(timeout=2)
        self.poll()
        return self.last_status
