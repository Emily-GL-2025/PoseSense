"""Main-process selection, timing, loss handling, and one-record-per-frame rule."""

import time
from enum import Enum

from workout.gesture import REACQUIRE_HOLD_SECONDS, GestureDetector
from workout.packets import AnalysisStatus, PosePacket, TrackedPerson, clean_pose
from workout.recorder import SessionRecorder

SESSION_SECONDS = 30.0


class SessionState(str, Enum):
    WAITING = "WAITING"
    ACTIVE_TRACKED = "ACTIVE_TRACKED"
    ACTIVE_LOST = "ACTIVE_LOST"
    FINISHED = "FINISHED"


class SessionController:
    def __init__(
        self,
        recorder: SessionRecorder | None = None,
        *,
        duration_seconds: float = SESSION_SECONDS,
    ):
        self.recorder = recorder if recorder is not None else SessionRecorder()
        self.duration_seconds = duration_seconds
        self.state = SessionState.WAITING
        self.start_gesture = GestureDetector()
        self.reacquire_gesture = GestureDetector(REACQUIRE_HOLD_SECONDS)
        self.selected_track_id: int | None = None
        self.initial_track_id: int | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.generation = 0
        self.rep_count = 0
        self.exercise_state = "UNKNOWN"
        self.last_status_frame = -1
        self.last_packet: PosePacket | None = None
        self.events: list[dict] = []
        self.stop_reason = ""
        self._last_now: float | None = None

    @property
    def active(self) -> bool:
        return self.state in (SessionState.ACTIVE_TRACKED, SessionState.ACTIVE_LOST)

    def elapsed(self, now: float | None = None) -> float:
        if self.start_time is None:
            return 0.0
        end = (
            self.end_time
            if self.end_time is not None
            else (time.perf_counter() if now is None else now)
        )
        return min(self.duration_seconds, max(0.0, end - self.start_time))

    def advance_time(self, now: float) -> None:
        if self._last_now is not None and now < self._last_now:
            raise ValueError("Session clock must be monotonic")
        self._last_now = now
        if self.active and now >= self.start_time + self.duration_seconds:
            self.finish(now, "completed")

    def finish(self, now: float, reason: str) -> None:
        if not self.active:
            return
        self.end_time = min(now, self.start_time + self.duration_seconds)
        self.stop_reason = (
            "completed"
            if self.end_time >= self.start_time + self.duration_seconds
            else reason
        )
        self.state = SessionState.FINISHED

    def _event(self, name: str, now: float, **details) -> None:
        self.events.append({"event": name, "timestamp": self.elapsed(now), **details})

    def _reset_continuity(self) -> None:
        self.generation += 1
        self.exercise_state = "UNKNOWN"

    def accept_status(
        self, status: AnalysisStatus | None, *, final: bool = False
    ) -> None:
        tracked = self.state == SessionState.ACTIVE_TRACKED
        if final:
            tracked = self.last_packet is not None and self.last_packet.present
        if (
            status is None
            or status.error
            or not tracked
            or self.last_packet is None
            or status.generation != self.generation
            or status.last_processed_frame <= self.last_status_frame
            or status.last_processed_frame > self.last_packet.frame_index
        ):
            return
        self.rep_count = max(self.rep_count, status.rep_count)
        self.exercise_state = status.exercise_state
        self.last_status_frame = status.last_processed_frame

    def process_frame(
        self,
        people: list[TrackedPerson],
        frame_index: int,
        now: float | None = None,
        status: AnalysisStatus | None = None,
    ) -> PosePacket | None:
        now = time.perf_counter() if now is None else now
        self.advance_time(now)
        if self.state == SessionState.FINISHED:
            return None
        by_id = {person.track_id: person for person in people}
        if self.state == SessionState.WAITING:
            selected = self.start_gesture.update(people, now)
            if selected is None:
                return None
            self.selected_track_id = self.initial_track_id = selected
            self.start_time = now
            self.state = SessionState.ACTIVE_TRACKED
            self._reset_continuity()
            self._event("started", now, track_id=selected)
            self.start_gesture.reset()

        person = by_id.get(self.selected_track_id)
        if person is None:
            if self.state != SessionState.ACTIVE_LOST:
                self.state = SessionState.ACTIVE_LOST
                self._reset_continuity()
                self.reacquire_gesture.reset()
                self._event("lost", now, track_id=self.selected_track_id)
            replacement = self.reacquire_gesture.update(people, now)
            if replacement is not None:
                previous = self.selected_track_id
                self.selected_track_id = replacement
                person = by_id[replacement]
                self._event(
                    "reassigned_by_gesture",
                    now,
                    old_track_id=previous,
                    new_track_id=replacement,
                )
        if person is not None and self.state == SessionState.ACTIVE_LOST:
            self.state = SessionState.ACTIVE_TRACKED
            self._reset_continuity()
            self.reacquire_gesture.reset()
            self._event("reacquired", now, track_id=self.selected_track_id)

        # Apply status only after loss/recovery transitions, to reject stale work.
        self.accept_status(status)
        packet = PosePacket(
            frame_index,
            self.elapsed(now),
            self.selected_track_id,
            clean_pose(person.pose if person is not None else None),
            person is not None,
            self.generation,
            self.rep_count,
        )
        self.last_packet = packet
        self.recorder.append(packet, self.rep_count)
        return packet

    def metadata(self, **source_info) -> dict:
        return {
            **source_info,
            "selected_track_id_at_start": self.initial_track_id,
            "selected_track_id_at_end": self.selected_track_id,
            "session_duration": self.elapsed(),
            "target_duration": self.duration_seconds,
            "completed": self.stop_reason == "completed",
            "stop_reason": self.stop_reason,
            "final_rep_count": self.rep_count,
            "last_analysis_frame": self.last_status_frame,
            "events": self.events,
        }
