"""Distance-readable OpenCV workout display; never updates session or analysis state."""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from workout.live_display import SKELETON, display_person, project
from workout.packets import KEYPOINT_CONFIDENCE
from workout.session import SessionState

BG = (22, 19, 16)
PANEL = (34, 29, 25)
WHITE = (247, 245, 240)
MUTED = (187, 180, 170)
GREEN = (144, 232, 150)
AMBER = (83, 202, 255)
RED = (111, 124, 255)
TRACK = (70, 61, 52)
MIN_UI_SCALE, MAX_UI_SCALE = 0.75, 2.0


def work_area() -> tuple[int, int]:
    """Use physical display pixels on Windows, including high-DPI laptop screens."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            user32.SetProcessDPIAware()
        rect = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
    return 2560, 1600


def remaining_label(session, now: float) -> str:
    # Ceiling avoids showing 00:00 while an active workout still has time left.
    seconds = math.ceil(max(0.0, session.duration_seconds - session.elapsed(now)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_text(image, message, box, height, color=WHITE, *, bold=False):
    """Fit text inside a bounded region, in final display pixels."""
    x, y, width, box_height = map(int, box)
    if not message or width < 4 or box_height < 4:
        return
    region = image[y : y + box_height, x : x + width]
    if not region.size:
        return
    thickness = max(1, round(height / (22 if bold else 30)))
    if height >= 24:
        thickness = max(2, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = cv2.getTextSize(message, font, 1, thickness)[0][1]
    scale = max(0.01, height / base)
    for _ in range(3):
        (text_width, text_height), baseline = cv2.getTextSize(
            message, font, scale, thickness
        )
        fit = min(
            1.0,
            (region.shape[1] - 4) / max(1, text_width),
            (region.shape[0] - 4) / max(1, text_height + baseline),
        )
        if fit >= 1:
            break
        scale *= max(0.01, fit)
    (text_width, text_height), baseline = cv2.getTextSize(
        message, font, scale, thickness
    )
    position = (
        max(0, (region.shape[1] - text_width) // 2),
        max(text_height, (region.shape[0] - text_height - baseline) // 2 + text_height),
    )
    cv2.putText(region, message, position, font, scale, color, thickness, cv2.LINE_AA)


class WorkoutDisplay:
    def __init__(
        self,
        *,
        max_width=2560,
        max_height=1600,
        ui_scale=1.0,
        fullscreen=False,
        diagnostics=False,
    ):
        self.max_width, self.max_height = max_width, max_height
        self.ui_scale = ui_scale
        self.fullscreen = fullscreen
        self.diagnostics = diagnostics
        self.size = (min(2048, max_width), min(1280, max_height))
        self.windowed_size = self.size
        self._canvas = None
        self._saved_count = None
        self._saved_until = 0.0

    def open(self, name):
        width, height = work_area()
        self.size = (
            min(self.max_width, round(width * 0.8)),
            min(self.max_height, round(height * 0.8)),
        )
        self.windowed_size = self.size
        # The canvas is rebuilt at the client size, with camera letterboxing.
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(name, *self.size)
        if self.fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def viewport_size(self, name):
        try:
            _, _, width, height = cv2.getWindowImageRect(name)
        except cv2.error:
            return self.size
        if width >= 320 and height >= 240:
            scale = min(1.0, self.max_width / width, self.max_height / height)
            self.size = (round(width * scale), round(height * scale))
        return self.size

    def handle_key(self, key, name):
        if key in (ord("d"), ord("D")):
            self.diagnostics = not self.diagnostics
        elif key in (ord("+"), ord("="), ord("-"), ord("_")):
            step = 0.1 if key in (ord("+"), ord("=")) else -0.1
            self.ui_scale = round(
                min(MAX_UI_SCALE, max(MIN_UI_SCALE, self.ui_scale + step)), 2
            )
        elif key in (ord("f"), ord("F")):
            if not self.fullscreen:
                self.windowed_size = self.viewport_size(name)
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )
            if not self.fullscreen:
                cv2.resizeWindow(name, *self.windowed_size)
                self.size = self.windowed_size

    def notify_saved(self, rep_count, now):
        self._saved_count = rep_count
        self._saved_until = now + 4.0

    def render(
        self,
        session,
        people,
        *,
        now,
        size=None,
        camera_width,
        camera_height,
        video=None,
        diagnostic_image=None,
        analysis_note="",
        dropped=0,
    ):
        width, height = size or self.size
        shape = (height, width, 3)
        if self._canvas is None or self._canvas.shape != shape:
            self._canvas = np.empty(shape, np.uint8)
        image = self._canvas
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), BG, -1)
        unit = min(width / 1280, height / 800)
        scale = unit * self.ui_scale
        band_height = min(round(height * 0.56), round(240 * scale))
        top = height - band_height
        content = image[:top]
        source = diagnostic_image if diagnostic_image is not None else video
        if source is not None:
            self._fit_image(content, source)
        else:
            self._draw_skeleton(
                content, session, people, camera_width, camera_height, unit
            )
        cv2.rectangle(image, (0, top), (width - 1, height - 1), PANEL, -1)
        self._draw_band(image, session, now, top, scale, unit, analysis_note)
        if self.diagnostics and diagnostic_image is None:
            draw_text(
                content,
                f"ID {session.selected_track_id} | Phase {session.exercise_state} | Analysis drops {dropped}",
                (0, max(0, top - round(32 * unit)), width, round(30 * unit)),
                17 * unit,
                AMBER,
            )
        return image

    @staticmethod
    def _fit_image(region, source):
        if not region.size:
            return
        height, width = region.shape[:2]
        scale = min(width / source.shape[1], height / source.shape[0])
        resized_width = max(1, round(source.shape[1] * scale))
        resized_height = max(1, round(source.shape[0] * scale))
        x, y = (width - resized_width) // 2, (height - resized_height) // 2
        region[y : y + resized_height, x : x + resized_width] = cv2.resize(
            source,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA,
        )

    @staticmethod
    def _draw_skeleton(image, session, people, camera_width, camera_height, unit):
        if not image.size:
            return
        pose, _, preview = display_person(session, people)
        height, width = image.shape[:2]
        if pose is None:
            draw_text(
                image,
                "STEP INTO CAMERA VIEW",
                (0, height // 3, width, max(1, height // 3)),
                35 * unit,
                MUTED,
            )
            return
        valid = np.isfinite(pose).all(axis=1) & (pose[:, 2] >= KEYPOINT_CONFIDENCE)
        points, valid, _, _ = project(
            pose, valid, camera_width, camera_height, (width, height)
        )
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                cv2.line(
                    image,
                    tuple(points[a]),
                    tuple(points[b]),
                    GREEN,
                    max(2, round(4 * unit)),
                    cv2.LINE_AA,
                )
        for joint in np.flatnonzero(valid):
            cv2.circle(
                image,
                tuple(points[joint]),
                max(3, round(6 * unit)),
                WHITE,
                -1,
                cv2.LINE_AA,
            )
        if preview:
            draw_text(
                image,
                "PREVIEW - NOT SELECTED",
                (0, 0, width, round(40 * unit)),
                20 * unit,
                MUTED,
            )

    def _draw_band(self, image, session, now, top, scale, unit, analysis_note):
        height, width = image.shape[:2]
        band = height - top
        waiting = session.state == SessionState.WAITING
        lost = session.state == SessionState.ACTIVE_LOST
        color = AMBER if waiting else RED if lost else GREEN
        label_top = top + round(band * 0.05)
        for column, label, value in (
            (0, "REPS", str(session.rep_count)),
            (1, "TIME LEFT", remaining_label(session, now)),
        ):
            left = column * (width // 2)
            draw_text(
                image,
                label,
                (left, label_top, width // 2, round(band * 0.12)),
                22 * scale,
                MUTED,
            )
            draw_text(
                image,
                value,
                (left, top + round(band * 0.18), width // 2, round(band * 0.39)),
                88 * scale,
                AMBER
                if column == 1
                and 0 < session.duration_seconds - session.elapsed(now) <= 5
                else WHITE,
                bold=True,
            )
        gesture = session.start_gesture if waiting else session.reacquire_gesture
        detail = ""
        if waiting:
            message = "READY - RAISE RIGHT HAND FOR 2s"
            if self._saved_count is not None and now < self._saved_until:
                message = f"SAVED {self._saved_count} REPS - READY FOR NEXT WORKOUT"
                detail = "RAISE RIGHT HAND FOR 2s TO START"
            elif gesture.progress > 0:
                detail = f"KEEP HOLDING - {gesture.progress * gesture.hold_seconds:.1f} / {gesture.hold_seconds:g}s"
        elif lost:
            message = "PERSON LOST - TIMER CONTINUES"
            detail = "RAISE RIGHT HAND FOR 1s TO RESELECT"
        elif session.state == SessionState.FINISHED:
            message = "WORKOUT COMPLETE"
        elif analysis_note.startswith(("ERROR", "OFFLINE")):
            message = "COUNTING UNAVAILABLE - RECORDING CONTINUES"
            color = RED
        elif session.exercise_state == "UNKNOWN":
            message = "LOWER HANDS AND BRING FEET TOGETHER"
        else:
            message = "KEEP GOING"
        draw_text(
            image,
            message,
            (0, top + round(band * 0.59), width, round(band * 0.16)),
            30 * scale,
            color,
            bold=True,
        )
        draw_text(
            image,
            detail,
            (0, top + round(band * 0.76), width, round(band * 0.10)),
            19 * scale,
            WHITE,
        )
        progress = (
            gesture.progress
            if waiting or lost
            else max(0.0, session.duration_seconds - session.elapsed(now))
            / max(0.001, session.duration_seconds)
        )
        margin = max(1, round(24 * unit))
        y = top + round(band * 0.88)
        bar_height = max(2, round(7 * unit))
        cv2.rectangle(image, (margin, y), (width - margin, y + bar_height), TRACK, -1)
        end = margin + round((width - 2 * margin) * min(1.0, max(0.0, progress)))
        if end > margin:
            cv2.rectangle(image, (margin, y), (end, y + bar_height), color, -1)
        draw_text(
            image,
            "V View   D Details   F Fullscreen   +/- Text   Q/Esc Exit",
            (0, top + round(band * 0.93), width, max(1, round(band * 0.07))),
            14 * unit,
            MUTED,
        )
