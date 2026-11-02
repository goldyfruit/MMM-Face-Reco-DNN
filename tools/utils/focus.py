# SPDX-License-Identifier: Apache-2.0
"""Face-directed autofocus geometry and bounded scans without image ownership."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import time
from typing import Any, Callable

Rectangle = tuple[int, int, int, int]


def rectangle(value: object) -> Rectangle | None:
    """Validate bounded x/y/width/height sensor geometry without retaining input."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    if any(
        isinstance(v, bool)
        or not isinstance(v, Real)
        or not math.isfinite(float(v))
        or v < 0
        or v >= 2**31
        or int(v) != v
        for v in value
    ):
        return None
    x, y, width, height = (int(v) for v in value)
    return (x, y, width, height) if width and height else None


def supports(controls: object, name: str, value: int | float) -> bool:
    """Check a control's advertisement and numeric range when supplied."""
    if not isinstance(controls, Mapping) or name not in controls:
        return False
    bounds = controls[name]
    if isinstance(bounds, (tuple, list)) and len(bounds) >= 2:
        low, high = bounds[:2]
        if isinstance(low, Real) and isinstance(high, Real):
            return bool(low <= value <= high)
    return True


def face_window(
    boxes: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    crop: Rectangle | None,
    maximum: Rectangle | None,
    rotation: int,
) -> Rectangle | None:
    """Map the largest detected face from rotated pixels to sensor coordinates.

    Boxes use dlib's top/right/bottom/left order; image_size is height/width.
    AfWindows uses the sensor coordinate system of ScalerCropMaximum, so undo
    software rotation before applying this request's actual ScalerCrop offset
    and extent. Inset the face by 15% per edge to avoid adjacent background.
    Ties prefer the central face, then coordinate order, independently of names.
    """
    height, width = image_size
    if height <= 0 or width <= 0 or crop is None or maximum is None:
        return None
    cx, cy, cw, ch = crop
    mx, my, mw, mh = maximum
    if cx < mx or cy < my or cx + cw > mx + mw or cy + ch > my + mh:
        return None
    candidates: list[tuple[float, float, tuple[float, ...]]] = []
    for box in boxes:
        if len(box) != 4 or any(
            isinstance(v, bool)
            or not isinstance(v, Real)
            or not math.isfinite(float(v))
            for v in box
        ):
            continue
        top, right, bottom, left = (float(v) for v in box)
        top, bottom = max(0.0, top), min(float(height), bottom)
        left, right = max(0.0, left), min(float(width), right)
        if right - left < 12 or bottom - top < 12:
            continue
        area = (right - left) * (bottom - top)
        distance = ((left + right) / 2 / width - 0.5) ** 2 + (
            (top + bottom) / 2 / height - 0.5
        ) ** 2
        candidates.append((-area, distance, (top, right, bottom, left)))
    if not candidates:
        return None
    top, right, bottom, left = min(candidates)[2]
    dx, dy = (right - left) * 0.15, (bottom - top) * 0.15
    x0, x1 = (left + dx) / width, (right - dx) / width
    y0, y1 = (top + dy) / height, (bottom - dy) / height
    if rotation == 0:  # Inverse of cv2.ROTATE_90_CLOCKWISE.
        x0, x1, y0, y1 = y0, y1, 1 - x1, 1 - x0
    elif rotation == 1:  # Inverse of cv2.ROTATE_180.
        x0, x1, y0, y1 = 1 - x1, 1 - x0, 1 - y1, 1 - y0
    elif rotation == 2:  # Inverse of cv2.ROTATE_90_COUNTERCLOCKWISE.
        x0, x1, y0, y1 = 1 - y1, 1 - y0, x0, x1
    elif rotation != -1:
        return None
    left = max(cx, math.floor(cx + x0 * cw))
    top = max(cy, math.floor(cy + y0 * ch))
    right = min(cx + cw, math.ceil(cx + x1 * cw))
    bottom = min(cy + ch, math.ceil(cy + y1 * ch))
    return (
        (left, top, right - left, bottom - top)
        if right > left and bottom > top
        else None
    )


def _overlap(first: Rectangle, second: Rectangle) -> float:
    """Measure ROI overlap to ignore small box jitter between detector frames."""
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
        0, min(y1 + h1, y2 + h2) - max(y1, y2)
    )
    return intersection / (w1 * h1 + w2 * h2 - intersection)


class FaceFocusController:
    """Run single face-window AF scans, hold successful focus and bound failures."""

    def __init__(
        self,
        controls: object,
        maximum: object,
        set_controls: Callable[[dict[str, Any]], None] | None,
        seed: float | None,
        rotation: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Require the complete ROI/trigger contract; never fall back to global AF."""
        self._controls = controls
        self._maximum = rectangle(maximum)
        self._set_controls = set_controls
        self._seed = seed
        self._good_lens = seed
        self._rotation = rotation
        self._clock = clock or time.monotonic
        self._supported = (
            self._maximum is not None
            and callable(set_controls)
            and seed is not None
            and supports(controls, "LensPosition", seed)
            and all(
                supports(controls, name, value)
                for name, value in (
                    ("AfMode", 0),
                    ("AfMode", 1),
                    ("AfMetering", 1),
                    ("AfTrigger", 0),
                    ("AfTrigger", 1),
                )
            )
            and isinstance(controls, Mapping)
            and "AfWindows" in controls
        )
        self._state = "waiting" if self._supported else "unsupported"
        self._window: Rectangle | None = None
        self._crop: Rectangle | None = None
        self._started_at = 0.0
        self._last_scan = -math.inf
        self._last_success = -math.inf
        self._has_faces = False
        self._seen_scanning = False
        self._failure_until = -math.inf
        self._restored_state = "waiting"

    @property
    def status(self) -> dict[str, Any]:
        """Return bounded geometry/state only; disappearance always permits logout."""
        return {
            "supported": self._supported,
            "state": self._state,
            "settled": not self._has_faces
            or self._state not in ("scanning", "restoring", "restore_failed"),
            "window": self._window,
            "lens_position": self._good_lens,
        }

    def _restore(self, now: float, failed: bool = True) -> None:
        """Cancel only this bounded scan and restore the last confirmed lens."""
        assert self._set_controls is not None
        self._set_controls(
            {"AfTrigger": 1, "AfMode": 0, "LensPosition": self._good_lens}
        )
        self._state = "restoring"
        self._started_at = now
        self._restored_state = "fallback" if failed else "waiting"
        if failed:
            self._failure_until = now + 30.0

    def pause(self) -> None:
        """Cancel in-flight focusing on pause while preserving the last good seed."""
        if self._supported and self._state == "scanning":
            self._restore(self._clock(), failed=False)
        self._has_faces = False
        self._window = None
        self._crop = None
        self._state = "waiting" if self._supported else "unsupported"

    def resume(self) -> None:
        """Reassert the safe held lens after a camera restart, without scanning."""
        if self._supported:
            assert self._set_controls is not None
            self._set_controls({"AfMode": 0, "LensPosition": self._good_lens})
        self._has_faces = False
        self._window = None
        self._crop = None
        self._state = "restoring" if self._supported else "unsupported"
        self._started_at = self._clock()
        self._restored_state = "waiting"

    def update(
        self,
        boxes: Sequence[Sequence[float]],
        image_size: tuple[int, int],
        crop: Rectangle | None,
        metadata: Mapping[str, object],
    ) -> bool:
        """Advance focus from a fresh request; return True only for a new AF scan."""
        if not self._supported:
            return False
        now = self._clock()
        window = face_window(boxes, image_size, crop, self._maximum, self._rotation)
        self._has_faces = bool(boxes)
        lens = metadata.get("LensPosition")
        valid_lens = (
            isinstance(lens, Real)
            and not isinstance(lens, bool)
            and math.isfinite(float(lens))
            and supports(self._controls, "LensPosition", float(lens))
        )
        if self._state == "scanning":
            af_state = metadata.get("AfState")
            self._seen_scanning |= af_state == 1
            if (
                window is None
                or crop != self._crop
                or (self._window is not None and _overlap(window, self._window) < 0.25)
            ):
                self._restore(now, failed=self._has_faces)
                if not self._has_faces:
                    self._window = None
            elif (
                af_state == 2
                and valid_lens
                and (self._seen_scanning or now - self._started_at >= 0.2)
            ):
                self._good_lens = float(lens)
                self._state = "locked"
                self._last_success = now
            elif af_state == 3 or now - self._started_at >= 6.0:
                self._restore(now)
            return False
        if self._state in ("restoring", "restore_failed"):
            if valid_lens and abs(float(lens) - self._good_lens) <= 0.15:
                self._state = self._restored_state
            elif now - self._started_at >= 2.0:
                self._state = "restore_failed"
                if self._has_faces:
                    raise RuntimeError("Face autofocus could not verify restored lens")
            return False
        if window is None:
            if not self._has_faces:
                self._window = None
            return False
        changed = self._window is None or _overlap(window, self._window) < 0.5
        retry_after = 30.0 if self._state == "fallback" else 15.0
        if (
            now < self._failure_until
            or now - self._last_scan < 2.0
            or (
                not changed
                and now - max(self._last_scan, self._last_success) < retry_after
            )
        ):
            return False
        assert self._set_controls is not None
        controls: dict[str, Any] = {
            "AfMode": 1,  # Auto never moves spontaneously; one explicit trigger below.
            "AfMetering": 1,  # Windows is mandatory; a rectangle alone is insufficient.
            "AfWindows": [window],
            "AfTrigger": 0,
        }
        if supports(self._controls, "AfRange", 0):
            controls["AfRange"] = 0
        self._set_controls(controls)
        self._window, self._crop = window, crop
        self._state = "scanning"
        self._started_at = self._last_scan = now
        self._seen_scanning = False
        return True
