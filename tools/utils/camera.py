# SPDX-License-Identifier: Apache-2.0
"""Lazy Picamera2 frames for small detection images and untouched enrollment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from tools.utils.image import Image


def aligned_lores_size(
    resolution: tuple[int, int], process_width: int
) -> tuple[int, int] | None:
    """Choose an even YUV420 ISP stream near the requested detection size.

    A 32-pixel width alignment avoids unnecessary YUV stride padding for common
    Pi configurations. The actual configured stride is still handled on read.
    """
    width, height = resolution
    if process_width == 0 or process_width >= width:
        return None
    lores_width = ((process_width + 31) // 32) * 32
    lores_height = max(2, int(round(lores_width * height / width / 2)) * 2)
    if lores_width > width or lores_height > height:
        return None
    return lores_width, lores_height


def _transform(
    image: np.ndarray, brightness: float, contrast: float, rotation: int
) -> np.ndarray:
    """Apply the same image adjustments to both streams without annotation."""
    image = Image.adjust_brightness_contrast(image, brightness, contrast)
    if rotation >= 0:
        image = cv2.rotate(image, rotation)
    return np.ascontiguousarray(image)


def _sycc420_to_bgr(yuv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Decode Picamera2 preview sYCC (full-range Rec.601) with its row padding.

    OpenCV's direct I420 conversion assumes limited video range and would turn
    full-range grey 32 into 19. Preview configurations instead default to sYCC.
    Decode its separate planes through OpenCV's full-range YCrCb conversion;
    preserve the main stream's sRGB primaries/transfer rather than changing the
    whole camera to a limited-range video colour space.
    """
    width, height = size
    if (
        yuv.dtype != np.uint8
        or yuv.ndim != 2
        or width % 2
        or height % 2
        or yuv.shape[0] != height * 3 // 2
        or yuv.shape[1] < width
        or yuv.shape[1] % 2
    ):
        raise ValueError("Expected a packed 8-bit YUV420 low-resolution frame")
    stride = yuv.shape[1]
    flat = np.ascontiguousarray(yuv).reshape(-1)
    y_size = height * stride
    chroma_size = height // 2 * (stride // 2)
    luminance = yuv[:height, :width]
    cb = flat[y_size:y_size + chroma_size].reshape(height // 2, stride // 2)[:, :width // 2]
    cr = flat[y_size + chroma_size:].reshape(height // 2, stride // 2)[:, :width // 2]
    cb = cv2.resize(cb, size, interpolation=cv2.INTER_LINEAR)
    cr = cv2.resize(cr, size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.merge((luminance, cr, cb)), cv2.COLOR_YCrCb2BGR)


@dataclass
class CameraFrame:
    """A request-owned frame whose full-resolution copy is materialized lazily.

    ``processing_frame`` is detached from the request and enrollment pixels, so
    drawing labels on it cannot contaminate a saved training photograph.
    ``full_frame`` must be called inside the stream's capture context. Its return
    value is detached and remains valid after that context releases the request.
    """

    processing_frame: np.ndarray
    _request: Any = field(repr=False)
    _brightness: float = field(default=0.0, repr=False)
    _contrast: float = field(default=0.0, repr=False)
    _rotation: int = field(default=-1, repr=False)
    _cached_main: np.ndarray | None = field(default=None, repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def full_frame(self) -> np.ndarray:
        """Return an independent, unannotated full-resolution BGR image."""
        if self._released:
            raise RuntimeError("Full-resolution frames require an active capture context")
        if self._cached_main is None:
            image = self._request.make_array("main")
            if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
                raise ValueError("Expected an 8-bit XRGB8888 main camera frame")
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            self._cached_main = _transform(
                bgr, self._brightness, self._contrast, self._rotation
            )
        return self._cached_main.copy()

    def release(self) -> None:
        """Release the underlying request exactly once."""
        if not self._released:
            self._released = True
            self._request.release()


class CameraStream:
    """Configure Picamera2 lazily and pair processing/enrollment per request.

    The optional low-resolution stream uses the camera ISP, not a second camera
    consumer. YUV420 is the portable choice; Pi 5 also supports RGB888, whose byte
    order is already BGR and avoids chroma subsampling/conversion. A failure to
    configure it is reported rather than silently changing capture behaviour.
    Set ``enable_lores=False`` for the legacy main-stream path. No camera module
    is imported until an instance is constructed.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        process_width: int = 500,
        enable_lores: bool = False,
        frame_rate: float | None = None,
        brightness: float = 0.0,
        contrast: float = 0.0,
        rotation: int = -1,
        camera_factory: Callable[[], Any] | None = None,
        lores_format: str = "YUV420",
    ) -> None:
        """Validate stream options and configure a camera without starting it."""
        if (
            len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in resolution)
        ):
            raise ValueError("Camera resolution must contain two positive integers")
        if isinstance(process_width, bool) or not isinstance(process_width, int) or process_width < 0:
            raise ValueError("process_width must be a non-negative integer")
        if frame_rate is not None and (
            isinstance(frame_rate, bool)
            or not math.isfinite(float(frame_rate))
            or frame_rate <= 0
        ):
            raise ValueError("frame_rate must be finite and positive, or None")
        if rotation not in (-1, 0, 1, 2):
            raise ValueError("rotation must be -1, 0, 1 or 2")
        if not all(math.isfinite(float(value)) for value in (brightness, contrast)):
            raise ValueError("Brightness and contrast must be finite")
        if lores_format not in ("YUV420", "RGB888"):
            raise ValueError("lores_format must be YUV420 or RGB888")
        self.resolution = tuple(resolution)
        self.process_width = process_width
        self.lores_size = aligned_lores_size(self.resolution, process_width) if enable_lores else None
        self.frame_rate = frame_rate
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.rotation = rotation
        self.lores_format = lores_format
        self._started = False
        self._closed = False
        if camera_factory is None:
            from picamera2 import Picamera2

            camera_factory = Picamera2
        self._camera = camera_factory()
        try:
            arguments: dict[str, Any] = {
                "main": {"size": self.resolution, "format": "XRGB8888"},
                "buffer_count": 4,
                "queue": False,
            }
            if self.lores_size is not None:
                arguments["lores"] = {"size": self.lores_size, "format": self.lores_format}
            if frame_rate is not None:
                arguments["controls"] = {"FrameRate": float(frame_rate)}
            config = self._camera.create_preview_configuration(**arguments)
            self._camera.configure(config)
            if self.lores_size is not None:
                lores_config = self._camera.stream_configuration("lores")
                if lores_config["format"] != self.lores_format:
                    raise ValueError("Camera changed the requested low-resolution pixel format")
                self.lores_size = tuple(lores_config["size"])
        except Exception:
            self._camera.close()
            self._closed = True
            raise

    def start(self) -> None:
        """Start the configured camera, including after an external pause."""
        if self._closed:
            raise RuntimeError("Camera stream is closed")
        if not self._started:
            self._camera.start()
            self._started = True

    def stop(self) -> None:
        """Pause capture idempotently without losing the configured streams."""
        if self._started:
            self._camera.stop()
            self._started = False

    def close(self) -> None:
        """Stop and close the camera, including cleanup after failures."""
        if not self._closed:
            self._closed = True
            try:
                self.stop()
            finally:
                self._camera.close()

    @contextmanager
    def capture(self) -> Iterator[CameraFrame]:
        """Yield a fresh processing frame and always release its camera request."""
        if self._closed or not self._started:
            raise RuntimeError("Camera stream must be started before capture")
        request = self._camera.capture_request()
        frame: CameraFrame | None = None
        try:
            frame = CameraFrame(
                np.empty((0, 0, 3), dtype=np.uint8), request,
                self.brightness, self.contrast, self.rotation,
            )
            if self.lores_size is not None:
                width, height = self.lores_size
                lowres = request.make_array("lores")
                if self.lores_format == "RGB888":
                    if (
                        lowres.dtype != np.uint8
                        or lowres.ndim != 3
                        or lowres.shape[2] != 3
                        or lowres.shape[0] != height
                        or lowres.shape[1] < width
                    ):
                        raise ValueError("Expected an 8-bit RGB888 low-resolution frame")
                    # Libcamera's RGB888 stores B,G,R bytes (not R,G,B).
                    bgr = lowres[:, :width]
                else:
                    bgr = _sycc420_to_bgr(lowres, (width, height))
                image = _transform(bgr, self.brightness, self.contrast, self.rotation)
            else:
                image = frame.full_frame()
            if self.process_width and image.shape[1] != self.process_width:
                image = Image.resize(image, width=self.process_width)
            frame.processing_frame = image.copy()
            yield frame
        finally:
            if frame is None:
                request.release()
            else:
                frame.release()
