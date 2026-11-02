# SPDX-License-Identifier: Apache-2.0
"""Synthetic camera tests that never open hardware or user face data."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from tools.utils.camera import CameraStream, aligned_lores_size


class RequestDouble:
    """Record reads/releases from a synchronized camera request."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        """Retain synthetic arrays and request call counts."""
        self.arrays = arrays
        self.reads: list[str] = []
        self.releases = 0

    def make_array(self, stream: str) -> np.ndarray:
        """Simulate Picamera2's detached stream-array copy."""
        self.reads.append(stream)
        return self.arrays[stream].copy()

    def release(self) -> None:
        """Record a reference release."""
        self.releases += 1


class CameraDouble:
    """Implement the small subset of Picamera2 used by CameraStream."""

    def __init__(self, request: RequestDouble) -> None:
        """Prepare the next synthetic request."""
        self.request = request
        self.options: dict[str, Any] = {}
        self.starts = self.stops = self.closes = 0

    def create_preview_configuration(self, **kwargs: Any) -> dict[str, Any]:
        """Keep the requested configuration for assertions."""
        self.options = kwargs
        return kwargs

    def configure(self, config: dict[str, Any]) -> None:
        """Accept a generated stream configuration."""
        self.config = config

    def stream_configuration(self, name: str) -> dict[str, Any]:
        """Return the configured stream geometry."""
        return self.config[name]

    def start(self) -> None:
        """Record a camera start."""
        self.starts += 1

    def stop(self) -> None:
        """Record a camera pause."""
        self.stops += 1

    def close(self) -> None:
        """Record resource cleanup."""
        self.closes += 1

    def capture_request(self) -> RequestDouble:
        """Return the next request."""
        return self.request


def synthetic_camera() -> tuple[CameraDouble, RequestDouble]:
    """Build main BGRX and small I420 streams with known pixel values."""
    main = np.empty((48, 64, 4), dtype=np.uint8)
    main[:] = (10, 20, 30, 255)
    small = np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8)
    request = RequestDouble({
        "main": main,
        "lores": cv2.cvtColor(small, cv2.COLOR_BGR2YUV_I420),
    })
    return CameraDouble(request), request


def test_lores_size_preserves_common_aspect_ratio_and_alignment() -> None:
    """Keep standard Pi5 processing close to 500px without odd YUV geometry."""
    assert aligned_lores_size((1920, 1080), 500) == (512, 288)
    assert aligned_lores_size((1920, 1080), 0) is None
    assert aligned_lores_size((1920, 1080), 1920) is None
    assert aligned_lores_size((1920, 1080), 2000) is None
    assert aligned_lores_size((100, 80), 99) is None


def test_isp_processing_does_not_copy_full_frame_until_enrollment() -> None:
    """Read only lores normally, then materialize synchronized main on demand."""
    camera, request = synthetic_camera()
    stream = CameraStream((64, 48), 32, True, 10, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert request.reads == ["lores"]
        assert frame.processing_frame.shape == (24, 32, 3)
        assert camera.options["controls"] == {"FrameRate": 10.0}
        assert camera.options["queue"] is False
        assert camera.options["main"]["size"] == (64, 48)
        assert camera.options["lores"]["size"] == (32, 24)
        assert np.allclose(frame.processing_frame[0, 0], (0, 0, 255), atol=2)
        frame.processing_frame[:] = 255  # Drawing on preview must not train on labels.
        full = frame.full_frame()
        assert request.reads == ["lores", "main"]
        assert full.shape == (48, 64, 3)
        assert tuple(full[0, 0]) == (10, 20, 30)
        assert not np.shares_memory(frame.processing_frame, full)
        full[:] = 0
        assert tuple(frame.full_frame()[0, 0]) == (10, 20, 30)
        assert request.reads == ["lores", "main"]
    assert request.releases == 1
    with pytest.raises(RuntimeError, match="active capture context"):
        frame.full_frame()


def test_padding_is_removed_after_yuv_conversion() -> None:
    """Handle Picamera2's packed I420 arrays with a padded row stride."""
    camera, request = synthetic_camera()
    padded = np.empty((24, 64, 3), dtype=np.uint8)
    padded[:, :32] = (0, 0, 255)
    padded[:, 32:] = (255, 0, 0)
    request.arrays["lores"] = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert frame.processing_frame.shape == (24, 32, 3)
        assert np.allclose(frame.processing_frame[:, -1], (0, 0, 255), atol=2)


def test_legacy_path_keeps_annotations_out_of_full_size_dataset() -> None:
    """Own separate preview pixels even without an ISP stream or resizing."""
    camera, request = synthetic_camera()
    stream = CameraStream((64, 48), 0, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert request.reads == ["main"]
        frame.processing_frame[:] = (255, 255, 255)
        assert tuple(frame.full_frame()[0, 0]) == (10, 20, 30)
    assert "lores" not in camera.options
    assert "controls" not in camera.options
    assert request.releases == 1


def test_rotation_and_brightness_apply_to_both_streams() -> None:
    """Transform the processing stream and the saved original consistently."""
    camera, _ = synthetic_camera()
    stream = CameraStream(
        (64, 48), 32, True, brightness=5, rotation=cv2.ROTATE_90_CLOCKWISE,
        camera_factory=lambda: camera,
    )
    stream.start()
    with stream.capture() as frame:
        assert frame.processing_frame.shape == (42, 32, 3)
        assert frame.full_frame().shape == (64, 48, 3)
        assert tuple(frame.full_frame()[0, 0]) == (15, 25, 35)


def test_request_released_on_consumer_exception() -> None:
    """Return buffers even when detection or encoding raises."""
    camera, request = synthetic_camera()
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with pytest.raises(LookupError, match="detector failed"):
        with stream.capture():
            raise LookupError("detector failed")
    assert request.releases == 1


def test_request_released_when_pixel_conversion_fails() -> None:
    """Do not leak a buffer after a malformed frame."""
    camera, request = synthetic_camera()
    request.arrays["lores"] = np.zeros((3, 3), dtype=np.float32)
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with pytest.raises(ValueError, match="YUV420"):
        with stream.capture():
            pytest.fail("Invalid frames cannot reach recognition")
    assert request.releases == 1


def test_start_stop_and_close_support_external_trigger() -> None:
    """A pause must not require a second camera object or reconfiguration."""
    camera, _ = synthetic_camera()
    stream = CameraStream((64, 48), camera_factory=lambda: camera)
    with pytest.raises(RuntimeError, match="started"):
        with stream.capture():
            pass
    stream.start()
    stream.start()
    stream.stop()
    stream.stop()
    stream.start()
    stream.close()
    stream.close()
    assert (camera.starts, camera.stops, camera.closes) == (2, 2, 1)
    with pytest.raises(RuntimeError, match="closed"):
        stream.start()


def test_configuration_failure_closes_camera_and_propagates() -> None:
    """Report hardware/configuration failures instead of silent fallback."""
    camera, _ = synthetic_camera()

    def fail(config: dict[str, Any]) -> None:
        """Simulate an unsupported stream combination."""
        raise RuntimeError("unsupported camera configuration")

    camera.configure = fail
    with pytest.raises(RuntimeError, match="unsupported camera configuration"):
        CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    assert camera.closes == 1


@pytest.mark.parametrize("options", [
    {"resolution": (0, 48)}, {"resolution": (64, -1)},
    {"process_width": -1}, {"process_width": True},
    {"frame_rate": 0}, {"frame_rate": -1}, {"frame_rate": float("nan")},
    {"frame_rate": float("inf")}, {"rotation": 3}, {"brightness": float("nan")},
])
def test_invalid_settings_do_not_open_camera(options: dict[str, Any]) -> None:
    """Reject invalid controls before touching hardware."""
    def forbid() -> None:
        """A validation error must happen before camera construction."""
        pytest.fail("Camera should not be opened")

    with pytest.raises(ValueError):
        CameraStream(camera_factory=forbid, **options)
