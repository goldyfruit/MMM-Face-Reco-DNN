# SPDX-License-Identifier: Apache-2.0
"""Synthetic camera tests that never open hardware or user face data."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import cv2
import numpy as np
import pytest

from tools.utils.camera import CameraStream, aligned_lores_size
from tools.utils.focus import FaceFocusController, face_window


class RequestDouble:
    """Record reads/releases from a synchronized camera request."""

    def __init__(self, arrays: dict[str, np.ndarray], metadata: object = None) -> None:
        """Retain synthetic arrays and request call counts."""
        self.arrays = arrays
        self.reads: list[str] = []
        self.releases = 0
        self.metadata = metadata

    def get_metadata(self) -> object:
        """Return synthetic metadata without opening or recording a camera."""
        return self.metadata

    def make_array(self, stream: str) -> np.ndarray:
        """Simulate Picamera2's detached stream-array copy."""
        self.reads.append(stream)
        return self.arrays[stream].copy()

    def release(self) -> None:
        """Record a reference release."""
        self.releases += 1


class CameraDouble:
    """Implement the small subset of Picamera2 used by CameraStream."""

    def __init__(self, request: RequestDouble, camera_controls: object = None) -> None:
        """Prepare the next synthetic request."""
        self.request = request
        self.options: dict[str, Any] = {}
        self.starts = self.stops = self.closes = 0
        self.camera_controls = camera_controls

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
    request = RequestDouble(
        {
            "main": main,
            "lores": full_range_sycc420(small),
        }
    )
    return CameraDouble(request), request


def full_range_sycc420(bgr: np.ndarray) -> np.ndarray:
    """Pack synthetic full-range YCbCr planes like Picamera2 preview buffers."""
    height, width = bgr.shape[:2]
    converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = converted[:, :, 0]
    cb = cv2.resize(
        converted[:, :, 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA
    )
    cr = cv2.resize(
        converted[:, :, 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA
    )
    return np.concatenate((y.reshape(-1), cb.reshape(-1), cr.reshape(-1))).reshape(
        height * 3 // 2, width
    )


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
    request.arrays["lores"] = full_range_sycc420(padded)
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert frame.processing_frame.shape == (24, 32, 3)
        assert np.allclose(frame.processing_frame[:, -1], (0, 0, 255), atol=2)


@pytest.mark.parametrize(
    "color",
    [
        (32, 32, 32),
        (224, 224, 224),
        (70, 120, 170),
        (130, 160, 200),
    ],
)
def test_sycc_full_range_preserves_grey_and_color(color: tuple[int, int, int]) -> None:
    """Do not decode full-range preview values with limited-range I420 math."""
    camera, request = synthetic_camera()
    bgr = np.full((24, 32, 3), color, dtype=np.uint8)
    request.arrays["lores"] = full_range_sycc420(bgr)
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert np.allclose(frame.processing_frame, color, atol=2)
        if color == (32, 32, 32):
            # The previous converter incorrectly stretched this grey to 19.
            old = cv2.cvtColor(request.arrays["lores"], cv2.COLOR_YUV2BGR_I420)
            assert tuple(old[0, 0]) == (19, 19, 19)
            assert tuple(frame.processing_frame[0, 0]) == color


def test_rgb888_lores_keeps_bgr_byte_order_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi5 RGB888 output is already BGR and needs neither I420 nor channel swap."""
    camera, request = synthetic_camera()
    lowres = np.full((24, 40, 3), (10, 20, 30), dtype=np.uint8)
    lowres[:, 32:] = (200, 150, 100)  # Simulate row padding without leaking it.
    request.arrays["lores"] = lowres
    stream = CameraStream(
        (64, 48),
        32,
        True,
        camera_factory=lambda: camera,
        lores_format="RGB888",
    )
    stream.start()

    def forbid(*args: object) -> None:
        """RGB888 processing must not call a YUV conversion or swap B/R."""
        pytest.fail("Unexpected colour conversion on RGB888")

    monkeypatch.setattr(cv2, "cvtColor", forbid)
    with stream.capture() as frame:
        assert camera.options["lores"]["format"] == "RGB888"
        assert frame.processing_frame.shape == (24, 32, 3)
        assert np.all(frame.processing_frame == (10, 20, 30))
        assert request.reads == ["lores"]
    assert request.releases == 1


def test_invalid_rgb888_frame_releases_request() -> None:
    """Do not reinterpret a four-channel frame as the requested RGB888 stream."""
    camera, request = synthetic_camera()
    request.arrays["lores"] = np.zeros((24, 32, 4), dtype=np.uint8)
    stream = CameraStream(
        (64, 48),
        32,
        True,
        camera_factory=lambda: camera,
        lores_format="RGB888",
    )
    stream.start()
    with pytest.raises(ValueError, match="RGB888"):
        with stream.capture():
            pytest.fail("Invalid pixel format cannot reach recognition")
    assert request.releases == 1


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
        (64, 48),
        32,
        True,
        brightness=5,
        rotation=cv2.ROTATE_90_CLOCKWISE,
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


def test_advertised_autofocus_uses_continuous_normal_without_fast_or_manual_shutter() -> (
    None
):
    """Enable motorized focus without requesting unsupported speed or fixed AE."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 2, 0),
        "AfRange": (0, 2, 0),
        "AeExposureMode": (0, 2, 0),
    }
    stream = CameraStream(
        camera_factory=lambda: camera, frame_rate=15, autofocus="continuous"
    )
    assert camera.options["controls"] == {
        "FrameRate": 15.0,
        "AfMode": 2,
        "AfRange": 0,
        "AeExposureMode": 0,
    }
    assert "AfSpeed" not in camera.options["controls"]
    assert "ExposureTime" not in camera.options["controls"]
    assert "AeEnable" not in camera.options["controls"]
    assert "AwbEnable" not in camera.options["controls"]
    selected = stream.configured_controls
    selected["AfMode"] = 0
    assert stream.configured_controls["AfMode"] == 2


@pytest.mark.parametrize("advertised", [None, {}, {"Brightness": (-1, 1, 0)}])
def test_fixed_focus_or_usb_camera_does_not_receive_optional_controls(
    advertised: object,
) -> None:
    """Absent AF/AE capabilities must preserve the existing camera path."""
    camera, _ = synthetic_camera()
    camera.camera_controls = advertised
    stream = CameraStream(camera_factory=lambda: camera, exposure_mode="short")
    assert "controls" not in camera.options
    assert stream.configured_controls == {}


def test_optional_control_ranges_and_partial_capabilities_are_respected() -> None:
    """Do not infer AF or a short exposure profile from another supported knob."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 1, 0),  # No continuous AF.
        "AfRange": (0, 2, 0),
        "AeExposureMode": (0, 0, 0),  # No short profile.
    }
    CameraStream(
        camera_factory=lambda: camera, autofocus="continuous", exposure_mode="short"
    )
    assert "controls" not in camera.options
    camera.camera_controls = {"AfMode": (0, 2, 0)}
    CameraStream(camera_factory=lambda: camera, autofocus="continuous")
    assert camera.options["controls"] == {"AfMode": 2}


@pytest.mark.parametrize(
    ("autofocus", "expected"),
    [
        ("default", {"AeExposureMode": 1}),
        ("manual", {"AfMode": 0, "AeExposureMode": 1}),
    ],
)
def test_focus_override_and_opt_in_short_exposure(
    autofocus: str, expected: dict[str, int]
) -> None:
    """Allow untouched or manual focus and explicit AE-short without forcing gain."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 2, 0),
        "AfRange": (0, 2, 0),
        "AeExposureMode": (0, 2, 0),
    }
    CameraStream(
        camera_factory=lambda: camera, autofocus=autofocus, exposure_mode="short"
    )
    assert camera.options["controls"] == expected


def test_generic_default_preserves_existing_focus_even_on_autofocus_camera() -> None:
    """Do not change known working focus or let a foreground reflection steal it."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 2, 0),
        "AfRange": (0, 2, 0),
        "LensPosition": (0.0, 15.0, 1.0),
    }
    stream = CameraStream(camera_factory=lambda: camera)
    assert stream.autofocus == "default"
    assert stream.lens_position is None
    assert "controls" not in camera.options


@pytest.mark.parametrize("position", [0.0, 1.0, 15.0])
def test_manual_lens_position_applies_supported_distance(position: float) -> None:
    """Use explicit dioptres only when manual focus and its distance are supported."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 2, 0),
        "AfRange": (0, 2, 0),
        "LensPosition": (0.0, 15.0, 1.0),
    }
    stream = CameraStream(
        camera_factory=lambda: camera, autofocus="manual", lens_position=position
    )
    assert stream.configured_controls == {"AfMode": 0, "LensPosition": position}
    assert "AfRange" not in camera.options["controls"]


@pytest.mark.parametrize(
    "advertised",
    [
        {},
        {"LensPosition": (0.0, 15.0, 1.0)},
        {"AfMode": (1, 2, 1), "LensPosition": (0.0, 15.0, 1.0)},
    ],
)
def test_manual_lens_is_not_sent_without_manual_focus_support(
    advertised: dict[str, tuple[float, float, float]],
) -> None:
    """Fixed-focus and USB cameras must not receive unsupported lens commands."""
    camera, _ = synthetic_camera()
    camera.camera_controls = advertised
    stream = CameraStream(
        camera_factory=lambda: camera, autofocus="manual", lens_position=1.0
    )
    assert stream.configured_controls == {}


def test_manual_lens_respects_advertised_distance_range() -> None:
    """Do not send a requested distance outside the camera's supported range."""
    camera, _ = synthetic_camera()
    camera.camera_controls = {
        "AfMode": (0, 2, 0),
        "LensPosition": (2.0, 15.0, 4.0),
    }
    stream = CameraStream(
        camera_factory=lambda: camera, autofocus="manual", lens_position=1.0
    )
    assert stream.configured_controls == {"AfMode": 0}


def test_request_metadata_is_bounded_scalar_only_and_detached_from_private_data() -> (
    None
):
    """Expose sensor diagnostics without leaking pixels, names or arbitrary keys."""

    class AfState(IntEnum):
        """Represent libcamera's numeric metadata enum."""

        FOCUSED = 2

    camera, request = synthetic_camera()
    request.metadata = {
        "ExposureTime": 12000,
        "FrameDuration": np.int64(66666),
        "AnalogueGain": np.float32(2.0),
        "LensPosition": 1.2,
        "AfState": AfState.FOCUSED,
        "AeLocked": np.bool_(True),
        "SensorTimestamp": 123456789,
        "DigitalGain": float("inf"),
        "Lux": float("nan"),
        "ColourTemperature": "4200",
        "FocusFoM": np.array([1, 2]),
        "AfMode": 10**1000,
        "AeExposureMode": -(2**63) - 1,
        "identity": "private-name",
        "image": request.arrays["main"],
        "ScalerCrop": [0, 0, 1920, 1080],
    }
    expected = {
        "ExposureTime": 12000,
        "FrameDuration": 66666,
        "AnalogueGain": 2.0,
        "LensPosition": 1.2,
        "AfState": 2,
        "AeLocked": True,
        "SensorTimestamp": 123456789,
    }
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with stream.capture() as frame:
        assert frame.metadata == expected
        assert stream.latest_metadata == expected
        assert request.reads == ["lores"]
        request.metadata["ExposureTime"] = 90000
        frame.metadata["LensPosition"] = 8.0
        stream.latest_metadata["SensorTimestamp"] = 0
        assert frame.metadata == expected
        assert stream.latest_metadata == expected
    assert frame.metadata == expected  # Scalars remain usable after buffer release.
    assert request.releases == 1


def test_missing_metadata_and_pause_never_reuse_previous_diagnostics() -> None:
    """A new request, failed frame or pause must not appear to have fresh old data."""
    camera, request = synthetic_camera()
    request.metadata = {"ExposureTime": 1000}
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    assert stream.latest_metadata == {}
    stream.start()
    with stream.capture():
        pass
    assert stream.latest_metadata == {"ExposureTime": 1000}
    stream.stop()
    assert stream.latest_metadata == {}
    stream.start()
    request.metadata = ["not a metadata mapping"]
    with stream.capture() as frame:
        assert frame.metadata == {}
    assert stream.latest_metadata == {}
    request.metadata = {"ExposureTime": 2000}
    with stream.capture():
        pass
    request.arrays["lores"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="YUV420"):
        with stream.capture():
            pytest.fail("Malformed frame cannot refresh diagnostics")
    assert stream.latest_metadata == {}


def test_metadata_read_error_still_releases_request() -> None:
    """Treat request failures as failures without retaining buffers or old health."""
    camera, request = synthetic_camera()

    def fail() -> object:
        """Simulate a failing request while querying its own metadata."""
        raise RuntimeError("metadata request failed")

    request.get_metadata = fail
    stream = CameraStream((64, 48), 32, True, camera_factory=lambda: camera)
    stream.start()
    with pytest.raises(RuntimeError, match="metadata request failed"):
        with stream.capture():
            pytest.fail("Request errors must propagate")
    assert request.releases == 1
    assert stream.latest_metadata == {}


def face_controls() -> dict[str, Any]:
    """Advertise a synthetic complete autofocus/window contract."""
    return {
        "AfMode": (0, 2, 0),
        "AfRange": (0, 2, 0),
        "AfMetering": (0, 1, 0),
        "AfTrigger": (0, 1, 0),
        "AfWindows": ((0, 0, 0, 0), (65535, 65535, 65535, 65535), (0, 0, 0, 0)),
        "LensPosition": (0.0, 15.0, 1.0),
    }


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (-1, (275, 340, 350, 280)),
        (0, (275, 580, 350, 280)),
        (1, (575, 580, 350, 280)),
        (2, (575, 340, 350, 280)),
    ],
)
def test_face_window_undoes_rotation_resize_and_crop_origin(
    rotation: int, expected: tuple[int, int, int, int]
) -> None:
    """Use sensor-pixel crop offsets after undoing the actual software transform."""
    crop = (100, 200, 1000, 800)
    maximum = (48, 40, 2000, 1600)
    window = face_window([(20, 60, 120, 10)], (200, 100), crop, maximum, rotation)
    assert window == pytest.approx(expected, abs=1)
    resized = face_window([(40, 120, 240, 20)], (400, 200), crop, maximum, rotation)
    assert resized == window


def test_face_window_target_is_largest_then_central_and_order_independent() -> None:
    """Several faces never produce several competing AF distances or identity ties."""
    crop = maximum = (100, 200, 1000, 800)
    boxes = [(10, 40, 40, 10), (40, 160, 140, 60), (10, 100, 110, 0)]
    chosen = face_window(boxes, (200, 200), crop, maximum, -1)
    assert chosen == face_window(list(reversed(boxes)), (200, 200), crop, maximum, -1)
    assert chosen == face_window([boxes[1]], (200, 200), crop, maximum, -1)
    assert chosen is not None
    assert chosen[2] < 500  # Inset excludes the edges of the selected face.


@pytest.mark.parametrize(
    ("boxes", "crop"),
    [
        ([], (100, 200, 1000, 800)),
        ([(10, 20, 15, 10)], (100, 200, 1000, 800)),
        ([(10, float("nan"), 80, 10)], (100, 200, 1000, 800)),
        ([(10, 80, 80, 10)], None),
        ([(10, 80, 80, 10)], (0, 0, 1000, 800)),
        ([(10, 80, 80, 10)], (100, 200, 3000, 800)),
    ],
)
def test_invalid_or_missing_face_geometry_cannot_enable_global_af(
    boxes: list[tuple[float, ...]], crop: tuple[int, int, int, int] | None
) -> None:
    """Ignore empty/invalid boxes and unproven sensor geometry entirely."""
    assert face_window(boxes, (200, 200), crop, (100, 200, 1000, 800), -1) is None


def focus_controller() -> tuple[FaceFocusController, list[dict[str, Any]], list[float]]:
    """Build a camera-free controller with a deterministic clock and command log."""
    commands: list[dict[str, Any]] = []
    clock = [0.0]
    controller = FaceFocusController(
        face_controls(), (0, 0, 1000, 800), commands.append, 1.0, -1, lambda: clock[0]
    )
    return controller, commands, clock


def test_face_scan_requires_roi_holds_lock_and_ignores_detector_jitter() -> None:
    """Trigger only in a real face window and avoid continuously hunting a face."""
    controller, commands, clock = focus_controller()
    crop = (0, 0, 1000, 800)
    box = [(40, 140, 140, 40)]
    assert not controller.update([], (200, 200), crop, {})
    assert not commands
    assert controller.update(box, (200, 200), crop, {"LensPosition": 1.0})
    assert commands[-1] == {
        "AfMode": 1,
        "AfRange": 0,
        "AfMetering": 1,
        "AfWindows": [(275, 220, 350, 280)],
        "AfTrigger": 0,
    }
    assert controller.status["settled"] is False
    clock[0] = 0.5
    assert not controller.update(
        box, (200, 200), crop, {"AfState": 1, "LensPosition": 2.0}
    )
    assert len(commands) == 1
    clock[0] = 1.0
    assert not controller.update(
        box, (200, 200), crop, {"AfState": 2, "LensPosition": 1.5}
    )
    assert controller.status["settled"] is True
    assert controller.status["lens_position"] == 1.5
    clock[0] = 3.0
    assert not controller.update([(41, 141, 141, 41)], (200, 200), crop, {})
    assert len(commands) == 1


@pytest.mark.parametrize("failure", ["failed", "timeout", "crop_changed", "no_face"])
def test_face_scan_failure_cancels_and_restores_without_global_fallback(
    failure: str,
) -> None:
    """Bound bad scans, restore known focus and let disappearance clear identity."""
    controller, commands, clock = focus_controller()
    crop = (0, 0, 1000, 800)
    boxes = [(40, 140, 140, 40)]
    controller.update(boxes, (200, 200), crop, {})
    clock[0] = 6.5 if failure == "timeout" else 0.5
    current_crop = (100, 100, 800, 600) if failure == "crop_changed" else crop
    current_boxes = [] if failure == "no_face" else boxes
    metadata = {"AfState": 3 if failure == "failed" else 1, "LensPosition": 5.79}
    assert not controller.update(current_boxes, (200, 200), current_crop, metadata)
    assert commands[-1] == {"AfTrigger": 1, "AfMode": 0, "LensPosition": 1.0}
    assert controller.status["state"] == "restoring"
    assert controller.status["settled"] is (failure == "no_face")
    clock[0] += 0.5
    controller.update(current_boxes, (200, 200), current_crop, {"LensPosition": 1.0})
    assert controller.status["state"] == (
        "waiting" if failure == "no_face" else "fallback"
    )
    assert controller.status["settled"] is True
    assert not controller.update(current_boxes, (200, 200), current_crop, {})
    assert len(commands) == 2


def test_unverified_restore_faults_and_resume_reasserts_last_good_lens() -> None:
    """A wrong/missing lens must trigger bounded recovery, never permit bad identity."""
    controller, commands, clock = focus_controller()
    crop = (0, 0, 1000, 800)
    box = [(40, 140, 140, 40)]
    controller.update(box, (200, 200), crop, {})
    clock[0] = 1.0
    controller.update(box, (200, 200), crop, {"AfState": 2, "LensPosition": 1.5})
    clock[0] = 16.5
    controller.update(box, (200, 200), crop, {})
    clock[0] = 23.0
    controller.update(box, (200, 200), crop, {})
    assert commands[-1]["LensPosition"] == 1.5
    clock[0] = 25.1
    with pytest.raises(RuntimeError, match="could not verify restored lens"):
        controller.update(box, (200, 200), crop, {"LensPosition": 5.79})
    assert controller.status["state"] == "restore_failed"
    assert controller.status["settled"] is False
    assert not controller.update([], (200, 200), crop, {})
    assert controller.status["settled"] is True  # No-face logout remains safe.
    controller.pause()
    controller.resume()
    assert commands[-1] == {"AfMode": 0, "LensPosition": 1.5}
    assert controller.status["state"] == "restoring"
    controller.update(box, (200, 200), crop, {"LensPosition": 1.5})
    assert controller.status["state"] == "waiting"


def test_failed_scan_cooldown_is_global_even_when_faces_move() -> None:
    """A permanent AF failure cannot hunt every two seconds as ROI boxes move."""
    controller, commands, clock = focus_controller()
    crop = (0, 0, 1000, 800)
    box = [(40, 140, 140, 40)]
    controller.update(box, (200, 200), crop, {})
    clock[0] = 0.5
    controller.update(box, (200, 200), crop, {"AfState": 3, "LensPosition": 5.79})
    clock[0] = 1.0
    controller.update(box, (200, 200), crop, {"LensPosition": 1.0})
    for stamp, moved in [
        (3.0, [(0, 70, 70, 0)]),
        (10.0, box),
        (29.0, [(120, 200, 200, 120)]),
    ]:
        clock[0] = stamp
        assert not controller.update(moved, (200, 200), crop, {"LensPosition": 1.0})
    assert len(commands) == 2
    clock[0] = 31.0
    assert controller.update(
        [(120, 200, 200, 120)], (200, 200), crop, {"LensPosition": 1.0}
    )
    assert len(commands) == 3


@pytest.mark.parametrize(
    "missing", ["AfMode", "AfMetering", "AfWindows", "AfTrigger", "LensPosition"]
)
def test_incomplete_face_af_contract_never_requests_an_unmetered_scan(
    missing: str,
) -> None:
    """Unsupported USB/fixed-focus controls cannot become global autofocus."""
    controls = face_controls()
    del controls[missing]
    commands: list[dict[str, Any]] = []
    controller = FaceFocusController(
        controls, (0, 0, 1000, 800), commands.append, 1.0, -1
    )
    assert not controller.update(
        [(40, 140, 140, 40)], (200, 200), (0, 0, 1000, 800), {}
    )
    assert controller.status == {
        "supported": False,
        "state": "unsupported",
        "settled": True,
        "window": None,
        "lens_position": 1.0,
    }
    assert not commands


def test_face_focus_uses_exact_request_lores_crop_and_keeps_geometry_private() -> None:
    """Pi5's lores crop can differ from stream0; use its own request metadata."""
    camera, request = synthetic_camera()
    camera.camera_controls = face_controls()
    camera.camera_properties = {"ScalerCropMaximum": (0, 0, 2000, 1600)}
    commands: list[dict[str, Any]] = []
    camera.set_controls = commands.append
    request.metadata = {
        "ScalerCrop": (0, 0, 2000, 1600),
        "ScalerCrops": [(0, 0, 2000, 1600), (100, 200, 1000, 800)],
        "LensPosition": 1.0,
    }
    stream = CameraStream(
        (64, 48), 32, True, camera_factory=lambda: camera, autofocus="face"
    )
    assert stream.configured_controls == {"AfMode": 0, "LensPosition": 1.0}
    assert camera.options["lores"]["preserve_ar"] is False
    stream.start()
    with stream.capture() as frame:
        assert frame.metadata == {"LensPosition": 1.0}
        assert not stream.focus_on_faces(
            frame, [(0, 32, 24, 0)]
        )  # Verify the seed first.
        assert stream.face_focus_status["settled"] is True
    with stream.capture() as frame:
        assert stream.focus_on_faces(frame, [(0, 32, 24, 0)])
        assert commands[-1]["AfWindows"] == [(250, 320, 700, 560)]
        assert not stream.focus_on_faces(frame, [(0, 32, 24, 0)])  # Once per request.
        assert stream.face_focus_status["settled"] is False
    with pytest.raises(ValueError, match="active camera frame"):
        stream.focus_on_faces(frame, [(0, 32, 24, 0)])
    stream.stop()
    assert commands[-1] == {"AfTrigger": 1, "AfMode": 0, "LensPosition": 1.0}


def test_malformed_per_stream_crop_never_substitutes_wrong_main_crop() -> None:
    """A present but incomplete ScalerCrops list must not focus the wrong region."""
    camera, request = synthetic_camera()
    camera.camera_controls = face_controls()
    camera.camera_properties = {"ScalerCropMaximum": (0, 0, 2000, 1600)}
    commands: list[dict[str, Any]] = []
    camera.set_controls = commands.append
    request.metadata = {
        "ScalerCrop": (0, 0, 2000, 1600),
        "ScalerCrops": [(0, 0, 2000, 1600)],
    }
    stream = CameraStream(
        (64, 48), 32, True, camera_factory=lambda: camera, autofocus="face"
    )
    stream.start()
    with stream.capture() as frame:
        assert not stream.focus_on_faces(frame, [(0, 32, 24, 0)])
    assert not any(command.get("AfTrigger") == 0 for command in commands)


@pytest.mark.parametrize(
    "options",
    [
        {"resolution": (0, 48)},
        {"resolution": (64, -1)},
        {"process_width": -1},
        {"process_width": True},
        {"frame_rate": 0},
        {"frame_rate": -1},
        {"frame_rate": float("nan")},
        {"frame_rate": float("inf")},
        {"rotation": 3},
        {"brightness": float("nan")},
        {"lores_format": "BGR888"},
        {"autofocus": "fast"},
        {"autofocus": True},
        {"exposure_mode": "long"},
        {"exposure_mode": None},
        {"autofocus": "manual", "lens_position": -0.1},
        {"autofocus": "manual", "lens_position": float("nan")},
        {"autofocus": "manual", "lens_position": float("inf")},
        {"autofocus": "manual", "lens_position": True},
        {"autofocus": "manual", "lens_position": "1.0"},
        {"autofocus": "continuous", "lens_position": 1.0},
        {"autofocus": "default", "lens_position": 1.0},
    ],
)
def test_invalid_settings_do_not_open_camera(options: dict[str, Any]) -> None:
    """Reject invalid controls before touching hardware."""

    def forbid() -> None:
        """A validation error must happen before camera construction."""
        pytest.fail("Camera should not be opened")

    with pytest.raises(ValueError):
        CameraStream(camera_factory=forbid, **options)
