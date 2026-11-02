# SPDX-License-Identifier: Apache-2.0
"""Exercise pixel-preserving preprocessing improvements on synthetic arrays."""

import cv2
import numpy as np
import pytest

from tools.utils.image import Image


def test_neutral_adjustment_avoids_copy_and_preserves_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A neutral camera configuration needs no full-resolution addWeighted."""
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    def forbidden(*args: object) -> None:
        """Neutral transforms must bypass the OpenCV allocation."""
        pytest.fail("Unexpected full-frame weighted copy")

    monkeypatch.setattr(cv2, "addWeighted", forbidden)
    assert Image.adjust_brightness_contrast(image, "0", 0) is image


def test_adjustment_retains_existing_linear_transform() -> None:
    """Non-neutral settings must keep the original contrast/brightness formula."""
    image = np.full((3, 4, 3), 20, dtype=np.uint8)
    adjusted = Image.adjust_brightness_contrast(image, 5, 50)
    assert np.all(adjusted == 35)
    assert np.all(image == 20)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_invalid_adjustments_rejected(value: float) -> None:
    """Reject non-finite inputs rather than generating misleading pixels."""
    with pytest.raises(ValueError):
        Image.adjust_brightness_contrast(np.zeros((2, 2, 3), np.uint8), value)


def test_resize_noop_and_portrait_aspect_ratio() -> None:
    """Keep existing dimensions when possible and avoid zero-size thin frames."""
    image = np.zeros((8, 4, 3), dtype=np.uint8)
    assert Image.resize(image, width=4) is image
    assert Image.resize(image) is image
    assert Image.resize(image, width=2).shape == (4, 2, 3)
    assert Image.resize(image, height=4).shape == (4, 2, 3)
    assert Image.resize(np.zeros((1, 100, 3), np.uint8), width=1).shape == (1, 1, 3)


@pytest.mark.parametrize("width", [0, -1, True])
def test_invalid_resize_width_rejected(width: int) -> None:
    """Produce a clear failure before invoking OpenCV with illegal dimensions."""
    with pytest.raises(ValueError):
        Image.resize(np.zeros((3, 4, 3), np.uint8), width=width)
