# SPDX-License-Identifier: Apache-2.0
"""Test opt-in postcard artwork using synthetic pixels and fixed capture times."""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from tools.utils.postcard import CREAM, GREEN, render_postcard
from tools.utils.arguments import Arguments


def test_postcard_preserves_full_family_image_outside_its_added_mat() -> None:
    """No decoration, crop or in-place mutation reaches the original family frame."""
    pixels = np.full((600, 900, 3), (20, 80, 120), np.uint8)
    pixels[0, 0] = (1, 2, 3)
    pixels[-1, -1] = (4, 5, 6)
    before = pixels.copy()
    image = render_postcard(pixels, "en-US", 1793926800)
    assert np.array_equal(pixels, before)
    assert not np.shares_memory(pixels, image)
    assert np.array_equal(image[36:636, 40:940], before)
    assert tuple(image[0, 0]) == CREAM
    assert tuple(image[18, 18]) == GREEN


def test_date_uses_configured_zone_at_capture_not_host_day_or_save_time() -> None:
    """The same instant belongs to Dec24 in Toronto and Dec25 in Paris."""
    instant = datetime(2026, 12, 25, 2, 30, tzinfo=timezone.utc).timestamp()
    same_local_day = datetime(2026, 12, 24, 12, tzinfo=timezone.utc).timestamp()
    pixels = np.zeros((600, 900, 3), np.uint8)
    toronto = render_postcard(pixels, "en-US", instant)
    earlier = render_postcard(pixels, "en-US", same_local_day)
    paris = render_postcard(pixels, "en-US", instant, "Europe/Paris")
    assert np.array_equal(toronto, earlier)
    assert not np.array_equal(toronto, paris)
    assert np.array_equal(toronto[:700], paris[:700])


def test_french_caption_has_vector_circumflex_and_own_date_order() -> None:
    """The French title includes Ô rather than unsupported UTF-8 question marks."""
    pixels = np.zeros((600, 900, 3), np.uint8)
    image = render_postcard(pixels, "fr-FR", 1793926800)
    title_height = cv2.getTextSize("O", cv2.FONT_HERSHEY_COMPLEX, 980 / 850, 2)[0][1]
    baseline = 36 + 600 + 57
    accent = image[baseline - title_height - 12 : baseline - title_height - 2, 42:120]
    assert np.any(np.all(accent == GREEN, axis=2))
    assert not np.array_equal(image, render_postcard(pixels, "en-US", 1793926800))


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1, 1), (4000, 3000)])
def test_landscape_portrait_and_small_postcards_stay_bounded(
    size: tuple[int, int],
) -> None:
    """Portrait cameras retain their full view and fit the same bounded JPEG path."""
    width, height = size
    image = render_postcard(np.zeros((height, width, 3), np.uint8), "fr-FR", 0)
    assert image.dtype == np.uint8 and image.shape[2] == 3
    assert max(image.shape[:2]) <= 1200


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((0, 2, 3), np.uint8),
        np.zeros((2, 2), np.uint8),
        np.zeros((2, 2, 3), float),
    ],
)
def test_invalid_source_frame_is_rejected(pixels: np.ndarray) -> None:
    """Only valid camera BGR arrays enter the compositor."""
    with pytest.raises(ValueError):
        render_postcard(pixels, "en-US", 0)


def test_photo_timezone_defaults_toronto_and_validates_iana_names() -> None:
    """The private module option passes a real IANA timezone, never a host offset guess."""
    Arguments.prepareRecognitionArguments([])
    assert Arguments.get("christmasPhotoTimezone") == "America/Toronto"
    Arguments.prepareRecognitionArguments(["--christmasPhotoTimezone", "Europe/Paris"])
    assert Arguments.get("christmasPhotoTimezone") == "Europe/Paris"
    for invalid in ("Mars/NorthPole", "/etc/passwd", "../UTC"):
        with pytest.raises(SystemExit):
            Arguments.prepareRecognitionArguments(["--christmasPhotoTimezone", invalid])
