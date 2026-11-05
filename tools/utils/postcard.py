# SPDX-License-Identifier: Apache-2.0
"""Compose an explicitly requested North Pole keepsake with OpenCV alone."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

CREAM = (232, 243, 249)
GREEN = (66, 86, 48)
GOLD = (101, 163, 185)


def _text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    *,
    serif: bool = False,
) -> None:
    """Draw fixed authored lettering without loading a font or an external asset."""
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_COMPLEX if serif else cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        GREEN,
        max(1, round(scale * 1.5)),
        cv2.LINE_AA,
    )


def _stamp(canvas: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Draw an original pine, polar star and circular postal seal as vector lines."""
    x, y = center
    cv2.circle(canvas, center, radius, GOLD, 1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius - 5, GOLD, 1, cv2.LINE_AA)
    # A tiny evergreen and trunk sit below an eight-point northern star.
    for offset, spread in ((-4, 11), (6, 17), (17, 23)):
        points = np.array(
            [
                (x, y + round((offset - 21) * radius / 50)),
                (x - round(spread * radius / 50), y + round(offset * radius / 50)),
                (x + round(spread * radius / 50), y + round(offset * radius / 50)),
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], True, GREEN, 1, cv2.LINE_AA)
    cv2.line(
        canvas,
        (x, y + round(18 * radius / 50)),
        (x, y + round(27 * radius / 50)),
        GREEN,
        1,
        cv2.LINE_AA,
    )
    star_y, length = y - round(31 * radius / 50), max(3, round(5 * radius / 50))
    for dx, dy in (
        (length, 0),
        (0, length),
        (length // 2, length // 2),
        (length // 2, -length // 2),
    ):
        cv2.line(
            canvas, (x - dx, star_y - dy), (x + dx, star_y + dy), GOLD, 1, cv2.LINE_AA
        )
    # Three short postal waves keep the seal outside the actual family image.
    for offset in (-8, 0, 8):
        cv2.ellipse(
            canvas,
            (x - radius - 17, y + offset),
            (20, 3),
            0,
            0,
            180,
            GOLD,
            1,
            cv2.LINE_AA,
        )


def render_postcard(
    pixels: np.ndarray,
    lang: str,
    captured_at: float,
    timezone: str = "America/Toronto",
) -> np.ndarray:
    """Frame a full uncropped photo and freeze its trusted local capture date.

    The returned BGR image owns its pixels. The camera frame is never modified;
    all lettering and artwork sit on an added mat, outside the family photo.
    The complete result stays within the existing 1200-pixel preview bound.
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or min(pixels.shape[:2]) < 1
    ):
        raise ValueError("A nonempty uint8 BGR photo is required")
    local = datetime.fromtimestamp(captured_at, ZoneInfo(timezone))
    height, width = pixels.shape[:2]
    scale = min(1080 / width, 850 / height, 1)
    size = max(1, round(width * scale)), max(1, round(height * scale))
    photo = (
        cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
        if size != (width, height)
        else pixels
    )
    width, height = size
    canvas_width, canvas_height = max(640, width + 80), height + 200
    canvas = np.full((canvas_height, canvas_width, 3), CREAM, np.uint8)
    left, top = (canvas_width - width) // 2, 36
    canvas[top : top + height, left : left + width] = photo
    cv2.rectangle(canvas, (12, 12), (canvas_width - 13, canvas_height - 13), GOLD, 1)
    cv2.rectangle(canvas, (18, 18), (canvas_width - 19, canvas_height - 19), GREEN, 1)
    cv2.rectangle(
        canvas, (left - 2, top - 2), (left + width + 1, top + height + 1), GOLD, 1
    )
    french = lang.lower().startswith("fr")
    title, subtitle = (
        ("POLE NORD", "UN MOMENT EN FAMILLE")
        if french
        else ("NORTH POLE", "A MOMENT TO KEEP")
    )
    title_scale = min(1.2, canvas_width / 850)
    title_origin = (42, top + height + 57)
    _text(canvas, title, title_origin, title_scale, serif=True)
    if french:
        # Hershey is ASCII-only. Draw the actual uppercase Ô accent ourselves,
        # aligned with its O glyph, instead of substituting a question mark.
        font = cv2.FONT_HERSHEY_COMPLEX
        thickness = max(1, round(title_scale * 1.5))
        prefix_width = cv2.getTextSize("P", font, title_scale, thickness)[0][0]
        (o_width, title_height), _ = cv2.getTextSize("O", font, title_scale, thickness)
        accent_x = title_origin[0] + prefix_width - thickness + o_width // 2
        accent_y = title_origin[1] - title_height - 4
        spread = max(3, round(5 * title_scale))
        cv2.polylines(
            canvas,
            [
                np.array(
                    [
                        (accent_x - spread, accent_y),
                        (accent_x, accent_y - spread),
                        (accent_x + spread, accent_y),
                    ],
                    np.int32,
                )
            ],
            False,
            GREEN,
            thickness,
            cv2.LINE_AA,
        )
    _text(canvas, subtitle, (43, top + height + 87), 0.42)
    date = local.strftime("%d / %m / %Y") if french else local.strftime("%Y / %m / %d")
    _text(canvas, date, (43, top + height + 120), 0.48)
    _stamp(canvas, (canvas_width - 102, canvas_height - 85), 50)
    return canvas
