# SPDX-License-Identifier: Apache-2.0
"""Reproducible CPU detector benchmark using only a public-domain NASA image."""

from pathlib import Path
import argparse
import json
import statistics
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tools.utils.vision import FaceDetector


def scenarios(image: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
    """Generate bounded test scenes; these do not represent population accuracy."""
    portrait = cv2.resize(image, (500, 500))
    cases = [
        ("portrait", portrait, 1),
        ("dim_portrait", (portrait * 0.4).astype(np.uint8), 1),
    ]
    for angle in (-25, 25):
        matrix = cv2.getRotationMatrix2D((250, 250), angle, 1)
        cases.append(
            (f"tilted_{angle}", cv2.warpAffine(portrait, matrix, (500, 500)), 1)
        )
    face_crop = image[40:190, 155:295]
    for size in (48, 64, 96):
        canvas = np.zeros((281, 500, 3), dtype=np.uint8)
        crop = cv2.resize(face_crop, (size, size))
        canvas[80 : 80 + size, 180 : 180 + size] = crop
        cases.append((f"face_crop_{size}", canvas, 1))
    canvas = np.zeros((281, 500, 3), dtype=np.uint8)
    crop = cv2.resize(face_crop, (96, 96))
    canvas[80:176, 80:176] = crop
    canvas[80:176, 300:396] = crop
    cases.extend(
        [
            ("two_faces", canvas, 2),
            ("blank", np.zeros((281, 500, 3), dtype=np.uint8), 0),
        ]
    )
    return cases


def benchmark(fixture: Path, model: Path, repeats: int = 7) -> dict[str, Any]:
    """Compare detection time and expected face counts on identical images."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    image = cv2.imread(str(fixture))
    if image is None:
        raise ValueError("Public test fixture could not be read")
    output: dict[str, Any] = {
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "repeats": repeats,
        "scope": "Detection only; one public portrait and synthetic transforms, not identity accuracy",
        "yunet_score_threshold": 0.8,
        "detectors": {},
    }
    for method in ("dnn", "yunet"):
        detector = FaceDetector(method=method, yunet_model=model)
        rows = []
        for name, frame, expected in scenarios(image):
            detector.detect(frame)
            durations = []
            count = 0
            for _ in range(repeats):
                started = time.perf_counter()
                faces = detector.detect(frame)
                durations.append((time.perf_counter() - started) * 1000)
                count = len(faces)
            rows.append(
                {
                    "scene": name,
                    "expected_faces": expected,
                    "detected_faces": count,
                    "median_detection_ms": round(statistics.median(durations), 3),
                    "min_detection_ms": round(min(durations), 3),
                }
            )
        output["detectors"][method] = rows
    return output


def main() -> int:
    """Run the benchmark explicitly, with no camera or user gallery access."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture", type=Path, default=root / "tests/fixtures/astronaut.png"
    )
    parser.add_argument(
        "--model", type=Path, default=root / "model/face_detection_yunet_2023mar.onnx"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opencv-threads", type=int, default=0)
    args = parser.parse_args()
    if args.opencv_threads < 0:
        parser.error("opencv-threads must be nonnegative")
    if args.opencv_threads:
        cv2.setNumThreads(args.opencv_threads)
    result = json.dumps(benchmark(args.fixture, args.model, args.repeats), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
