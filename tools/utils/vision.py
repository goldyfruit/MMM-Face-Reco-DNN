"""Validated face detection and matching independent of camera ownership.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Detection:
    """A face box in dlib's top, right, bottom, left coordinate order."""

    box: Box
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RecognitionMatch:
    """The independently evaluated result for one face descriptor."""

    name: str
    distance: float | None
    accepted: bool
    ambiguous: bool = False


def clamp_box(box: Sequence[float], width: int, height: int) -> Box | None:
    """Clip a finite, nonempty face box to the image; discard invalid boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(box) != 4:
        return None
    try:
        top, right, bottom, left = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (top, right, bottom, left)):
        return None
    if right <= left or bottom <= top:
        return None
    clipped = (
        max(0, min(height, math.floor(top))),
        max(0, min(width, math.ceil(right))),
        max(0, min(height, math.ceil(bottom))),
        max(0, min(width, math.floor(left))),
    )
    if clipped[1] <= clipped[3] or clipped[2] <= clipped[0]:
        return None
    return clipped


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    """Validate and return contiguous BGR pixels from BGR or BGRX capture."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Face detection requires a uint8 NumPy image")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("Face detection requires BGR or BGRX colour channels")
    if not frame.shape[0] or not frame.shape[1]:
        raise ValueError("Face detection requires a nonempty image")
    return np.ascontiguousarray(frame[:, :, :3])


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Return contiguous uint8 RGB, including for Picamera2 BGRX buffers."""
    bgr = _validate_frame(frame)
    return np.ascontiguousarray(bgr[:, :, ::-1])


class FaceDetector:
    """Reuse one selected detector, keeping legacy dlib boxes compatible.

    YuNet only replaces detection. Its boxes still feed the existing dlib
    descriptor, so users do not need to rebuild their 128-element gallery.
    Dependencies can be injected for camera-free deterministic unit tests.
    """

    def __init__(
        self,
        method: str = "dnn",
        detection_method: str = "hog",
        cascade_path: str | Path | None = None,
        yunet_model: str | Path | None = None,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        upsample: int = 1,
        cv_module: Any | None = None,
        face_module: Any | None = None,
    ) -> None:
        """Validate configuration and initialise only the requested backend."""
        if method not in {"dnn", "haar", "yunet"}:
            raise ValueError("Detection method must be dnn, haar or yunet")
        if detection_method not in {"hog", "cnn"}:
            raise ValueError("dlib detection model must be hog or cnn")
        if isinstance(upsample, bool) or not isinstance(upsample, int) or upsample < 0:
            raise ValueError("Upsampling must be a nonnegative integer")
        for name, value in (("score", score_threshold), ("NMS", nms_threshold)):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} threshold must be greater than 0 and at most 1")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("YuNet top_k must be a positive integer")
        self.method = method
        self.detection_method = detection_method
        self.upsample = upsample
        self.score_threshold = score_threshold
        self._cv = cv_module
        self._face = face_module
        self._detector: Any | None = None
        if method == "dnn":
            if self._face is None:
                self._face = importlib.import_module("face_recognition")
        else:
            if self._cv is None:
                self._cv = importlib.import_module("cv2")
            if method == "haar":
                if cascade_path is None:
                    raise ValueError("Haar detection requires a cascade path")
                self._detector = self._cv.CascadeClassifier(str(cascade_path))
                if self._detector.empty():
                    raise ValueError("Could not load Haar cascade")
            else:
                if yunet_model is None or not Path(yunet_model).is_file():
                    raise FileNotFoundError("YuNet requires an existing ONNX model file")
                create = getattr(self._cv, "FaceDetectorYN_create", None)
                if create is None:
                    create = getattr(getattr(self._cv, "FaceDetectorYN", None), "create", None)
                if create is None:
                    raise RuntimeError("This OpenCV build does not provide FaceDetectorYN")
                self._detector = create(
                    str(yunet_model), "", (320, 320), score_threshold,
                    nms_threshold, top_k,
                )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect finite, bounded faces without mutating the caller's pixels."""
        bgr = _validate_frame(frame)
        height, width = bgr.shape[:2]
        detections: list[Detection] = []
        if self.method == "dnn":
            boxes = self._face.face_locations(
                to_rgb(bgr), number_of_times_to_upsample=self.upsample,
                model=self.detection_method,
            )
            for box in boxes:
                clipped = clamp_box(box, width, height)
                if clipped is not None:
                    detections.append(Detection(clipped))
        elif self.method == "haar":
            gray = self._cv.cvtColor(bgr, self._cv.COLOR_BGR2GRAY)
            rectangles = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                flags=self._cv.CASCADE_SCALE_IMAGE,
            )
            for x, y, w, h in rectangles:
                clipped = clamp_box((y, x + w, y + h, x), width, height)
                if clipped is not None:
                    detections.append(Detection(clipped))
        else:
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(bgr)
            if faces is None or np.asarray(faces).size == 0:
                return []
            faces = np.asarray(faces)
            if faces.ndim != 2 or faces.shape[1] < 15:
                raise RuntimeError("YuNet returned an invalid detection matrix")
            for face in faces:
                x, y, w, h = (float(value) for value in face[:4])
                score = float(face[14])
                if not math.isfinite(score) or not self.score_threshold <= score <= 1:
                    continue
                clipped = clamp_box((y, x + w, y + h, x), width, height)
                if clipped is not None:
                    detections.append(Detection(clipped, score))
        return detections


class FaceMatcher:
    """Cache validated descriptors and optionally reject ambiguous identities."""

    def __init__(
        self,
        encodings: Sequence[Sequence[float]] | np.ndarray,
        names: Sequence[str],
        tolerance: float = 0.6,
        identity_margin: float = 0.0,
    ) -> None:
        """Load the existing gallery without changing its descriptor format."""
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Matching tolerance must be finite and positive")
        if not math.isfinite(identity_margin) or identity_margin < 0:
            raise ValueError("Identity margin must be finite and nonnegative")
        matrix = np.asarray(encodings, dtype=np.float64)
        if matrix.size == 0 and len(names) == 0:
            matrix = np.empty((0, 128), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 128:
            raise ValueError("Known descriptors must have shape (count, 128)")
        if len(names) != len(matrix):
            raise ValueError("Known names and descriptor counts must agree")
        if not np.isfinite(matrix).all():
            raise ValueError("Known descriptors must contain finite numbers")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Known identities must be nonempty strings")
        self.encodings = np.array(matrix, copy=True, order="C")
        self.encodings.flags.writeable = False
        self.names = tuple(names)
        self.tolerance = float(tolerance)
        self.identity_margin = float(identity_margin)
        self._identity_indexes: dict[str, np.ndarray] = {
            name: np.flatnonzero(np.asarray(self.names) == name)
            for name in dict.fromkeys(self.names)
        }

    def match(self, encoding: Sequence[float] | np.ndarray) -> RecognitionMatch:
        """Return this face's score; optionally reject nearby other identities."""
        descriptor = np.asarray(encoding, dtype=np.float64)
        if descriptor.shape != (128,) or not np.isfinite(descriptor).all():
            raise ValueError("A face descriptor must contain 128 finite numbers")
        if not self.names:
            return RecognitionMatch("unknown", None, False)
        distances = np.linalg.norm(self.encodings - descriptor, axis=1)
        best_index = int(np.argmin(distances))
        distance = float(distances[best_index])
        name = self.names[best_index]
        ambiguous = False
        if self.identity_margin > 0:
            other = [
                float(np.min(distances[indexes]))
                for identity, indexes in self._identity_indexes.items()
                if identity != name
            ]
            ambiguous = bool(other and min(other) - distance < self.identity_margin)
        accepted = distance < self.tolerance and not ambiguous and name != "unknown"
        return RecognitionMatch(name if accepted else "unknown", distance, accepted, ambiguous)

    def match_many(self, encodings: Iterable[Sequence[float]]) -> list[RecognitionMatch]:
        """Evaluate each face independently, preserving result order."""
        return [self.match(encoding) for encoding in encodings]
