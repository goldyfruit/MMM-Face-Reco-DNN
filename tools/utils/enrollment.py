# SPDX-License-Identifier: Apache-2.0
"""Build a face gallery without labelling ambiguous photographs as one person."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Callable, Iterable


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass
class EnrollmentReport:
    """Aggregate gallery statistics without exposing identity names in logs."""

    total_images: int = 0
    enrolled_images: int = 0
    unreadable_images: int = 0
    no_face_images: int = 0
    multiple_face_images: int = 0
    invalid_encoding_images: int = 0
    unlabelled_images: int = 0
    output_written: bool = False

    def summary(self) -> str:
        """Describe accepted and rejected image counts for the command line."""
        return (
            f"enrolled={self.enrolled_images}, unreadable={self.unreadable_images}, "
            f"no_face={self.no_face_images}, multiple_faces={self.multiple_face_images}, "
            f"invalid_encoding={self.invalid_encoding_images}, "
            f"unlabelled={self.unlabelled_images}, total={self.total_images}"
        )


def image_paths(dataset: Path) -> Iterable[Path]:
    """Yield supported images deterministically beneath the dataset directory."""
    return sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def valid_encoding(encoding: Any) -> bool:
    """Require a finite one-dimensional dlib descriptor with 128 components."""
    shape = getattr(encoding, "shape", None)
    if shape is not None and tuple(shape) != (128,):
        return False
    try:
        return len(encoding) == 128 and all(
            math.isfinite(float(value)) for value in encoding
        )
    except (TypeError, ValueError, OverflowError):
        return False


def write_gallery_atomic(output: Path, gallery: dict[str, list[Any]]) -> None:
    """Replace the gallery only after a complete private temporary file is written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".encodings-",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            pickle.dump(gallery, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def encode_dataset(
    dataset: str | Path,
    output: str | Path,
    detection_method: str = "hog",
    *,
    cv_module: Any = None,
    face_module: Any = None,
    report: Callable[[str], None] = print,
) -> EnrollmentReport:
    """Enroll exactly one face per image, retaining the existing pickle format.

    A person's label is the first directory below ``dataset``. Images directly
    inside the dataset have no label and are skipped. Descendant directories
    may organise a person's photographs without becoming new identities.
    Multi-face photographs are deliberately skipped: assigning every face to
    the folder's label silently corrupts recognition accuracy.

    Empty or entirely rejected datasets leave an existing gallery untouched.
    Native dependencies are imported lazily so this function can be unit tested
    without a camera, dlib installation, or any private face photographs.
    """
    dataset = Path(dataset)
    output = Path(output)
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    if detection_method not in {"hog", "cnn"}:
        raise ValueError("Encoding detection method must be 'hog' or 'cnn'")
    if cv_module is None:
        import cv2 as cv_module
    if face_module is None:
        import face_recognition as face_module

    paths = list(image_paths(dataset))
    statistics = EnrollmentReport(total_images=len(paths))
    encodings: list[Any] = []
    names: list[str] = []
    image_error = getattr(cv_module, "error", ValueError)
    for index, path in enumerate(paths, start=1):
        report(f"[INFO] processing image {index}/{len(paths)}")
        relative = path.relative_to(dataset)
        if len(relative.parts) < 2:
            statistics.unlabelled_images += 1
            continue
        try:
            image = cv_module.imread(str(path))
            if image is None:
                statistics.unreadable_images += 1
                continue
            rgb = cv_module.cvtColor(image, cv_module.COLOR_BGR2RGB)
        except (OSError, ValueError, TypeError, image_error):
            statistics.unreadable_images += 1
            continue

        boxes = face_module.face_locations(rgb, model=detection_method)
        if not boxes:
            statistics.no_face_images += 1
            continue
        if len(boxes) != 1:
            statistics.multiple_face_images += 1
            continue
        descriptors = face_module.face_encodings(rgb, boxes)
        if len(descriptors) != 1 or not valid_encoding(descriptors[0]):
            statistics.invalid_encoding_images += 1
            continue
        encodings.append(descriptors[0])
        names.append(relative.parts[0])
        statistics.enrolled_images += 1

    report("[INFO] " + statistics.summary())
    if not encodings:
        report(
            "[ERROR] No unambiguous valid faces enrolled; existing gallery unchanged."
        )
        return statistics
    write_gallery_atomic(output, {"encodings": encodings, "names": names})
    statistics.output_written = True
    report("[INFO] Gallery saved atomically.")
    return statistics
