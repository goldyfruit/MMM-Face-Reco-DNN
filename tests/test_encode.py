# SPDX-License-Identifier: Apache-2.0
"""Exercise enrollment safety with synthetic descriptors and mocked images."""

from pathlib import Path
import importlib.util
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.utils.enrollment import encode_dataset, valid_encoding, write_gallery_atomic


class FakeCV:
    """Provide image loading and conversion without native OpenCV."""

    COLOR_BGR2RGB = 1
    error = ValueError

    def __init__(self, unreadable: set[str] | None = None) -> None:
        """Identify files that should simulate unreadable photographs."""
        self.unreadable = unreadable or set()

    def imread(self, path: str) -> str | None:
        """Use a file name as the stand-in for an image array."""
        return None if Path(path).name in self.unreadable else Path(path).name

    def cvtColor(self, image: str, code: int) -> str:
        """Keep the stand-in while asserting the correct color conversion."""
        assert code == self.COLOR_BGR2RGB
        return image


class FakeFaces:
    """Return controlled face counts and descriptors for each synthetic image."""

    def __init__(self, counts: dict[str, int], invalid: set[str] | None = None) -> None:
        """Store per-file outcomes and all embedding requests."""
        self.counts = counts
        self.invalid = invalid or set()
        self.encoding_calls: list[str] = []

    def face_locations(
        self, image: str, *, model: str
    ) -> list[tuple[int, int, int, int]]:
        """Produce boxes for the selected detection model."""
        assert model in {"hog", "cnn"}
        return [(1, 10, 10, 1)] * self.counts.get(image, 1)

    def face_encodings(
        self, image: str, boxes: list[tuple[int, int, int, int]]
    ) -> list[list[float]]:
        """Produce invalid or finite synthetic descriptor vectors."""
        self.encoding_calls.append(image)
        return [[float("nan")] * 128] if image in self.invalid else [[0.25] * 128]


def image(dataset: Path, relative: str) -> Path:
    """Create a placeholder that the fake loader treats as a photograph."""
    path = dataset / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic placeholder; not an actual photo")
    return path


def test_one_face_per_image_preserves_gallery_format(tmp_path: Path) -> None:
    """Never assign the label of a multi-face photograph to every detected face."""
    dataset = tmp_path / "dataset"
    for relative in ("person-a/one.jpg", "person-a/group.jpg", "person-b/two.PNG"):
        image(dataset, relative)
    faces = FakeFaces({"group.jpg": 2})
    output = tmp_path / "model/encodings.pickle"
    result = encode_dataset(
        dataset, output, cv_module=FakeCV(), face_module=faces, report=lambda _: None
    )
    assert result.enrolled_images == 2
    assert result.multiple_face_images == 1
    assert "group.jpg" not in faces.encoding_calls
    with output.open("rb") as handle:
        saved = pickle.load(handle)
    assert set(saved) == {"encodings", "names"}
    assert saved["names"] == ["person-a", "person-b"]
    assert all(len(encoding) == 128 for encoding in saved["encodings"])


def test_rejected_images_do_not_erase_existing_gallery(tmp_path: Path) -> None:
    """Unusable input leaves the last valid serialized model unchanged."""
    dataset = tmp_path / "dataset"
    for relative in (
        "person/broken.jpg",
        "person/empty.jpg",
        "person/group.jpg",
        "person/invalid.jpg",
        "unlabelled.jpg",
    ):
        image(dataset, relative)
    output = tmp_path / "encodings.pickle"
    original = b"previous valid gallery stands in for private existing data"
    output.write_bytes(original)
    faces = FakeFaces({"empty.jpg": 0, "group.jpg": 2}, invalid={"invalid.jpg"})
    result = encode_dataset(
        dataset,
        output,
        cv_module=FakeCV({"broken.jpg"}),
        face_module=faces,
        report=lambda _: None,
    )
    assert not result.output_written
    assert (
        result.unreadable_images
        == result.no_face_images
        == result.multiple_face_images
        == 1
    )
    assert result.invalid_encoding_images == result.unlabelled_images == 1
    assert output.read_bytes() == original


def test_nested_photographs_keep_top_level_identity(tmp_path: Path) -> None:
    """Organising one person's images by lighting does not invent new labels."""
    dataset = tmp_path / "dataset"
    image(dataset, "person-a/daylight/portrait.jpg")
    output = tmp_path / "encodings.pickle"
    encode_dataset(
        dataset,
        output,
        cv_module=FakeCV(),
        face_module=FakeFaces({}),
        report=lambda _: None,
    )
    with output.open("rb") as handle:
        assert pickle.load(handle)["names"] == ["person-a"]


@pytest.mark.parametrize(
    "encoding",
    [[], [0.0] * 127, [float("nan")] * 128, [float("inf")] * 128, [[0.0]] * 128, None],
)
def test_invalid_descriptors_are_rejected(encoding: object) -> None:
    """Reject malformed numeric input instead of saving a broken gallery."""
    assert not valid_encoding(encoding)


def test_zero_valid_images_create_no_output(tmp_path: Path) -> None:
    """An empty dataset is an explicit failed enrollment, not an empty model."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "encodings.pickle"
    result = encode_dataset(
        dataset,
        output,
        cv_module=FakeCV(),
        face_module=FakeFaces({}),
        report=lambda _: None,
    )
    assert not result.output_written and result.total_images == 0
    assert not output.exists()


def test_atomic_serialization_failure_keeps_old_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing serialization cannot truncate the existing face model."""
    output = tmp_path / "encodings.pickle"
    output.write_bytes(b"last good model")

    def fail(*args: object, **kwargs: object) -> None:
        """Simulate an interrupted or failed serializer."""
        raise OSError("simulated serialization failure")

    monkeypatch.setattr("tools.utils.enrollment.pickle.dump", fail)
    with pytest.raises(OSError, match="simulated"):
        write_gallery_atomic(
            output, {"encodings": [[0.0] * 128], "names": ["person-a"]}
        )
    assert output.read_bytes() == b"last good model"
    assert list(tmp_path.glob(".encodings-*.tmp")) == []


def test_atomic_replace_failure_keeps_old_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing final rename leaves the old model and cleans up the temporary."""
    output = tmp_path / "encodings.pickle"
    output.write_bytes(b"last good model")

    def fail(*args: object, **kwargs: object) -> None:
        """Simulate a rejected atomic replace."""
        raise OSError("simulated replace failure")

    monkeypatch.setattr("tools.utils.enrollment.os.replace", fail)
    with pytest.raises(OSError, match="replace"):
        write_gallery_atomic(
            output, {"encodings": [[0.0] * 128], "names": ["person-a"]}
        )
    assert output.read_bytes() == b"last good model"
    assert list(tmp_path.glob(".encodings-*.tmp")) == []


def test_dataset_and_method_are_validated(tmp_path: Path) -> None:
    """Reject misconfiguration before importing optional native dependencies."""
    with pytest.raises(ValueError, match="directory"):
        encode_dataset(tmp_path / "missing", tmp_path / "out")
    with pytest.raises(ValueError, match="method"):
        encode_dataset(tmp_path, tmp_path / "out", detection_method="unknown")


def test_failed_color_conversion_is_reported_as_unreadable(tmp_path: Path) -> None:
    """A corrupt decoder result does not crash or erase an existing gallery."""
    dataset = tmp_path / "dataset"
    image(dataset, "person-a/corrupt.jpg")
    cv = FakeCV()

    def fail(image: str, code: int) -> str:
        """Simulate OpenCV rejecting corrupt image pixels."""
        raise ValueError("invalid image data")

    cv.cvtColor = fail
    output = tmp_path / "encodings.pickle"
    output.write_bytes(b"last good model")
    result = encode_dataset(
        dataset, output, cv_module=cv, face_module=FakeFaces({}), report=lambda _: None
    )
    assert result.unreadable_images == 1 and not result.output_written
    assert output.read_bytes() == b"last good model"


@pytest.fixture
def cli_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the CLI without parsing arguments or requiring shared legacy imports."""
    values = {
        "dataset": "dataset",
        "encodings": "model.pickle",
        "detection_method": "hog",
    }
    calls: list[str] = []
    arguments = SimpleNamespace(
        prepareEncodingArguments=lambda: calls.append("parse"),
        get=lambda name: values[name],
    )
    monkeypatch.setitem(
        sys.modules, "tools.utils.arguments", SimpleNamespace(Arguments=arguments)
    )
    path = Path(__file__).resolve().parents[1] / "tools/encode.py"
    spec = importlib.util.spec_from_file_location("tools.encode_test_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert calls == []
    return module


@pytest.mark.parametrize("saved, expected", [(True, 0), (False, 1)])
def test_cli_reports_empty_or_successful_enrollment(
    cli_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    saved: bool,
    expected: int,
) -> None:
    """The command exits unsuccessfully when no new gallery was written."""

    def enroll(**kwargs: object) -> object:
        """Validate the existing CLI fields reach the safe enrollment builder."""
        assert kwargs == {
            "dataset": "dataset",
            "output": "model.pickle",
            "detection_method": "hog",
        }
        return SimpleNamespace(output_written=saved)

    monkeypatch.setattr(cli_module, "encode_dataset", enroll)
    assert cli_module.main() == expected


def test_cli_reports_output_failure_without_traceback(
    cli_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected filesystem errors produce an actionable nonzero CLI result."""

    def fail(**kwargs: object) -> object:
        """Simulate a permission error while preserving the old model."""
        raise OSError("read-only output directory")

    monkeypatch.setattr(cli_module, "encode_dataset", fail)
    assert cli_module.main() == 1
    assert "read-only output directory" in capsys.readouterr().err
