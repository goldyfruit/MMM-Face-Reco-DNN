"""Camera-free regression tests for detector geometry and identity decisions.

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.utils.vision import Detection, FaceDetector, FaceMatcher, clamp_box, to_rgb


@pytest.mark.parametrize(
    "box, expected",
    [
        ((-2, 15, 14, -4), (0, 10, 10, 0)),
        ((1.2, 8.2, 9.1, 2.8), (1, 9, 10, 2)),
        ((0, 10, 10, 0), (0, 10, 10, 0)),
        ((0, 2, 5, 3), None),
        ((1, 8, 1, 3), None),
        ((11, 20, 18, 12), None),
        ((0, float("nan"), 2, 0), None),
        ((0, float("inf"), 2, 0), None),
        ((0, 2, 3), None),
    ],
)
def test_boxes_are_clipped_or_discarded(box: tuple, expected: tuple | None) -> None:
    """Never feed negative, empty or nonfinite rectangles to an encoder."""
    assert clamp_box(box, 10, 10) == expected


def test_rgb_is_contiguous_and_handles_picamera_bgrx() -> None:
    """Check channel order and dlib's memory requirement on strided capture."""
    capture = np.zeros((3, 8, 4), dtype=np.uint8)
    capture[:] = [11, 22, 33, 255]
    rgb = to_rgb(capture[:, ::2])
    assert rgb.shape == (3, 4, 3)
    assert rgb.flags.c_contiguous
    np.testing.assert_array_equal(rgb[0, 0], [33, 22, 11])
    np.testing.assert_array_equal(capture[0, 0], [11, 22, 33, 255])


@pytest.mark.parametrize(
    "frame", [np.zeros((3, 3)), np.zeros((3, 3, 3), dtype=float),
              np.zeros((0, 3, 3), dtype=np.uint8), np.zeros((3, 3, 2), dtype=np.uint8)],
)
def test_invalid_frames_fail_clearly(frame: np.ndarray) -> None:
    """Reject unsafe implicit type/channel coercion."""
    with pytest.raises(ValueError):
        to_rgb(frame)


def test_legacy_detection_preserves_model_and_upsampling() -> None:
    """Keep the existing HOG/CNN path and its configured semantics."""
    calls = []

    def locations(frame: np.ndarray, **kwargs: object) -> list[tuple]:
        calls.append((frame.copy(), kwargs))
        return [(-1, 8, 4, -1), (1, 2, 1, 0)]

    detector = FaceDetector(detection_method="cnn", upsample=2,
                            face_module=SimpleNamespace(face_locations=locations))
    assert detector.detect(np.zeros((5, 6, 4), dtype=np.uint8)) == [Detection((0, 6, 4, 0))]
    assert calls[0][0].shape == (5, 6, 3)
    assert calls[0][1] == {"number_of_times_to_upsample": 2, "model": "cnn"}


def test_yunet_network_is_reused_and_size_tracks_frame(tmp_path) -> None:
    """Avoid loading a model for every frame and preserve safe confidence boxes."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test network injected below")
    sizes = []
    creates = []
    rows = np.zeros((4, 15), dtype=np.float32)
    rows[0, :4], rows[0, 14] = [-2, 1, 10, 12], .95
    rows[1, :4], rows[1, 14] = [2, 2, 2, 2], .5
    rows[2, :4], rows[2, 14] = [2, 2, -1, 2], .99
    rows[3, :4], rows[3, 14] = [2, 2, 1, 2], np.nan
    network = SimpleNamespace(setInputSize=sizes.append, detect=lambda _: (1, rows))

    def create(*args: object) -> object:
        creates.append(args)
        return network

    detector = FaceDetector(method="yunet", yunet_model=model,
                            cv_module=SimpleNamespace(FaceDetectorYN_create=create))
    first = detector.detect(np.zeros((6, 8, 3), dtype=np.uint8))
    assert len(first) == 1
    assert first[0].box == (1, 8, 6, 0)
    assert first[0].confidence == pytest.approx(.95)
    detector.detect(np.zeros((9, 7, 3), dtype=np.uint8))
    assert sizes == [(8, 6), (7, 9)]
    assert len(creates) == 1


def test_yunet_missing_model_does_not_silently_switch_detector() -> None:
    """Make optional-backend configuration errors visible."""
    with pytest.raises(FileNotFoundError):
        FaceDetector(method="yunet", yunet_model="/missing/model.onnx", cv_module=object())


def test_bundled_yunet_detects_public_portrait_and_rejects_blank_frame() -> None:
    """Exercise the actual network on a public fixture, not a private gallery."""
    import cv2

    root = Path(__file__).resolve().parents[1]
    portrait = cv2.imread(str(root / "tests/fixtures/astronaut.png"))
    assert portrait is not None
    detector = FaceDetector(method="yunet", yunet_model=root / "model/face_detection_yunet_2023mar.onnx")
    results = detector.detect(portrait)
    assert len(results) == 1
    top, right, bottom, left = results[0].box
    # Broad manually defined face region; avoid encoding one backend's exact box.
    expected = (55, 280, 185, 165)
    overlap = max(0, min(right, expected[1]) - max(left, expected[3])) * max(
        0, min(bottom, expected[2]) - max(top, expected[0])
    )
    assert overlap / ((right - left) * (bottom - top)) > .9
    assert results[0].confidence >= .9
    assert detector.detect(np.zeros_like(portrait)) == []


def test_haar_cascade_is_loaded_only_for_haar() -> None:
    """Unused legacy models must not be required by another backend."""
    seen = []
    cascade = SimpleNamespace(empty=lambda: False,
                              detectMultiScale=lambda *a, **k: [(1, 2, 4, 3)])
    cv = SimpleNamespace(CascadeClassifier=lambda p: seen.append(p) or cascade,
                         cvtColor=lambda f, _: f[:, :, 0], COLOR_BGR2GRAY=7,
                         CASCADE_SCALE_IMAGE=8)
    detector = FaceDetector(method="haar", cascade_path="cascade.xml", cv_module=cv)
    assert detector.detect(np.zeros((8, 9, 3), dtype=np.uint8)) == [Detection((2, 5, 5, 1))]
    assert seen == ["cascade.xml"]


def test_empty_gallery_is_unknown() -> None:
    """A new installation without enrolled people is valid."""
    assert FaceMatcher([], []).match(np.zeros(128)).name == "unknown"
    assert FaceMatcher([], []).match(np.zeros(128)).distance is None


def test_each_face_keeps_its_own_matching_distance() -> None:
    """Prevent the last face's score from leaking into other faces' results."""
    first = np.zeros(128)
    second = np.zeros(128)
    second[0] = 1
    matcher = FaceMatcher([first, second], ["alice", "bob"])
    queries = np.zeros((2, 128))
    queries[:, 0] = [.1, .8]
    results = matcher.match_many(queries)
    assert [r.name for r in results] == ["alice", "bob"]
    assert results[0].distance == pytest.approx(.1)
    assert results[1].distance == pytest.approx(.2)


def test_margin_compares_distinct_identities_not_same_person_templates() -> None:
    """A person's near-duplicate gallery images are not a competing identity."""
    matrix = np.zeros((3, 128))
    matrix[:, 0] = [.10, .11, .40]
    result = FaceMatcher(matrix, ["alice", "alice", "bob"], identity_margin=.05).match(np.zeros(128))
    assert result.accepted and result.name == "alice"
    matrix[2, 0] = .12
    result = FaceMatcher(matrix, ["alice", "alice", "bob"], identity_margin=.05).match(np.zeros(128))
    assert not result.accepted and result.ambiguous and result.name == "unknown"


def test_default_margin_and_threshold_remain_legacy_compatible() -> None:
    """Preserve strict tolerance and first-template tie behaviour by default."""
    gallery = np.zeros((2, 128))
    gallery[:, 0] = .2
    assert FaceMatcher(gallery, ["first", "second"]).match(np.zeros(128)).name == "first"
    result = FaceMatcher(gallery, ["first", "second"], tolerance=.2).match(np.zeros(128))
    assert not result.accepted


def test_gallery_is_copied_so_callers_cannot_invalidate_cache() -> None:
    """Keep stable vectors even if an external loader reuses its array."""
    gallery = np.zeros((1, 128))
    matcher = FaceMatcher(gallery, ["alice"])
    gallery[:] = 100
    assert matcher.match(np.zeros(128)).name == "alice"
    assert not matcher.encodings.flags.writeable


@pytest.mark.parametrize(
    "vectors,names", [([[0] * 127], ["alice"]), ([[0] * 128], []),
                      ([[np.nan] * 128], ["alice"]), ([[0] * 128], [""])],
)
def test_invalid_gallery_is_rejected(vectors: list, names: list) -> None:
    """Fail on corrupt enrollment instead of issuing unreliable identities."""
    with pytest.raises(ValueError):
        FaceMatcher(vectors, names)


@pytest.mark.parametrize("query", [np.zeros(127), np.full(128, np.inf)])
def test_invalid_query_is_rejected(query: np.ndarray) -> None:
    """Require correctly shaped finite embeddings from the encoder."""
    with pytest.raises(ValueError):
        FaceMatcher([np.zeros(128)], ["alice"]).match(query)
