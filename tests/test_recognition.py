# SPDX-License-Identifier: Apache-2.0
"""Regression tests for identity isolation, safe enrollment and worker cleanup."""

import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tools import recognition
from tools.utils.arguments import Arguments
from tools.utils.print import Print
from tools.utils.vision import Detection, FaceMatcher, RecognitionMatch


def test_empty_detection_does_not_compute_descriptors() -> None:
    """Avoid an expensive encoder call when the detector found no faces."""
    face = SimpleNamespace(face_encodings=Mock())
    engine = recognition.RecognitionEngine(
        SimpleNamespace(detect=lambda _: []), FaceMatcher([], []), face
    )
    assert engine.recognize(np.zeros((20, 20, 3), dtype=np.uint8)).names == []
    face.face_encodings.assert_not_called()


def test_two_faces_keep_independent_scores_and_rgb_buffer() -> None:
    """Prevent the last-distance label bug and maintain contiguous RGB input."""
    gallery = np.array([np.zeros(128), np.ones(128)])
    face = SimpleNamespace(
        face_encodings=Mock(return_value=[np.zeros(128), np.ones(128) + 0.01])
    )
    detector = SimpleNamespace(
        detect=lambda _: [Detection((0, 10, 10, 0)), Detection((10, 20, 20, 10))]
    )
    engine = recognition.RecognitionEngine(
        detector, FaceMatcher(gallery, ["first", "second"]), face
    )
    image = np.zeros((20, 20, 4), dtype=np.uint8)
    image[:, :, 0] = 30
    result = engine.recognize(image)
    assert result.names == ["first", "second"]
    assert result.matches[0].distance == 0
    assert result.matches[1].distance == pytest.approx(np.sqrt(128) * 0.01)
    rgb = face.face_encodings.call_args.args[0]
    assert rgb.flags.c_contiguous and rgb.shape == (20, 20, 3)
    assert rgb[0, 0].tolist() == [0, 0, 30]


def test_mismatched_descriptors_fail_instead_of_mislabeling() -> None:
    """Do not silently zip the wrong faces to identity results."""
    engine = recognition.RecognitionEngine(
        SimpleNamespace(detect=lambda _: [Detection((0, 10, 10, 0))]),
        FaceMatcher([], []),
        SimpleNamespace(face_encodings=lambda *a, **k: []),
    )
    with pytest.raises(RuntimeError, match="Descriptor count"):
        engine.recognize(np.zeros((20, 20, 3), dtype=np.uint8))


def test_annotations_are_separate_and_use_per_face_distance() -> None:
    """Protect full-resolution training pixels from preview rectangles/text."""
    texts = []
    cv = SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        rectangle=lambda a, *args: a.fill(255),
        putText=lambda a, text, *args: texts.append(text),
    )
    result = recognition.FrameResult(
        [Detection((0, 5, 5, 0)), Detection((5, 10, 10, 5))],
        [RecognitionMatch("first", 0.11, True), RecognitionMatch("second", 0.32, True)],
    )
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    preview = recognition.annotate(original, result, cv)
    assert texts == ["first (0.11)", "second (0.32)"]
    assert not original.any() and preview.all()


def test_dataset_extension_fetches_main_only_on_single_face_login(tmp_path) -> None:
    """Lazy main reads preserve resolution without copying every frame."""
    frame = np.ones((108, 192, 3), dtype=np.uint8)
    fetch = Mock(return_value=frame)
    writer = Mock(return_value=True)
    result = recognition.FrameResult(
        [Detection((1, 10, 10, 1))], [RecognitionMatch("person", 0.2, True)]
    )
    assert recognition.extend_dataset(
        str(tmp_path), result, ["person"], fetch, SimpleNamespace(imwrite=writer)
    )
    fetch.assert_called_once()
    assert writer.call_args.args[1] is frame
    assert not recognition.extend_dataset(
        str(tmp_path), result, [], fetch, SimpleNamespace(imwrite=writer)
    )
    assert fetch.call_count == 1


def test_dataset_extension_skips_group_images(tmp_path) -> None:
    """A group frame must not train every visible face as one identity."""
    fetch = Mock()
    result = recognition.FrameResult(
        [Detection((1, 10, 10, 1))] * 2, [RecognitionMatch("person", 0.2, True)] * 2
    )
    assert not recognition.extend_dataset(str(tmp_path), result, ["person"], fetch)
    assert result.names == ["person"]
    fetch.assert_not_called()


@pytest.mark.parametrize("name", ["../escape", "/outside", ".", ""])
def test_dataset_labels_cannot_escape_the_dataset(tmp_path, name: str) -> None:
    """Reject path-like labels without reading or writing outside the dataset."""
    result = recognition.FrameResult(
        [Detection((1, 10, 10, 1))], [RecognitionMatch(name, 0.2, True)]
    )
    with pytest.raises(ValueError, match="safe directory"):
        recognition.extend_dataset(str(tmp_path), result, [name], Mock())


def test_dataset_write_failure_is_visible(tmp_path) -> None:
    """Do not report successful extension if the SD write failed."""
    result = recognition.FrameResult(
        [Detection((1, 10, 10, 1))], [RecognitionMatch("unknown", 0.8, False)]
    )
    with pytest.raises(RuntimeError, match="save dataset"):
        recognition.extend_dataset(
            str(tmp_path),
            result,
            ["unknown"],
            lambda: np.zeros((3, 3, 3)),
            SimpleNamespace(imwrite=lambda *a: False),
        )


def test_parent_eof_and_exact_commands_stop_worker() -> None:
    """A closed parent or an explicit exit cannot leave the camera orphaned."""
    active, closed = threading.Event(), threading.Event()
    recognition.monitor_commands(io.StringIO("restart\n"), active, closed)
    assert not active.is_set() and closed.is_set()
    active, closed = threading.Event(), threading.Event()
    recognition.monitor_commands(
        io.StringIO("stop\nstart\nexit\nstop\n"), active, closed
    )
    assert active.is_set() and closed.is_set()


@pytest.mark.parametrize(
    "flag,value",
    [("--interval", "0"), ("--previewQuality", "101"), ("--tolerance", "-1")],
)
def test_invalid_runtime_values_fail_before_camera_access(
    flag: str, value: str
) -> None:
    """Reject unsafe timing or malformed image settings at startup."""
    with pytest.raises(SystemExit):
        Arguments.prepareRecognitionArguments([flag, value])


def test_new_arguments_keep_legacy_defaults() -> None:
    """Existing users opt in to YuNet/ISP changes rather than switching silently."""
    Arguments.prepareRecognitionArguments([])
    assert Arguments.get("method") == "dnn" and Arguments.get("cameraLores") is False
    assert Arguments.get("cameraFrameRate") is None
    assert Arguments.get("detectionUpsample") == 1


def test_json_messages_do_not_interleave(capsys) -> None:
    """Concurrent events remain valid JSON lines for PythonShell."""
    threads = [
        threading.Thread(target=Print.printJson, args=("event", i)) for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    values = [
        json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()
    ]
    assert sorted(values) == list(range(20))
