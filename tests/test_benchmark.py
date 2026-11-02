# SPDX-License-Identifier: Apache-2.0
"""Check reproducibility and boundaries of the public-fixture benchmark."""

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from tools import benchmark


def test_scenes_include_positive_negative_and_multiple_faces() -> None:
    """Keep simple negative and group scenes, not just easy face portraits."""
    scenes = benchmark.scenarios(np.zeros((512, 512, 3), dtype=np.uint8))
    assert any(count == 0 for _, _, count in scenes)
    assert any(count == 2 for _, _, count in scenes)
    assert {name for name, _, _ in scenes} >= {
        "dim_portrait",
        "tilted_25",
        "face_crop_48",
    }


def test_benchmark_reports_both_detectors_on_the_same_scenes(monkeypatch) -> None:
    """Measure equivalent inputs with injected detections, without dlib."""
    calls = []

    class Detector:
        """Remember backend and image dimensions in a bounded fake detector."""

        def __init__(self, **kwargs) -> None:
            self.method = kwargs["method"]

        def detect(self, frame: np.ndarray) -> list:
            calls.append((self.method, frame.shape))
            return []

    monkeypatch.setattr(benchmark, "FaceDetector", Detector)
    fixture = Path(__file__).parent / "fixtures/astronaut.png"
    result = benchmark.benchmark(fixture, Path("unused"), repeats=1)
    left, right = result["detectors"]["dnn"], result["detectors"]["yunet"]
    assert [x["scene"] for x in left] == [x["scene"] for x in right]
    assert all(x["median_detection_ms"] >= 0 for x in left + right)
    assert [shape for method, shape in calls if method == "dnn"] == [
        shape for method, shape in calls if method == "yunet"
    ]


def test_invalid_benchmark_inputs_fail_clearly(tmp_path) -> None:
    """Reject impossible repetitions and unavailable fixtures."""
    with pytest.raises(ValueError, match="positive"):
        benchmark.benchmark(tmp_path / "missing", tmp_path / "model", repeats=0)
    with pytest.raises(ValueError, match="fixture"):
        benchmark.benchmark(tmp_path / "missing", tmp_path / "model")


@pytest.mark.parametrize("threads", [0, 2])
def test_benchmark_cli_thread_selection(monkeypatch, capsys, threads: int) -> None:
    """Only an explicit positive benchmark setting changes OpenCV threads."""
    from unittest.mock import Mock

    setter = Mock()
    monkeypatch.setattr(benchmark.cv2, "setNumThreads", setter)
    monkeypatch.setattr(benchmark, "benchmark", lambda *args: {})
    monkeypatch.setattr("sys.argv", ["benchmark", "--opencv-threads", str(threads)])
    assert benchmark.main() == 0
    if threads:
        setter.assert_called_once_with(threads)
    else:
        setter.assert_not_called()


def test_benchmark_cli_rejects_negative_threads(monkeypatch) -> None:
    """Fail before detector work when a thread setting is invalid."""
    monkeypatch.setattr("sys.argv", ["benchmark", "--opencv-threads", "-1"])
    with pytest.raises(SystemExit):
        benchmark.main()
