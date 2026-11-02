# SPDX-License-Identifier: Apache-2.0
"""Exercise worker lifecycle without a real camera, threads or private data."""

from __future__ import annotations

import io
import base64
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import numpy as np
import pytest

from tools import recognition
from tools.utils.arguments import Arguments
from tools.utils.vision import Detection, RecognitionMatch


class FakeRequest:
    """Track a request release even when processing raises an exception."""

    def __init__(self, camera: FakeCamera) -> None:
        """Associate the request with its fake camera owner."""
        self.camera = camera
        self.processing_frame = np.zeros((12, 16, 3), dtype=np.uint8)
        self.full_frame = Mock(return_value=np.ones((24, 32, 3), dtype=np.uint8))

    def __enter__(self) -> FakeRequest:
        """Expose synthetic pixels to the worker."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Account for returning the camera request under every exit path."""
        self.camera.releases += 1


class FakeCamera:
    """Track ownership operations without starting any device."""

    def __init__(self) -> None:
        """Create counters and configurable failure injection points."""
        self.starts = 0
        self.stops = 0
        self.closes = 0
        self.releases = 0
        self.requests: list[FakeRequest] = []
        self.start_error: Exception | None = None
        self.capture_error: Exception | None = None

    def start(self) -> None:
        """Record a start, optionally simulating a disconnected camera."""
        self.starts += 1
        if self.start_error:
            raise self.start_error

    def stop(self) -> None:
        """Record notification-driven suspension."""
        self.stops += 1

    def close(self) -> None:
        """Record final resource ownership release."""
        self.closes += 1

    def capture(self) -> FakeRequest:
        """Create a tracked request, optionally simulating a capture failure."""
        if self.capture_error:
            raise self.capture_error
        request = FakeRequest(self)
        self.requests.append(request)
        return request


class FakeEvent:
    """Deterministic event whose waits may drive scripted external commands."""

    def __init__(self) -> None:
        """Start unset, with no callback and no actual clock waiting."""
        self.state = False
        self.waits: list[float] = []
        self.on_wait: Callable[[], None] | None = None

    def set(self) -> None:
        """Set the event."""
        self.state = True

    def clear(self) -> None:
        """Clear the event."""
        self.state = False

    def is_set(self) -> bool:
        """Return the scripted state."""
        return self.state

    def wait(self, timeout: float) -> bool:
        """Record pacing without sleeping or starting a thread."""
        self.waits.append(timeout)
        if self.on_wait is not None and not self.state:
            self.on_wait()
        return self.state


@pytest.fixture
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a fully isolated worker with a synthetic, compatible gallery."""
    gallery = tmp_path / "gallery.pickle"
    gallery.write_bytes(
        pickle.dumps({"encodings": [np.zeros(128)], "names": ["subject"]})
    )
    monkeypatch.setattr(Arguments, "args", None)
    Arguments.prepareRecognitionArguments(
        [
            "--encodings",
            str(gallery),
            "--dataset",
            str(tmp_path / "dataset"),
            "--interval",
            "1000",
            "--output",
            "0",
            "--outputmm",
            "0",
        ]
    )
    settings = dict(Arguments.args)
    camera = FakeCamera()
    factory = Mock(return_value=camera)
    monkeypatch.setattr(recognition, "CameraStream", factory)
    detector_factory = Mock(return_value=object())
    monkeypatch.setattr(recognition, "FaceDetector", detector_factory)
    engine = SimpleNamespace(
        recognize=Mock(
            return_value=recognition.FrameResult(
                [Detection((1, 10, 10, 1))], [RecognitionMatch("subject", 0.1, True)]
            )
        )
    )
    engine_factory = Mock(return_value=engine)
    monkeypatch.setattr(recognition, "RecognitionEngine", engine_factory)
    signals: dict[int, Callable] = {}
    monkeypatch.setattr(
        recognition,
        "signal",
        SimpleNamespace(
            SIGINT=2,
            SIGTERM=15,
            signal=lambda number, callback: signals.__setitem__(number, callback),
        ),
    )
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        recognition,
        "Print",
        SimpleNamespace(printJson=lambda kind, value: events.append((kind, value))),
    )
    closed, active = FakeEvent(), FakeEvent()
    created_events = iter((closed, active))
    thread_factory = Mock()
    monkeypatch.setattr(
        recognition,
        "threading",
        SimpleNamespace(Event=lambda: next(created_events), Thread=thread_factory),
    )
    # Stable time also verifies the interval pacing rather than testing sleeps.
    monkeypatch.setattr(recognition.time, "monotonic", lambda: 12.0)
    return SimpleNamespace(
        settings=settings,
        camera=camera,
        factory=factory,
        detector_factory=detector_factory,
        engine=engine,
        engine_factory=engine_factory,
        signals=signals,
        events=events,
        closed=closed,
        active=active,
        thread_factory=thread_factory,
        gallery=gallery,
    )


def test_sigterm_releases_request_and_camera(worker: SimpleNamespace) -> None:
    """A stop signal may interrupt recognition without leaking a buffer."""
    result = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        worker.signals[15](15, None)
        return result

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert worker.camera.starts == 1
    assert worker.camera.releases == 1 and worker.camera.closes == 1
    assert worker.closed.is_set()
    assert ("login", {"names": ["subject"]}) in worker.events
    assert worker.thread_factory.call_count == 0


def test_pause_resume_logs_out_then_restarts_camera(worker: SimpleNamespace) -> None:
    """A notification pause cannot keep stale identities logged in."""
    worker.settings["run_only_on_notification"] = 1
    results = worker.engine.recognize.return_value
    captures = 0

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal captures
        captures += 1
        if captures == 2:
            worker.signals[2](2, None)
        return results

    def command_step() -> None:
        if len(worker.closed.waits) == 1:
            worker.active.clear()
        elif len(worker.closed.waits) == 2:
            worker.active.set()

    worker.engine.recognize.side_effect = recognize
    worker.closed.on_wait = command_step
    assert recognition.run(worker.settings) == 0
    assert worker.camera.starts == 2 and worker.camera.stops == 1
    assert worker.camera.releases == 2 and worker.camera.closes == 1
    assert worker.closed.waits[:2] == [1.0, 0.05]
    messages = [item for item in worker.events if item[0] in {"login", "logout"}]
    assert messages == [
        ("login", {"names": ["subject"]}),
        ("logout", {"names": ["subject"]}),
        ("login", {"names": ["subject"]}),
    ]
    worker.thread_factory.return_value.start.assert_called_once()


def test_parent_eof_closes_constructed_camera_without_starting(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-closed parent cannot leave a device running on startup."""
    worker.settings["run_only_on_notification"] = 1
    monkeypatch.setattr(recognition.sys, "stdin", io.StringIO(""))

    def thread(*, target: Callable, args: tuple, daemon: bool) -> SimpleNamespace:
        assert daemon
        return SimpleNamespace(start=lambda: target(*args))

    monkeypatch.setattr(recognition.threading, "Thread", thread)
    assert recognition.run(worker.settings) == 0
    assert worker.camera.starts == 0 and worker.camera.closes == 1
    assert not worker.camera.requests


def test_notification_thread_start_failure_closes_constructed_camera(
    worker: SimpleNamespace,
) -> None:
    """A failed command listener cannot strand an already configured camera."""
    worker.settings["run_only_on_notification"] = 1
    worker.thread_factory.return_value.start.side_effect = RuntimeError(
        "listener start failed"
    )
    with pytest.raises(RuntimeError, match="listener start failed"):
        recognition.run(worker.settings)
    assert worker.camera.closes == 1
    assert worker.camera.starts == 0 and not worker.camera.requests


@pytest.mark.parametrize("stage", ["start", "capture", "recognize", "enrollment"])
def test_failures_always_close_the_camera(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Cover failures both before and after a camera request is acquired."""
    error = RuntimeError("injected " + stage + " failure")
    if stage == "start":
        worker.camera.start_error = error
    elif stage == "capture":
        worker.camera.capture_error = error
    elif stage == "recognize":
        worker.engine.recognize.side_effect = error
    else:
        worker.settings["extendDataset"] = True
        monkeypatch.setattr(recognition, "extend_dataset", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="injected"):
        recognition.run(worker.settings)
    assert worker.camera.closes == 1
    assert worker.camera.releases == (1 if stage in {"recognize", "enrollment"} else 0)


def test_preview_encoding_failure_releases_camera_buffer(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JPEG failure cannot retain an ISP request or send a broken preview."""
    worker.settings["outputmm"] = 1
    monkeypatch.setattr(recognition, "annotate", Mock(return_value=np.zeros((2, 3, 3))))
    monkeypatch.setattr(
        recognition,
        "cv2",
        SimpleNamespace(IMWRITE_JPEG_QUALITY=1, imencode=lambda *args: (False, None)),
    )
    with pytest.raises(RuntimeError, match="encode camera preview"):
        recognition.run(worker.settings)
    assert worker.camera.releases == 1 and worker.camera.closes == 1
    assert not any(kind == "camera_image" for kind, _ in worker.events)


def test_disappearing_face_logs_out_and_keeps_preview_protocol(
    worker: SimpleNamespace,
) -> None:
    """Face disappearance clears identity while JPEG preview continues."""
    worker.settings["outputmm"] = 1
    present = worker.engine.recognize.return_value
    count = 0

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal count
        count += 1
        if count == 1:
            return present
        worker.signals[15](15, None)
        return recognition.FrameResult([], [])

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert [
        (kind, value) for kind, value in worker.events if kind in {"login", "logout"}
    ] == [
        ("login", {"names": ["subject"]}),
        ("logout", {"names": ["subject"]}),
    ]
    images = [value["image"] for kind, value in worker.events if kind == "camera_image"]
    assert len(images) == 2
    for encoded in images:
        data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        assert recognition.cv2.imdecode(data, recognition.cv2.IMREAD_COLOR).shape == (
            12,
            16,
            3,
        )
    assert worker.camera.releases == 2 and worker.camera.closes == 1


def test_local_preview_quit_destroys_windows_and_closes_camera(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-requested window quit follows the same camera cleanup path."""
    worker.settings["output"] = 1
    preview = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(recognition, "annotate", Mock(return_value=preview))
    show, destroy = Mock(), Mock()
    monkeypatch.setattr(
        recognition,
        "cv2",
        SimpleNamespace(
            imshow=show, waitKey=lambda _: ord("q"), destroyAllWindows=destroy
        ),
    )
    assert recognition.run(worker.settings) == 0
    show.assert_called_once_with("Frame", preview)
    destroy.assert_called_once()
    assert worker.camera.releases == 1 and worker.camera.closes == 1


@pytest.mark.parametrize(
    "problem", ["missing", "invalid_pickle", "invalid_gallery", "invalid_resolution"]
)
def test_invalid_input_fails_before_camera_creation(
    worker: SimpleNamespace, problem: str
) -> None:
    """Malformed configuration or enrollment cannot take camera ownership."""
    if problem == "missing":
        worker.gallery.unlink()
        expected = FileNotFoundError
    elif problem == "invalid_pickle":
        worker.gallery.write_bytes(b"not a pickle")
        expected = pickle.UnpicklingError
    elif problem == "invalid_gallery":
        worker.gallery.write_bytes(
            pickle.dumps({"encodings": [[0] * 127], "names": ["subject"]})
        )
        expected = ValueError
    else:
        worker.settings["resolution"] = "1920"
        expected = ValueError
    with pytest.raises(expected):
        recognition.run(worker.settings)
    worker.factory.assert_not_called()


def test_main_parses_explicit_arguments_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the CLI entry point without importing a camera at module load."""
    settings = {"example": "synthetic"}
    parse = Mock(side_effect=lambda argv: setattr(Arguments, "args", settings))
    monkeypatch.setattr(Arguments, "args", None)
    monkeypatch.setattr(Arguments, "prepareRecognitionArguments", parse)
    run = Mock(return_value=7)
    monkeypatch.setattr(recognition, "run", run)
    assert recognition.main(["--interval", "500"]) == 7
    parse.assert_called_once_with(["--interval", "500"])
    run.assert_called_once_with(settings)
