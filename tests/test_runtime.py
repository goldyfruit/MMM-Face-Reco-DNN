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
        self.focus_error: Exception | None = None
        self.focus_calls: list[tuple[FakeRequest, list[tuple[int, int, int, int]]]] = []
        self.face_focus_status = {"state": "waiting", "settled": True}

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

    def focus_on_faces(
        self, captured: FakeRequest, boxes: list[tuple[int, int, int, int]]
    ) -> bool:
        """Record same-request face targeting without a lens or camera."""
        assert captured is self.requests[-1]
        assert self.releases == len(self.requests) - 1
        self.focus_calls.append((captured, boxes))
        if self.focus_error:
            raise self.focus_error
        return bool(boxes)


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


def test_presence_is_fresh_per_inference_and_does_not_export_cached_sessions(
    worker: SimpleNamespace,
) -> None:
    """A consumer receives current labels, not delayed login state or pixels."""
    result = worker.engine.recognize.return_value
    calls = 0

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        """Return one present frame followed by a genuinely empty frame."""
        nonlocal calls
        calls += 1
        if calls == 2:
            worker.closed.set()
            return recognition.FrameResult([], [])
        return result

    worker.engine.recognize.side_effect = recognize
    recognition.run(worker.settings)
    values = [value for kind, value in worker.events if kind == "presence"]
    assert len(values) == 2
    assert values[0]["labels"] == ["subject"] and values[1]["labels"] == []
    assert values[0]["sequence"] == 1 and values[1]["sequence"] == 2
    assert values[0]["epoch"] == values[1]["epoch"] and len(values[0]["epoch"]) == 32
    assert set(values[0]) == {
        "version",
        "epoch",
        "sequence",
        "observed_at",
        "available",
        "labels",
    }


def test_presence_timestamp_precedes_slow_inference(
    worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delayed result cannot relabel old observed pixels with a fresh timestamp."""
    now = [1000.0]
    monkeypatch.setattr(recognition.time, "time", lambda: now[0])
    result = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        """Simulate slow processing, without sleeping or touching a camera."""
        now[0] += 10
        worker.closed.set()
        return result

    worker.engine.recognize.side_effect = recognize
    recognition.run(worker.settings)
    value = next(value for kind, value in worker.events if kind == "presence")
    assert value["observed_at"] == 1000.0 and now[0] == 1010.0


@pytest.mark.parametrize("count", [0, 2, 8, 9])
def test_presence_preserves_duplicate_face_count_and_bounds_payload(
    worker: SimpleNamespace, count: int
) -> None:
    """Two same-label faces stay a group; over-cap groups become unavailable."""
    worker.engine.recognize.return_value = recognition.FrameResult(
        [Detection((1, 10, 10, 1))] * count,
        [RecognitionMatch("subject", 0.1, True)] * count,
    )
    worker.closed.on_wait = worker.closed.set
    recognition.run(worker.settings)
    value = next(value for kind, value in worker.events if kind == "presence")
    assert value["available"] is (count <= 8)
    assert value["labels"] == (["subject"] * count if count <= 8 else [])


def test_presence_unavailable_during_autofocus_transition(
    worker: SimpleNamespace,
) -> None:
    """Unsettled autofocus cannot expose current identities as settled presence."""
    worker.settings["cameraAutofocus"] = "face"
    worker.camera.face_focus_status["settled"] = False
    worker.closed.on_wait = worker.closed.set
    recognition.run(worker.settings)
    value = next(value for kind, value in worker.events if kind == "presence")
    assert value["available"] is False and value["labels"] == []


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
    assert [kind for kind, _ in worker.events].count("ready") == 2
    worker.thread_factory.return_value.start.assert_called_once()


@pytest.mark.parametrize("pause_between_frames", [True, False])
@pytest.mark.parametrize("known_face", [True, False])
def test_resume_reconciles_sessions_without_a_preview(
    worker: SimpleNamespace, pause_between_frames: bool, known_face: bool
) -> None:
    """Fast pause/start restores the same known face after frontend reset."""
    worker.settings["run_only_on_notification"] = 1
    captures = 0
    present = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal captures
        captures += 1
        if captures == 2:
            worker.closed.set()
        return present if known_face else recognition.FrameResult([], [])

    def command_step() -> None:
        reset_ready = worker.thread_factory.call_args.kwargs["args"][3]
        if len(worker.closed.waits) == 1:
            reset_ready()
            worker.active.clear()
            if not pause_between_frames:
                reset_ready()
                worker.active.set()
        elif len(worker.closed.waits) == 2:
            reset_ready()
            worker.active.set()

    worker.engine.recognize.side_effect = recognize
    worker.closed.on_wait = command_step
    assert recognition.run(worker.settings) == 0
    assert captures == 2
    assert [kind for kind, _ in worker.events].count("ready") == 2
    logins = [value for kind, value in worker.events if kind == "login"]
    assert logins == ([{"names": ["subject"]}] * 2 if known_face else [])
    assert not any(kind == "camera_image" for kind, _ in worker.events)


def test_commands_invalidate_readiness_on_each_activation_change() -> None:
    """The actual command monitor marks quick stop/start cycles as new work."""
    active, closed = FakeEvent(), FakeEvent()
    reset_ready = Mock()
    recognition.monitor_commands(
        io.StringIO("stop\nstart\nexit\n"), active, closed, reset_ready
    )
    assert reset_ready.call_count == 2
    assert active.is_set() and closed.is_set()


def test_command_change_during_recognition_rejects_stale_identity_and_enrollment(
    worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-flight pre-pause result cannot reopen a session or train a face."""
    worker.settings.update(run_only_on_notification=1, extendDataset=True)
    known = worker.engine.recognize.return_value
    unknown = recognition.FrameResult(
        known.detections, [RecognitionMatch("unknown", 0.9, False)]
    )
    captures = 0

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal captures
        captures += 1
        if captures == 2:
            reset_ready = worker.thread_factory.call_args.kwargs["args"][3]
            reset_ready()
            worker.active.clear()
            reset_ready()
            worker.active.set()
            return unknown
        if captures == 3:
            worker.closed.set()
        return known

    extension = Mock(return_value=False)
    monkeypatch.setattr(recognition, "extend_dataset", extension)
    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert [event for event in worker.events if event[0] in {"login", "logout"}] == [
        ("login", {"names": ["subject"]})
    ] * 2
    assert extension.call_count == 2
    assert all(call.args[1] is known for call in extension.call_args_list)
    assert [event[0] for event in worker.events].count("ready") == 2
    assert worker.camera.releases == 3


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


@pytest.mark.parametrize(
    "stage", ["start", "capture", "recognize", "focus", "enrollment"]
)
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
    elif stage == "focus":
        worker.settings["cameraAutofocus"] = "face"
        worker.camera.focus_error = error
    else:
        worker.settings["extendDataset"] = True
        monkeypatch.setattr(recognition, "extend_dataset", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="injected"):
        recognition.run(worker.settings)
    assert worker.camera.closes == 1
    assert worker.camera.releases == (
        1 if stage in {"recognize", "focus", "enrollment"} else 0
    )
    assert not any(kind == "ready" for kind, _ in worker.events)


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


@pytest.mark.parametrize("threads", [None, 0, 2])
def test_opencv_thread_budget_preserves_default_or_applies_before_devices(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    threads: int | None,
) -> None:
    """Zero must not disable OpenCV threading; a positive budget applies early."""
    if threads is None:
        worker.settings.pop("opencvThreads", None)
    else:
        worker.settings["opencvThreads"] = threads
    order = []
    set_threads = Mock(side_effect=lambda value: order.append(("threads", value)))
    monkeypatch.setattr(recognition.cv2, "setNumThreads", set_threads)
    worker.engine_factory.side_effect = lambda *args: (
        order.append(("engine", None)) or worker.engine
    )
    worker.factory.side_effect = lambda **kwargs: (
        order.append(("camera", None)) or worker.camera
    )
    result = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        worker.signals[15](15, None)
        return result

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    if threads:
        set_threads.assert_called_once_with(2)
        assert order == [("threads", 2), ("engine", None), ("camera", None)]
    else:
        set_threads.assert_not_called()
        assert order == [("engine", None), ("camera", None)]


@pytest.mark.parametrize("threads", ["-1", "65", "1.5"])
def test_invalid_opencv_thread_budget_is_rejected(threads: str) -> None:
    """Reject negative, excessive and non-integral budgets before camera access."""
    with pytest.raises(SystemExit):
        Arguments.prepareRecognitionArguments(["--opencvThreads", threads])


def test_opencv_thread_budget_parser_keeps_default_and_accepts_bounds() -> None:
    """The new option is opt-in and accepts only its documented integer range."""
    Arguments.prepareRecognitionArguments([])
    assert Arguments.get("opencvThreads") == 0
    for threads in (0, 2, 64):
        Arguments.prepareRecognitionArguments(["--opencvThreads", str(threads)])
        assert Arguments.get("opencvThreads") == threads


def test_ready_is_emitted_once_after_first_request_release(
    worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability means processing succeeded and the camera buffer returned."""
    result = worker.engine.recognize.return_value
    captures = 0
    releases_when_ready = []

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal captures
        captures += 1
        if captures == 2:
            worker.signals[15](15, None)
        return result

    def message(kind: str, value: Any) -> None:
        if kind == "ready":
            assert value is True
            releases_when_ready.append(worker.camera.releases)

    worker.engine.recognize.side_effect = recognize
    monkeypatch.setattr(recognition, "Print", SimpleNamespace(printJson=message))
    assert recognition.run(worker.settings) == 0
    assert releases_when_ready == [1]
    assert worker.camera.releases == 2


@pytest.mark.parametrize(
    "focus,exposure,lens_position",
    [
        ("continuous", "normal", None),
        ("manual", "short", 1.0),
        ("default", "normal", None),
        ("face", "normal", 1.0),
    ],
)
def test_camera_control_settings_reach_the_stream(
    worker: SimpleNamespace,
    focus: str,
    exposure: str,
    lens_position: float | None,
) -> None:
    """The user-facing options are connected to the advertised-control layer."""
    worker.settings["cameraAutofocus"] = focus
    worker.settings["cameraExposureMode"] = exposure
    worker.settings["cameraLensPosition"] = lens_position
    result = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        worker.closed.set()
        return result

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert worker.factory.call_args.kwargs["autofocus"] == focus
    assert worker.factory.call_args.kwargs["exposure_mode"] == exposure
    assert worker.factory.call_args.kwargs["lens_position"] == lens_position
    if focus != "face":
        assert worker.camera.focus_calls == []


@pytest.mark.parametrize("face_present", [True, False])
def test_face_focus_targets_current_detection_without_requiring_identity(
    worker: SimpleNamespace, face_present: bool
) -> None:
    """Unknown faces and empty scenes reach the ROI controller before release."""
    worker.settings["cameraAutofocus"] = "face"
    detections = [Detection((1, 10, 10, 1))] if face_present else []
    matches = [RecognitionMatch("unknown", 0.9, False)] if face_present else []

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        worker.closed.set()
        return recognition.FrameResult(detections, matches)

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert worker.camera.focus_calls == [
        (worker.camera.requests[0], [detection.box for detection in detections])
    ]
    assert worker.camera.releases == 1 and worker.camera.closes == 1


@pytest.mark.parametrize("focus_state", ["pending", "scanning", "restoring"])
def test_face_focus_transition_keeps_preview_but_defers_identity_and_enrollment(
    worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, focus_state: str
) -> None:
    """A transient blurred result cannot replace a known session or training set."""
    worker.settings.update(cameraAutofocus="face", extendDataset=True, outputmm=1)
    known = worker.engine.recognize.return_value
    unknown = recognition.FrameResult(
        known.detections, [RecognitionMatch("unknown", 0.9, False)]
    )
    steps = iter(((known, True), (unknown, False), (known, True)))
    captures = 0

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal captures
        captures += 1
        result, settled = next(steps)
        worker.camera.face_focus_status = {
            "state": "locked" if settled else focus_state,
            "settled": settled,
        }
        if captures == 3:
            worker.closed.set()
        return result

    extension = Mock(return_value=False)
    monkeypatch.setattr(recognition, "extend_dataset", extension)
    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert [event for event in worker.events if event[0] in {"login", "logout"}] == [
        ("login", {"names": ["subject"]})
    ]
    assert [event[0] for event in worker.events].count("camera_image") == 3
    assert extension.call_count == 2
    assert all(call.args[1] is known for call in extension.call_args_list)
    assert extension.call_args_list[0].args[2] == ["subject"]
    assert extension.call_args_list[1].args[2] == []
    assert worker.camera.releases == 3 and worker.camera.closes == 1


def test_no_face_cancels_scan_without_preserving_a_ghost_session(
    worker: SimpleNamespace,
) -> None:
    """The camera controller's safe no-face state permits normal identity logout."""
    worker.settings["cameraAutofocus"] = "face"
    known = worker.engine.recognize.return_value
    unknown = recognition.FrameResult(
        known.detections, [RecognitionMatch("unknown", 0.9, False)]
    )
    steps = iter(
        ((known, True), (unknown, False), (recognition.FrameResult([], []), True))
    )

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        result, settled = next(steps)
        worker.camera.face_focus_status = {"settled": settled}
        if not result.detections:
            worker.closed.set()
        return result

    worker.engine.recognize.side_effect = recognize
    assert recognition.run(worker.settings) == 0
    assert [event for event in worker.events if event[0] in {"login", "logout"}] == [
        ("login", {"names": ["subject"]}),
        ("logout", {"names": ["subject"]}),
    ]
    assert worker.camera.focus_calls[-1][1] == []


@pytest.mark.parametrize(
    "flag,value", [("--cameraAutofocus", "fast"), ("--cameraExposureMode", "invalid")]
)
def test_invalid_camera_control_modes_fail_before_hardware(
    flag: str, value: str
) -> None:
    """Reject unsupported user modes rather than silently substituting one."""
    with pytest.raises(SystemExit):
        Arguments.prepareRecognitionArguments([flag, value])


def test_camera_control_default_modes_are_explicit() -> None:
    """Camera defaults preserve the existing focus and normal exposure profile."""
    Arguments.prepareRecognitionArguments([])
    assert Arguments.get("cameraAutofocus") == "default"
    assert Arguments.get("cameraLensPosition") is None
    assert Arguments.get("cameraExposureMode") == "normal"


@pytest.mark.parametrize("position", ["-1", "nan", "inf", "-inf", "wrong"])
def test_invalid_manual_lens_position_is_rejected(position: str) -> None:
    """A lens request cannot bypass finite nonnegative input validation."""
    with pytest.raises(SystemExit):
        Arguments.prepareRecognitionArguments(
            ["--cameraAutofocus", "manual", "--cameraLensPosition", position]
        )


def test_lens_position_requires_supported_mode_and_preserves_dioptres() -> None:
    """Ignore no focus requests silently; manual focus accepts the stated units."""
    with pytest.raises(SystemExit):
        Arguments.prepareRecognitionArguments(["--cameraLensPosition", "1.0"])
    for mode in ("manual", "face"):
        Arguments.prepareRecognitionArguments(
            ["--cameraAutofocus", mode, "--cameraLensPosition", "1.0"]
        )
        assert Arguments.get("cameraLensPosition") == 1.0


def test_christmas_features_borrow_same_live_request_and_reset_on_worker_close(
    worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new features cannot create another camera or retain a photo after exit."""
    scanner = SimpleNamespace(scan=Mock())
    booth = SimpleNamespace(process_frame=Mock(), reset=Mock())
    monkeypatch.setattr(recognition, "ExperiencePropScanner", lambda: scanner)
    monkeypatch.setattr(recognition, "PhotoBooth", lambda *args, **kwargs: booth)
    calls = 0
    original = worker.engine.recognize.return_value

    def recognize(_: np.ndarray) -> recognition.FrameResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            worker.closed.set()
        return original

    worker.engine.recognize.side_effect = recognize
    recognition.run(worker.settings)
    assert worker.factory.call_count == 1
    assert scanner.scan.call_count == booth.process_frame.call_count == 1
    assert scanner.scan.call_args.args[0] is worker.camera.requests[0].full_frame
    assert booth.process_frame.call_args.args[0] is worker.camera.requests[0].full_frame
    booth.reset.assert_called_once()
    assert worker.camera.requests[0].full_frame.call_count == 0
