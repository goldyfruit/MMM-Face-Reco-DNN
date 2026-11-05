"""Detect and recognize faces with a fresh-frame, testable camera pipeline."""

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle
import signal
import sys
import threading
import time
import uuid
from types import FrameType
from typing import Any, Callable

# Support both `python tools/recognition.py` and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from tools.utils.arguments import Arguments
from tools.utils.camera import CameraStream
from tools.utils.experience_camera import ExperiencePropScanner, PhotoBooth
from tools.utils.letter_marker import LetterScanner
from tools.utils.print import Print
from tools.utils.vision import (
    Detection,
    FaceDetector,
    FaceMatcher,
    RecognitionMatch,
    to_rgb,
)


@dataclass(frozen=True)
class FrameResult:
    """Keep each face's coordinates and identity score associated."""

    detections: list[Detection]
    matches: list[RecognitionMatch]

    @property
    def names(self) -> list[str]:
        """Return distinct labels in detection order for login/logout events."""
        return list(dict.fromkeys(match.name for match in self.matches))


class RecognitionEngine:
    """Separate detection, unchanged dlib descriptors and gallery matching."""

    def __init__(
        self, detector: FaceDetector, matcher: FaceMatcher, face_module: Any = None
    ) -> None:
        """Allow dependency injection without a camera or private gallery."""
        if face_module is None:
            import face_recognition

            face_module = face_recognition
        self.detector = detector
        self.matcher = matcher
        self.face_module = face_module

    def recognize(self, bgr_frame: np.ndarray) -> FrameResult:
        """Compute descriptors only for detected faces and keep box alignment."""
        detections = self.detector.detect(bgr_frame)
        if not detections:
            return FrameResult([], [])
        encodings = self.face_module.face_encodings(
            to_rgb(bgr_frame),
            known_face_locations=[face.box for face in detections],
            num_jitters=1,
            model="small",
        )
        if len(encodings) != len(detections):
            raise RuntimeError("Descriptor count does not match detected faces")
        return FrameResult(detections, self.matcher.match_many(encodings))


def annotate(
    frame: np.ndarray, result: FrameResult, cv_module: Any = cv2
) -> np.ndarray:
    """Draw labels on a copy, never on a future enrollment image."""
    preview = frame.copy()
    for detection, match in zip(result.detections, result.matches):
        top, right, bottom, left = detection.box
        cv_module.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)
        distance = "n/a" if match.distance is None else f"{match.distance:.2f}"
        cv_module.putText(
            preview,
            f"{match.name} ({distance})",
            (left, max(12, top - 15)),
            cv_module.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
    return preview


def extend_dataset(
    dataset: str,
    result: FrameResult,
    logins: list[str],
    full_frame: Callable[[], np.ndarray],
    cv_module: Any = cv2,
) -> bool:
    """Save one unannotated full-resolution frame only for an unambiguous subject.

    Group images must not silently become training examples for one identity.
    The existing encoder format and existing images remain untouched.
    """
    if len(result.detections) != 1 or len(result.matches) != 1 or not logins:
        return False
    name = result.matches[0].name
    if name not in logins:
        return False
    base = Path(dataset).resolve()
    destination = (base / name).resolve()
    if not name or destination.parent != base:
        raise ValueError("Dataset label must be a single safe directory name")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (
        name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    )
    if not cv_module.imwrite(str(target), full_frame()):
        raise RuntimeError("Could not save dataset image")
    return True


def monitor_commands(
    stream: Any,
    active: threading.Event,
    closed: threading.Event,
    reset_ready: Callable[[], None] | None = None,
) -> None:
    """Accept complete commands and stop cleanly when the parent closes stdin."""
    for line in stream:
        command = line.strip().lower()
        if command == "start":
            if reset_ready is not None:
                reset_ready()
            active.set()
        elif command == "stop":
            if reset_ready is not None:
                reset_ready()
            active.clear()
        elif command == "exit":
            closed.set()
            return
    closed.set()


def run(settings: dict[str, Any]) -> int:
    """Run the worker while releasing every camera request and camera handle."""
    opencv_threads = settings.get("opencvThreads", 0)
    if opencv_threads > 0:
        cv2.setNumThreads(opencv_threads)
    closed, active = threading.Event(), threading.Event()
    active.set()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        """Request shutdown from SIGINT or SIGTERM without camera I/O in a handler."""
        closed.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    Print.printJson("status", "loading encodings + face detector...")
    # Legacy encoding files are trusted local training data, not remote input.
    with Path(settings["encodings"]).open("rb") as source:
        data = pickle.load(source)
    matcher = FaceMatcher(
        data["encodings"],
        data["names"],
        settings["tolerance"],
        settings["identityMargin"],
    )
    detector = FaceDetector(
        method=settings["method"],
        detection_method=settings["detectionMethod"],
        cascade_path=settings["cascade"],
        yunet_model=settings["yunetModel"],
        score_threshold=settings["yunetScoreThreshold"],
        nms_threshold=settings["yunetNmsThreshold"],
        top_k=settings["yunetTopK"],
        upsample=settings["detectionUpsample"],
    )
    engine = RecognitionEngine(detector, matcher)
    parts = settings["resolution"].split(",")
    if len(parts) != 2:
        raise ValueError("Resolution must contain width,height")
    camera = CameraStream(
        resolution=tuple(int(value) for value in parts),
        process_width=settings["processWidth"],
        enable_lores=settings["cameraLores"],
        frame_rate=settings["cameraFrameRate"],
        lores_format=settings["cameraLoresFormat"],
        autofocus=settings.get("cameraAutofocus", "default"),
        lens_position=settings.get("cameraLensPosition"),
        exposure_mode=settings.get("cameraExposureMode", "normal"),
        brightness=settings["brightness"],
        contrast=settings["contrast"],
        rotation=settings["rotateCamera"],
    )
    previous: list[str] = []
    running = False
    activation_generation = 0
    previous_generation = 0
    reported_ready_generation = None
    presence_epoch = uuid.uuid4().hex
    presence_sequence = 0
    letter_scanner = LetterScanner()
    experience_scanner = ExperiencePropScanner()
    photo_directory = settings.get("christmasPhotoDirectory")
    photo_booth = PhotoBooth(
        Print.printJson,
        directory=Path(photo_directory) if photo_directory else None,
    )

    def reset_ready() -> None:
        """Require a new processed-frame readiness event after a command change."""
        nonlocal activation_generation
        activation_generation += 1

    try:
        if settings["run_only_on_notification"]:
            threading.Thread(
                target=monitor_commands,
                args=(sys.stdin, active, closed, reset_ready),
                daemon=True,
            ).start()
        Print.printJson("status", "starting video stream...")
        Print.printJson("status", "Starting face recognition loop.")
        while not closed.is_set():
            started = time.monotonic()
            if not active.is_set():
                photo_booth.reset()
                reported_ready_generation = None
                if running:
                    camera.stop()
                    running = False
                    if previous:
                        Print.printJson("logout", {"names": previous})
                        previous = []
                closed.wait(0.05)
                continue
            if not running:
                camera.start()
                running = True
            frame_generation = activation_generation
            if previous_generation != frame_generation:
                # The browser cleared its sessions on pause, even when stop/start
                # arrived too quickly for this loop to observe the inactive state.
                previous = []
                previous_generation = frame_generation
            with camera.capture() as captured:
                observed_at = time.time()
                result = engine.recognize(captured.processing_frame)
                identities_settled = True
                if settings.get("cameraAutofocus") == "face":
                    camera.focus_on_faces(
                        captured, [detection.box for detection in result.detections]
                    )
                    identities_settled = camera.face_focus_status.get("settled", False)
                identities_settled = (
                    identities_settled
                    and active.is_set()
                    and frame_generation == activation_generation
                )
                names = result.names
                presence_sequence += 1
                # Fresh inference metadata only: no image, geometry, confidence,
                # gallery or cached login state is exported to other modules.
                Print.printJson(
                    "presence",
                    {
                        "version": 1,
                        "epoch": presence_epoch,
                        "sequence": presence_sequence,
                        "observed_at": observed_at,
                        "available": identities_settled and len(result.matches) <= 8,
                        "labels": [match.name for match in result.matches]
                        if identities_settled and len(result.matches) <= 8
                        else [],
                    },
                )
                logins = (
                    [name for name in names if name not in previous]
                    if identities_settled
                    else []
                )
                logouts = (
                    [name for name in previous if name not in names]
                    if identities_settled
                    else []
                )
                if identities_settled and settings["extendDataset"]:
                    extend_dataset(
                        settings["dataset"], result, logins, captured.full_frame
                    )
                if settings["output"] or settings["outputmm"]:
                    preview = annotate(captured.processing_frame, result)
                    if settings["output"]:
                        cv2.imshow("Frame", preview)
                    if settings["outputmm"]:
                        ok, buffer = cv2.imencode(
                            ".jpg",
                            preview,
                            [cv2.IMWRITE_JPEG_QUALITY, settings["previewQuality"]],
                        )
                        if not ok:
                            raise RuntimeError("Could not encode camera preview")
                        Print.printJson(
                            "camera_image",
                            {"image": base64.b64encode(buffer).decode("ascii")},
                        )
                if logins:
                    Print.printJson("login", {"names": logins})
                if logouts:
                    Print.printJson("logout", {"names": logouts})
                if identities_settled:
                    previous = names
                if active.is_set() and not closed.is_set():
                    letter_scanner.scan(captured.full_frame, observed_at, started)
                    experience_scanner.scan(captured.full_frame, observed_at, started)
                    photo_booth.process_frame(captured.full_frame, observed_at, started)
            if (
                active.is_set()
                and frame_generation == activation_generation
                and reported_ready_generation != frame_generation
            ):
                Print.printJson("ready", True)
                reported_ready_generation = frame_generation
            if settings["output"] and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            closed.wait(
                max(0.0, settings["interval"] / 1000 - (time.monotonic() - started))
            )
    finally:
        photo_booth.reset()
        camera.close()
        if settings["output"]:
            cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse options and start a worker only for an explicit CLI invocation."""
    Arguments.prepareRecognitionArguments(argv)
    return run(Arguments.args)


if __name__ == "__main__":
    sys.exit(main())
