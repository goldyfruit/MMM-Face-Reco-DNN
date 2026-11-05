# SPDX-License-Identifier: Apache-2.0
"""Borrow existing camera requests for explicit Christmas props and photo turns."""

from __future__ import annotations

import base64
import json
import math
import os
import socket
import stat
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np

from tools.utils.experience_state import (
    GATEWAY,
    MAX_REVISION as MAX_REVISION,
    PROPS as PROPS,
    MARKERS,
    STATE_FILE,
    ExperienceView as ExperienceView,
    experience_view,
    fresh as fresh,
)

MAX_PHOTO_BYTES = 500000
MAX_PUBLIC_PHOTO_BYTES = 700000
PHOTO_SECONDS = 3


class ExperiencePropScanner:
    """Decode only an expected fixed stamp from the shared live camera request."""

    def __init__(self, state_file: Path = STATE_FILE, gateway: Path = GATEWAY) -> None:
        """Retain pacing/latch metadata only, never camera frames."""
        self.state_file, self.gateway = state_file, gateway
        self._next_sample = 0.0
        self._next_send = 0.0
        self._latched = False
        self._latch_key: tuple[Any, ...] | None = None
        self._blank_samples = 0
        self._detector: Any = None

    def scan(
        self, frame: Callable[[], np.ndarray], observed_at: float, monotonic: float
    ) -> None:
        """Borrow at most two frames per second only for the current expected prop."""
        if monotonic < self._next_sample:
            return
        self._next_sample = monotonic + 0.5
        view = experience_view(self.state_file, observed_at)
        if view is None or not view.listening or view.expected_prop is None:
            return
        latch_key = (*view.identity, view.expected_prop)
        if self._latch_key != latch_key:
            self._latched = False
            self._blank_samples = 0
            self._latch_key = latch_key
        try:
            if not stat.S_ISSOCK(self.gateway.lstat().st_mode):
                return
            prop = self._decode(frame())
        except (
            OSError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            cv2.error,
        ):
            return
        if prop != view.expected_prop:
            self._blank_samples += 1
            if self._blank_samples >= 2:
                self._latched = False
            return
        self._blank_samples = 0
        if self._latched or monotonic < self._next_send:
            return
        self._next_send = monotonic + 1
        self._latched = self._send(view.revision, prop, observed_at)

    def _decode(self, frame: np.ndarray) -> str | None:
        """Ignore arbitrary QR text, URLs and every noncatalog prop."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape
        if max(height, width) > 1920:
            scale = 1920 / max(height, width)
            grey = cv2.resize(
                grey, (max(1, round(width * scale)), max(1, round(height * scale)))
            )
        if self._detector is None:
            self._detector = cv2.QRCodeDetector()
        decoded, _, _ = self._detector.detectAndDecode(grey)
        return MARKERS.get(decoded)

    def _send(self, revision: int, prop: str, observed_at: float) -> bool:
        """Send only a bounded observation and wait briefly for local admission."""
        message = {
            "version": 1,
            "revision": revision,
            "prop": prop,
            "observed_at": observed_at,
        }
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.1)
                client.connect(str(self.gateway))
                client.sendall(json.dumps(message, allow_nan=False).encode() + b"\n")
                raw = client.recv(257)
            if len(raw) > 256 or not raw.endswith(b"\n"):
                return False
            reply = json.loads(raw)
        except (OSError, ValueError, RecursionError):
            return False
        return (
            isinstance(reply, dict)
            and set(reply) == {"accepted", "retry"}
            and reply["accepted"] is True
            and reply["retry"] is False
        )


class PhotoBooth:
    """Capture a requested keepsake preview in memory; write only on explicit save."""

    def __init__(
        self,
        emit: Callable[[str, dict[str, Any]], None],
        state_file: Path = STATE_FILE,
        directory: Path | None = None,
    ) -> None:
        """Configure public input/output without opening a camera or creating files."""
        self.emit, self.state_file = emit, state_file
        self.directory = (
            directory or Path.home() / ".local/share/thalovant/christmas-photos"
        )
        self.identity: tuple[str, str, int, str] | None = None
        self.preview: bytes | None = None
        self.revision = 0
        self.capture_revision = 0
        self.preview_capture_revision = 0
        self._seen_revision = -1
        self._action = "none"
        self._started: float | None = None
        self._last_status: tuple[int, str, int | None] | None = None
        self._filename: str | None = None
        self._consumed: dict[tuple[Any, ...], None] = {}

    def _status(self, status: str, seconds: int | None = None) -> None:
        """Publish one bounded status transition; only previews carry image bytes."""
        key = (self.revision, status, seconds)
        if key == self._last_status:
            return
        value: dict[str, Any] = {
            "version": 1,
            "revision": self.revision,
            "status": status,
        }
        if self.capture_revision > 0:
            value["capture_revision"] = self.capture_revision
        if seconds is not None:
            value["seconds"] = seconds
        if status == "preview" and self.preview is not None:
            value["image"] = base64.b64encode(self.preview).decode("ascii")
        if len(json.dumps(value).encode()) <= MAX_PUBLIC_PHOTO_BYTES:
            self.emit("christmas_photo", value)
            self._last_status = key

    def reset(self) -> None:
        """Drop an uncommitted image on exit, disable, stale state or worker pause."""
        had_preview = self.preview is not None or self._started is not None
        self.preview = None
        self.preview_capture_revision = 0
        self._filename = None
        self._started = None
        self._action = "none"
        self.identity = None
        self._seen_revision = -1
        if had_preview:
            self._status("discarded")

    def process_frame(
        self, frame: Callable[[], np.ndarray], observed_at: float, monotonic: float
    ) -> None:
        """Call the shared lazy full frame exactly once after an explicit countdown."""
        view = experience_view(self.state_file, observed_at)
        if view is None or view.stage != "keepsake":
            self.reset()
            return
        if self.identity != view.identity:
            self.reset()
            self.identity = view.identity
        if view.revision < self._seen_revision:
            self.reset()
            return
        self.revision = view.revision
        self.capture_revision = view.photo_capture_revision
        mismatched_preview = (
            self.preview is not None
            and self.preview_capture_revision != view.photo_capture_revision
        )
        if mismatched_preview:
            self.preview = None
            self.preview_capture_revision = 0
            self._filename = None
        if view.revision != self._seen_revision:
            self._seen_revision = view.revision
            command = (*view.identity, view.revision, view.photo_action)
            if command in self._consumed:
                self._action = "none"
                return
            self._consumed[command] = None
            while len(self._consumed) > 128:
                self._consumed.pop(next(iter(self._consumed)))
            self._action = view.photo_action
            self._started = None
            if view.photo_action == "capture":
                self.preview = None
                self.preview_capture_revision = 0
                self._filename = None
            elif view.photo_action == "discard":
                self.preview = None
                self.preview_capture_revision = 0
                self._filename = None
                self._status("unavailable" if mismatched_preview else "discarded")
            elif view.photo_action == "save":
                if (
                    self.preview is None
                    or self.preview_capture_revision != view.photo_capture_revision
                ):
                    self._status("unavailable")
                else:
                    self._save()
            elif self.preview is not None:
                self._status("preview")
        if self._action != "capture":
            return
        if not view.listening:
            self._started = None
            return
        if self._started is None:
            self._started = monotonic
        remaining = PHOTO_SECONDS - (monotonic - self._started)
        if remaining > 0:
            self._status("countdown", min(PHOTO_SECONDS, max(1, math.ceil(remaining))))
            return
        self._action = "none"
        self._started = None
        try:
            pixels = frame()
            height, width = pixels.shape[:2]
            if max(height, width) > 1200:
                scale = 1200 / max(height, width)
                pixels = cv2.resize(
                    pixels,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                )
            for quality in (85, 70, 50):
                ok, encoded = cv2.imencode(
                    ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if ok and len(encoded) <= MAX_PHOTO_BYTES:
                    current = experience_view(self.state_file, observed_at)
                    if (
                        current is None
                        or current.identity != view.identity
                        or current.revision != view.revision
                        or current.stage != "keepsake"
                        or not current.listening
                        or current.photo_action != "capture"
                        or current.photo_capture_revision != view.photo_capture_revision
                    ):
                        self.reset()
                        return
                    self.preview = encoded.tobytes()
                    self.preview_capture_revision = view.photo_capture_revision
                    self._filename = "christmas-" + uuid4().hex + ".jpg"
                    self._status("preview")
                    return
        except (AttributeError, TypeError, ValueError, RuntimeError, cv2.error):
            pass
        self.preview = None
        self.preview_capture_revision = 0
        self._status("unavailable")

    def _save(self) -> None:
        """Create a private image in the configured owned folder, never the face gallery."""
        destination: Path | None = None
        created = False
        try:
            if (
                not self.directory.is_absolute()
                or not self._filename
                or self.preview is None
            ):
                raise ValueError("An explicit owned photo directory is required")
            for parent in (*reversed(self.directory.parents), self.directory):
                if parent.is_symlink():
                    raise ValueError("Linked photo directory")
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            info = self.directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError("Photo directory must be private and owned")
            destination = self.directory / self._filename
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            created = True
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.preview)
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, ValueError):
            if created and destination is not None:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            self._status("unavailable")
            return
        self.preview = None
        self.preview_capture_revision = 0
        self._filename = None
        self._action = "none"
        self._status("saved")
