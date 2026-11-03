# SPDX-License-Identifier: Apache-2.0
"""Read the two North Pole letter stamps from an armed camera request."""

import json
import os
from pathlib import Path
import socket
import stat
from typing import Callable

import cv2
import numpy as np

STATE_FILE = Path("/christmas-state/state.json")
GATEWAY = Path("/christmas-greetings/letter.sock")
HUNT_ID = "a66c45cff88f36456a334ea2"
PROFILE_ID = "practice-lost-letter"
MARKERS = {"THALOVANT:LETTER:EN": "en", "THALOVANT:LETTER:FR": "fr"}
MAX_STATE_BYTES = 16384
MAX_ACK_BYTES = 256


def _fresh(value: object, now: float, limit: float) -> bool:
    """Accept a bounded timestamp from the shared local wall clock."""
    return type(value) in (int, float) and now - limit <= value <= now + 2


def armed_revision(state_file: Path, now: float) -> int | None:
    """Return only a fresh, idle-Voice, armed lost-letter view's revision."""
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(state_file, flags), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            body = stream.read(MAX_STATE_BYTES + 1)
        if len(body) > MAX_STATE_BYTES:
            return None
        envelope = json.loads(body)
    except (OSError, ValueError, RecursionError):
        return None
    if (
        not isinstance(envelope, dict)
        or type(envelope.get("schema_version")) is not int
        or envelope["schema_version"] != 1
        or envelope.get("connected") is not True
        or not _fresh(envelope.get("updated_at"), now, 20)
        or envelope.get("voiceState") != "listening"
        or not _fresh(envelope.get("voiceUpdatedAt"), now, 20)
        or envelope.get("presentation") is not None
    ):
        return None
    media = envelope.get("media")
    if media is not None and (
        not isinstance(media, dict) or media.get("video") is not None
    ):
        return None
    state = envelope.get("state")
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != 1
        or state.get("hunt_id") != HUNT_ID
        or state.get("status") != "paused"
        or state.get("take_armed") is not True
        or state.get("clue", False) is not None
        or "hint" in state
        or "reveal" in state
    ):
        return None
    profile = state.get("profile")
    progress = state.get("progress")
    revision = state.get("revision")
    if (
        not isinstance(profile, dict)
        or profile.get("id") != PROFILE_ID
        or not isinstance(progress, dict)
        or type(progress.get("found")) is not int
        or progress["found"] != 0
        or type(progress.get("total")) is not int
        or progress["total"] != 1
        or type(revision) is not int
        or not 0 <= revision <= 2**53 - 1
    ):
        return None
    return revision


class LetterScanner:
    """Scan in-memory frames only while the existing camera and Hub are ready."""

    def __init__(self, state_file: Path = STATE_FILE, gateway: Path = GATEWAY) -> None:
        """Keep only a pacing timer and a presentation latch between frames."""
        self.state_file = state_file
        self.gateway = gateway
        self._detector = None
        self._next_sample = 0.0
        self._next_send = 0.0
        self._latched = False
        self._blank_samples = 0

    def scan(
        self, frame: Callable[[], np.ndarray], observed_at: float, monotonic: float
    ) -> None:
        """Borrow one full frame at most twice a second and discard it immediately."""
        if monotonic < self._next_sample:
            return
        self._next_sample = monotonic + 0.5
        try:
            if not stat.S_ISSOCK(self.gateway.lstat().st_mode):
                return
        except OSError:
            return
        revision = armed_revision(self.state_file, observed_at)
        if revision is None:
            return
        try:
            language = self._decode(frame())
        except (AttributeError, ValueError, cv2.error):
            return
        if language is None:
            if self._latched:
                self._blank_samples += 1
                self._latched = self._blank_samples < 2
            return
        self._blank_samples = 0
        if self._latched or monotonic < self._next_send:
            return
        self._next_send = monotonic + 1.0
        self._latched = self._send(revision, language, observed_at)

    def _decode(self, frame: np.ndarray) -> str | None:
        """Match the exact fixed vocabulary and discard all decoded QR contents."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape
        longest = max(width, height)
        if longest > 1920:
            scale = 1920 / longest
            grey = cv2.resize(
                grey,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if self._detector is None:
            self._detector = cv2.QRCodeDetector()
        decoded, _, _ = self._detector.detectAndDecode(grey)
        return MARKERS.get(decoded)

    def _send(self, revision: int, language: str, observed_at: float) -> bool:
        """Return true once the gateway consumes this physical presentation."""
        message = (
            json.dumps(
                {"revision": revision, "language": language, "observedAt": observed_at},
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.1)
                client.connect(str(self.gateway))
                client.sendall(message)
                response = client.recv(MAX_ACK_BYTES + 1)
            if len(response) > MAX_ACK_BYTES or not response.endswith(b"\n"):
                return False
            reply = json.loads(response)
        except (OSError, ValueError, RecursionError):
            return False
        return (
            isinstance(reply, dict)
            and reply.keys() == {"accepted", "retry"}
            and type(reply["accepted"]) is bool
            and reply["retry"] is False
        )
