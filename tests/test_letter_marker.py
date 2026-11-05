# SPDX-License-Identifier: Apache-2.0
"""Exercise camera stamps with synthetic pixels and a local Unix socket."""

import json
from pathlib import Path
import socket
import stat
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from tools.utils.letter_marker import HUNT_ID, PROFILE_ID, LetterScanner, armed_revision


def armed(revision: int = 4) -> dict[str, Any]:
    """Build a public Hub envelope containing no household information."""
    return {
        "schema_version": 1,
        "updated_at": 90,
        "connected": True,
        "voiceState": "listening",
        "voiceUpdatedAt": 90,
        "state": {
            "schema_version": 1,
            "hunt_id": HUNT_ID,
            "revision": revision,
            "status": "paused",
            "take_armed": True,
            "profile": {"id": PROFILE_ID},
            "clue": None,
            "progress": {"found": 0, "total": 1},
        },
    }


def camera_fixture(marker: str) -> np.ndarray:
    """Place a printed-code simulation into an otherwise empty HD frame."""
    pixels = cv2.QRCodeEncoder_create().encode(marker)
    pixels = np.pad(pixels, ((3, 3), (3, 3)), constant_values=255)
    pixels = cv2.resize(pixels, (110, 110), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((1080, 1920), 240, dtype=np.uint8)
    canvas[485:595, 905:1015] = pixels
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("THALOVANT:LETTER:EN", "en"),
        ("THALOVANT:LETTER:FR", "fr"),
        ("https://example.invalid/", None),
        ("THALOVANT:LETTER:DE", None),
    ],
)
def test_only_exact_codes_decode_from_camera_frames(
    marker: str, expected: str | None
) -> None:
    """Decode synthetic printable markers and suppress arbitrary QR payloads."""
    scanner = LetterScanner()
    assert scanner._decode(camera_fixture(marker)) == expected


def test_gate_rejects_unarmed_stale_foreign_or_busy_views(tmp_path: Path) -> None:
    """Avoid opening full frames for other activities, stale states and old takes."""
    path = tmp_path / "state.json"
    baseline = armed()
    path.write_text(json.dumps(baseline))
    assert armed_revision(path, 100) == 4
    for key, value in (
        ("connected", False),
        ("updated_at", 60),
        ("voiceState", "speaking"),
        ("voiceUpdatedAt", 60),
        ("presentation", {}),
        ("media", {"video": {}}),
    ):
        path.write_text(json.dumps({**baseline, key: value}))
        assert armed_revision(path, 100) is None
    for key, value in (
        ("hunt_id", "other-hunt"),
        ("profile", {"id": "other-profile"}),
        ("status", "clue"),
        ("take_armed", False),
        ("take_armed", 1),
        ("progress", {"found": 1, "total": 1}),
        ("revision", True),
    ):
        state = {**baseline["state"], key: value}
        path.write_text(json.dumps({**baseline, "state": state}))
        assert armed_revision(path, 100) is None
    path.write_bytes(b" " * 16385)
    assert armed_revision(path, 100) is None
    link = tmp_path / "link.json"
    link.symlink_to(path)
    path.write_text(json.dumps(baseline))
    assert armed_revision(link, 100) is None


def test_scan_pacing_retry_and_latch_across_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wait for two blank armed frames before accepting a stamp on another take."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps(armed()))
    gateway = Mock(spec=Path)
    gateway.lstat.side_effect = FileNotFoundError
    scanner = LetterScanner(state, gateway)
    calls = []
    frame_calls = 0
    present = True

    def frame() -> np.ndarray:
        """Count every demand for a real camera buffer."""
        nonlocal frame_calls
        frame_calls += 1
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def send(revision: int, language: str, observed_at: float) -> bool:
        """Force a busy response once, then accept successive presentations."""
        calls.append((revision, language, observed_at))
        return len(calls) != 1

    monkeypatch.setattr(scanner, "_decode", lambda _: "en" if present else None)
    monkeypatch.setattr(scanner, "_send", send)
    scanner.scan(frame, 100, 1.0)
    assert frame_calls == 0
    gateway.lstat.side_effect = None
    gateway.lstat.return_value = SimpleNamespace(st_mode=stat.S_IFSOCK)
    for moment in (2.0, 2.1, 2.5, 3.0):
        scanner.scan(frame, 100, moment)
    assert frame_calls == 3 and calls == [(4, "en", 100), (4, "en", 100)]
    state.write_text(json.dumps(armed(revision=9)))
    scanner.scan(frame, 100, 4)
    assert len(calls) == 2
    present = False
    scanner.scan(frame, 100, 4.5)
    present = True
    scanner.scan(frame, 100, 5)
    assert len(calls) == 2
    present = False
    scanner.scan(frame, 100, 5.5)
    scanner.scan(frame, 100, 6)
    present = True
    scanner.scan(frame, 100, 6.5)
    assert calls[-1] == (9, "en", 100)


@pytest.mark.parametrize(
    ("reply", "consumed"),
    [
        (b'{"accepted":true,"retry":false}\n', True),
        (b'{"accepted":false,"retry":false}\n', True),
        (b'{"accepted":false,"retry":true}\n', False),
        (b'{"accepted":false,"retry":0}\n', False),
        (b'{"accepted":true,"retry":false}', False),
    ],
)
def test_local_socket_transmits_only_revision_language_and_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: bytes, consumed: bool
) -> None:
    """No image data, arbitrary QR text, or configurable target crosses the socket."""
    gateway = tmp_path / "letter.sock"
    handle = MagicMock()
    client = handle.__enter__.return_value
    client.recv.return_value = reply
    connect = Mock(return_value=handle)
    monkeypatch.setattr(socket, "socket", connect)
    scanner = LetterScanner(gateway=gateway)
    assert scanner._send(19, "fr", 100.5) is consumed
    connect.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout.assert_called_once_with(0.1)
    client.connect.assert_called_once_with(str(gateway))
    client.recv.assert_called_once_with(257)
    assert client.sendall.call_args.args[0].endswith(b"\n")
    assert json.loads(client.sendall.call_args.args[0]) == {
        "revision": 19,
        "language": "fr",
        "observedAt": 100.5,
    }
