# SPDX-License-Identifier: Apache-2.0
"""Admit fixed local prop observations through the owner-authorized Voice client."""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.experience_state import (
    GATEWAY,
    MAX_REVISION,
    PROPS,
    STATE_FILE,
    experience_view,
    fresh,
)


class ExperienceGateway:
    """One bounded prop queue; received data can never choose an arbitrary action."""

    def __init__(
        self,
        state_file: Path = STATE_FILE,
        *,
        client_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time,
        submit: Callable[[Callable[[], None]], Any] | None = None,
    ) -> None:
        """Delay Voice credentials and worker creation until an admitted observation."""
        self.state_file, self.clock = state_file, clock
        self.client_factory = client_factory or self._client
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="christmas-prop")
            if submit is None
            else None
        )
        self._submit = submit or self._executor.submit
        self._lock = threading.Lock()
        self._seen: dict[tuple[Any, ...], None] = {}
        self._pending = False
        self._closed = False

    @staticmethod
    def _client() -> Any:
        """Use the existing local owner endpoint, never caller-supplied credentials."""
        from thalovant_voice.control import ControlClient

        return ControlClient.from_file()

    def process(self, value: Any) -> dict[str, bool]:
        """Acknowledge a fresh observation before any possibly slow Voice work."""
        denied = {"accepted": False, "retry": False}
        now = self.clock()
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "revision", "prop", "observed_at"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or type(value["revision"]) is not int
            or not 0 <= value["revision"] <= MAX_REVISION
            or not isinstance(value["prop"], str)
            or value["prop"] not in PROPS
            or not fresh(value["observed_at"], now, 3)
        ):
            return denied
        view = experience_view(self.state_file, now)
        if (
            view is None
            or not view.listening
            or view.revision != value["revision"]
            or view.expected_prop != value["prop"]
        ):
            return denied
        key = (*view.identity, view.revision, value["prop"])
        prop = value["prop"]
        with self._lock:
            if self._closed:
                return denied
            if key in self._seen:
                return {"accepted": True, "retry": False}
            if self._pending:
                return {"accepted": False, "retry": True}
            self._pending = True
            self._seen[key] = None
            while len(self._seen) > 128:
                self._seen.pop(next(iter(self._seen)))

        def dispatch() -> None:
            """Recheck selection before invoking the sole fixed owner-authorized route."""
            succeeded = False
            try:
                with self._lock:
                    if self._closed:
                        return
                current = experience_view(self.state_file, self.clock())
                if (
                    current is None
                    or current.identity != view.identity
                    or current.revision != view.revision
                    or not current.listening
                    or current.expected_prop != prop
                ):
                    return
                reply = self.client_factory().call(
                    "POST",
                    "/v1/ask",
                    {
                        "text": f"Christmas prop {prop} revision {view.revision}",
                        "lang": "en-US",
                        "speak": True,
                    },
                    timeout=5.0,
                )
                succeeded = isinstance(reply, dict) and reply.get("ok") is True
            except (
                ImportError,
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                TimeoutError,
            ):
                # Endpoint errors can contain credentials or transcripts; do not log them.
                pass
            finally:
                with self._lock:
                    self._pending = False
                    if not succeeded:
                        self._seen.pop(key, None)

        try:
            self._submit(dispatch)
        except RuntimeError:
            with self._lock:
                self._pending = False
                self._seen.pop(key, None)
            return denied
        return {"accepted": True, "retry": False}

    def close(self) -> None:
        """Refuse new observations and cancel queued work on service shutdown."""
        with self._lock:
            self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


def serve(gateway: ExperienceGateway, path: Path = GATEWAY) -> None:
    """Bind a private Unix socket; do not replace a running or unowned endpoint."""
    if not path.is_absolute() or any(
        parent.is_symlink() for parent in (path.parent, *path.parents)
    ):
        raise ValueError("A real absolute socket directory is required")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Refusing to replace a non-owned socket")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.1)
            try:
                probe.connect(str(path))
            except (ConnectionRefusedError, FileNotFoundError):
                path.unlink()
            else:
                raise ValueError("Experience gateway already running")

    class Handler(socketserver.StreamRequestHandler):
        """Read one tiny metadata observation and publish a tiny admission result."""

        def handle(self) -> None:
            """Bound input time and bytes independently of the Voice request."""
            self.request.settimeout(1)
            try:
                raw = self.rfile.readline(513)
                value = (
                    json.loads(raw) if len(raw) <= 512 and raw.endswith(b"\n") else None
                )
                reply = gateway.process(value)
                self.wfile.write(json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
            except (OSError, ValueError, RecursionError):
                return

    try:
        with socketserver.UnixStreamServer(str(path), Handler) as server:
            path.chmod(0o600)
            server.serve_forever(poll_interval=0.2)
    finally:
        gateway.close()
        if path.exists() and stat.S_ISSOCK(path.lstat().st_mode):
            path.unlink()


def main() -> None:
    """Run explicitly beside the existing Voice owner, not from the display helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--socket", type=Path, default=GATEWAY)
    args = parser.parse_args()
    try:
        serve(ExperienceGateway(args.state_file), args.socket)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
