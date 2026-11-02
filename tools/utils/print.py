"""Atomic JSON-line events consumed by MagicMirror's PythonShell."""

import json
import sys
import threading
from typing import Any


class Print:
    """Serialize events without interleaving worker-thread output."""

    _lock = threading.Lock()

    @staticmethod
    def printJson(event_type: str, message: Any) -> None:
        """Write one complete, finite JSON event and flush immediately."""
        encoded = json.dumps({event_type: message}, allow_nan=False)
        with Print._lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()
