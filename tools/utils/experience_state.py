# SPDX-License-Identifier: Apache-2.0
"""Lightweight public camera gates shared with the fixed Voice-side gateway."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

STATE_FILE = Path("/christmas-state/state.json")
GATEWAY = Path("/christmas-greetings/experience.sock")
PROPS = frozenset({"letter", "star", "bell", "gift"})
MARKERS = {"THALOVANT:PROP:" + prop.upper(): prop for prop in PROPS}
MAX_STATE_BYTES = 16384
MAX_REVISION = 2**53 - 1


def fresh(value: object, now: float, seconds: float = 20) -> bool:
    """Reject booleans, nonfinite numbers, old evidence and future timestamps."""
    return (
        type(value) in (int, float)
        and 0 <= value <= MAX_REVISION
        and math.isfinite(value)
        and now - seconds <= value <= now + 2
    )


@dataclass(frozen=True)
class ExperienceView:
    """Keep only the current public episode identity and explicit camera action."""

    revision: int
    identity: tuple[str, str, int, str]
    stage: str
    expected_prop: str | None
    photo_action: str
    photo_capture_revision: int
    listening: bool


def experience_view(state_file: Path, now: float) -> ExperienceView | None:
    """Read a bounded regular public file; unrelated/stale/off scenes use no pixels."""
    try:
        descriptor = os.open(state_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
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
        or not fresh(envelope.get("updated_at"), now)
        or not fresh(envelope.get("voiceUpdatedAt"), now)
        or envelope.get("voiceState")
        not in ("listening", "hearing", "thinking", "speaking")
        or envelope.get("presentation") is not None
    ):
        return None
    state = envelope.get("state")
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != 1
        or state.get("protocol_enabled") is not True
        or state.get("status") not in ("clue", "completed")
    ):
        return None
    metadata, profile = state.get("experience"), state.get("profile")
    revision, hunt = state.get("revision"), state.get("hunt_id")
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("version")) is not int
        or metadata["version"] != 1
        or not isinstance(profile, dict)
        or not isinstance(profile.get("id"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", profile["id"])
        or not isinstance(hunt, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", hunt)
        or type(revision) is not int
        or not 0 <= revision <= MAX_REVISION
        or type(metadata.get("chapter")) is not int
        or not 1 <= metadata["chapter"] <= 25
        or metadata.get("mode") not in ("daily", "family", "party", "eve", "cozy")
        or metadata.get("stage")
        not in (
            "invitation",
            "letter",
            "choice",
            "challenge",
            "hunt",
            "finale",
            "keepsake",
            "goodnight",
        )
        or metadata.get("expected_prop") not in (*PROPS, None)
        or metadata.get("photo_action", "none")
        not in ("none", "capture", "save", "discard")
        or (
            metadata.get("photo_action", "none") != "none"
            and metadata["stage"] != "keepsake"
        )
        or type(metadata.get("photo_capture_revision", 0)) is not int
        or not 0 <= metadata.get("photo_capture_revision", 0) <= revision
        or (
            metadata.get("photo_action", "none") != "none"
            and metadata.get("photo_capture_revision", 0) == 0
        )
        or (
            metadata.get("photo_action") == "capture"
            and metadata.get("photo_capture_revision") != revision
        )
    ):
        return None
    media = envelope.get("media")
    if media is not None and (
        not isinstance(media, dict) or media.get("video") is not None
    ):
        return None
    return ExperienceView(
        revision,
        (hunt, profile["id"], metadata["chapter"], metadata["mode"]),
        metadata["stage"],
        metadata.get("expected_prop"),
        metadata.get("photo_action", "none"),
        metadata.get("photo_capture_revision", 0),
        envelope.get("voiceState") == "listening",
    )
