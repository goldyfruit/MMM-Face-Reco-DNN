# SPDX-License-Identifier: Apache-2.0
"""Test fixed props and requested keepsakes with synthetic pixels only."""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tools.experience_gateway import ExperienceGateway
from tools.utils.experience_camera import ExperiencePropScanner, PhotoBooth
from tools.utils.experience_state import experience_view


def write_state(
    path: Path,
    *,
    revision: int = 4,
    stage: str = "letter",
    prop: str | None = "letter",
    photo: str = "none",
    voice: str = "listening",
    now: float = 100,
    capture_revision: int = 4,
    style: str | None = None,
    lang: str = "en-US",
) -> dict:
    """Create only synthetic public episode data and fresh Voice evidence."""
    value = {
        "schema_version": 1,
        "updated_at": now,
        "connected": True,
        "voiceState": voice,
        "voiceUpdatedAt": now,
        "presentation": None,
        "state": {
            "schema_version": 1,
            "hunt_id": "fictional-episode",
            "status": "clue",
            "protocol_enabled": True,
            "revision": revision,
            "lang": lang,
            "profile": {"id": "experience-family", "display_name": "Family"},
            "experience": {
                "version": 1,
                "chapter": 1,
                "mode": "family",
                "stage": stage,
                "expected_prop": prop,
                "photo_action": photo,
                "photo_capture_revision": capture_revision
                if stage == "keepsake"
                else 0,
            },
        },
    }
    if style is not None:
        value["state"]["experience"]["photo_style"] = style
    path.write_text(json.dumps(value))
    return value


def pixels(marker: str | None = None) -> np.ndarray:
    """Build an original plain synthetic frame, optionally containing a fixed stamp."""
    image = np.full((720, 1280, 3), (40, 100, 160), dtype=np.uint8)
    if marker:
        code = cv2.QRCodeEncoder_create().encode(marker)
        code = np.pad(code, ((4, 4), (4, 4)), constant_values=255)
        code = cv2.resize(code, (200, 200), interpolation=cv2.INTER_NEAREST)
        image[260:460, 540:740] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    return image


@pytest.mark.parametrize("prop", ["letter", "star", "bell", "gift"])
def test_fixed_printed_props_decode_and_arbitrary_codes_do_not(prop: str) -> None:
    """The decoder returns only one opaque catalog ID, never arbitrary QR text."""
    scanner = ExperiencePropScanner()
    assert scanner._decode(pixels("THALOVANT:PROP:" + prop.upper())) == prop
    assert scanner._decode(pixels("https://example.invalid/")) is None
    assert scanner._decode(pixels("THALOVANT:PROP:CAMERA")) is None


@pytest.mark.parametrize(
    "field,bad",
    [
        ("protocol_enabled", False),
        ("status", "paused"),
        ("revision", True),
        ("revision", 2**53),
        ("experience", None),
        ("profile", {}),
    ],
)
def test_off_unrelated_or_malformed_state_never_borrows_pixels(
    tmp_path: Path, field: str, bad: object
) -> None:
    """Camera activity is gated before lazily materializing the full camera frame."""
    path = tmp_path / "state.json"
    value = write_state(path)
    value["state"][field] = bad
    path.write_text(json.dumps(value))
    frame = Mock(side_effect=AssertionError("unexpected pixels"))
    scanner = ExperiencePropScanner(path, tmp_path / "missing.sock")
    scanner.scan(frame, 100, 0)
    PhotoBooth(Mock(), path, tmp_path / "photos").process_frame(frame, 100, 0)
    assert not frame.called


@pytest.mark.parametrize(
    "field,bad",
    [
        ("mode", []),
        ("stage", {}),
        ("expected_prop", []),
        ("photo_action", {}),
        ("chapter", True),
        ("version", True),
        ("photo_action", "upload"),
        ("photo_style", "upload"),
        ("photo_style", None),
        ("photo_style", []),
    ],
)
def test_invalid_nested_metadata_fails_quietly(
    tmp_path: Path, field: str, bad: object
) -> None:
    """Malformed JSON shapes cannot crash the camera worker or enable a feature."""
    path = tmp_path / "state.json"
    value = write_state(path)
    value["state"]["experience"][field] = bad
    path.write_text(json.dumps(value))
    assert experience_view(path, 100) is None


def test_gate_rejects_symlinks_stale_voice_oversize_and_presentations(
    tmp_path: Path,
) -> None:
    """Neither a retained card nor an unrelated provider can authorize camera pixels."""
    path = tmp_path / "state.json"
    value = write_state(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert experience_view(link, 100) is None
    assert experience_view(path, 121) is None
    value["voiceUpdatedAt"] = True
    path.write_text(json.dumps(value))
    assert experience_view(path, 100) is None
    value["voiceUpdatedAt"] = 10**400
    path.write_text(json.dumps(value))
    assert experience_view(path, 100) is None
    value["voiceUpdatedAt"] = 100
    value["presentation"] = {"kind": "riddle"}
    path.write_text(json.dumps(value))
    assert experience_view(path, 100) is None
    path.write_text(" " * 17000)
    assert experience_view(path, 100) is None


def test_prop_scanner_uses_current_revision_once_until_prop_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding a stamp over multiple polls is one physical presentation."""
    path = tmp_path / "state.json"
    write_state(path)
    gateway = SimpleNamespace(lstat=lambda: SimpleNamespace(st_mode=stat.S_IFSOCK))
    scanner = ExperiencePropScanner(path, gateway)
    send = Mock(return_value=True)
    monkeypatch.setattr(scanner, "_send", send)
    frame = Mock(return_value=pixels("THALOVANT:PROP:LETTER"))
    scanner.scan(frame, 100, 0)
    scanner.scan(frame, 100.1, 0.1)
    scanner.scan(frame, 101, 1)
    assert send.call_args.args == (4, "letter", 100)
    assert send.call_count == 1 and frame.call_count == 2
    scanner.scan(lambda: pixels(), 102, 2)
    scanner.scan(lambda: pixels(), 103, 3)
    scanner.scan(frame, 104, 4)
    assert send.call_count == 2
    write_state(path, revision=5, prop="star")
    scanner.scan(frame, 105, 5)
    assert send.call_count == 2


def booth_fixture(tmp_path: Path, *, voice: str = "listening") -> SimpleNamespace:
    """Prepare a requested keepsake with an isolated private output path."""
    path = tmp_path / "state.json"
    write_state(path, stage="keepsake", prop=None, photo="capture", voice=voice)
    events = []
    booth = PhotoBooth(
        lambda kind, value: events.append((kind, value)), path, tmp_path / "photos"
    )
    frame = Mock(return_value=pixels())
    # A normally running camera observes the ready scene before a new request.
    write_state(
        path,
        revision=3,
        stage="keepsake",
        prop=None,
        photo="none",
        capture_revision=0,
        voice=voice,
    )
    booth.process_frame(frame, 100, 0)
    write_state(path, stage="keepsake", prop=None, photo="capture", voice=voice)
    return SimpleNamespace(
        path=path,
        events=events,
        booth=booth,
        frame=frame,
        directory=tmp_path / "photos",
    )


def capture(fixture: SimpleNamespace) -> None:
    """Complete the three-second countdown without sleeping or camera access."""
    for second in range(4):
        fixture.booth.process_frame(fixture.frame, 100 + second, second)


def test_photo_countdown_preview_in_memory_and_explicit_private_save(
    tmp_path: Path,
) -> None:
    """No file exists until save; one unannotated bounded JPEG is then committed."""
    f = booth_fixture(tmp_path)
    capture(f)
    assert [
        value["seconds"] for _, value in f.events if value["status"] == "countdown"
    ] == [3, 2, 1]
    assert f.frame.call_count == 1 and f.booth.preview is not None
    assert not f.directory.exists()
    preview = f.events[-1][1]
    assert preview["status"] == "preview" and len(json.dumps(preview)) < 700000
    encoded = base64.b64decode(preview["image"])
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] <= 1200 and decoded.shape[0] <= 1200
    f.booth.process_frame(f.frame, 105, 5)
    assert f.frame.call_count == 1
    write_state(f.path, revision=5, stage="keepsake", prop=None, photo="save", now=105)
    f.booth.process_frame(f.frame, 105, 5)
    saved = list(f.directory.iterdir())
    assert len(saved) == 1 and saved[0].read_bytes() == encoded
    assert saved[0].stat().st_mode & 0o777 == 0o600
    assert f.directory.stat().st_mode & 0o777 == 0o700
    assert f.events[-1][1] == {
        "version": 1,
        "revision": 5,
        "capture_revision": 4,
        "status": "saved",
    }
    assert f.booth.preview is None
    f.booth.process_frame(f.frame, 106, 6)
    assert len(list(f.directory.iterdir())) == 1


def test_worker_restart_consumes_persisted_capture_until_new_explicit_retake(
    tmp_path: Path,
) -> None:
    """A new worker cannot replay an old capture just because its heartbeat is fresh."""
    f = booth_fixture(tmp_path)
    restarted = PhotoBooth(
        lambda kind, value: f.events.append((kind, value)), f.path, f.directory
    )
    for second in range(4):
        restarted.process_frame(f.frame, 100 + second, second)
    assert f.events[-1][1] == {
        "version": 1,
        "revision": 4,
        "capture_revision": 4,
        "status": "unavailable",
    }
    assert len(f.events) == 1 and not f.frame.called and restarted.preview is None
    assert not f.directory.exists()
    restarted.reset()
    restarted.process_frame(f.frame, 104, 4)
    assert not f.frame.called
    write_state(
        f.path,
        revision=5,
        capture_revision=5,
        stage="keepsake",
        prop=None,
        photo="capture",
        now=105,
    )
    for second in range(4):
        restarted.process_frame(f.frame, 105 + second, 5 + second)
    assert f.frame.call_count == 1 and restarted.preview_capture_revision == 5
    assert f.events[-1][1]["status"] == "preview"


@pytest.mark.parametrize("ready", ["none", "other-scene"])
def test_ready_observation_arms_a_later_explicit_capture(
    tmp_path: Path, ready: str
) -> None:
    """A valid ready scene arms the next new photo turn before any pixels are read."""
    f = booth_fixture(tmp_path)
    restarted = PhotoBooth(
        lambda kind, value: f.events.append((kind, value)), f.path, f.directory
    )
    if ready == "other-scene":
        write_state(
            f.path, revision=3, stage="letter", photo="none", capture_revision=0
        )
    else:
        write_state(
            f.path,
            revision=3,
            stage="keepsake",
            prop=None,
            photo="none",
            capture_revision=0,
        )
    restarted.process_frame(f.frame, 100, 0)
    assert not f.frame.called
    write_state(
        f.path, revision=4, stage="keepsake", prop=None, photo="capture", now=101
    )
    for second in range(4):
        restarted.process_frame(f.frame, 101 + second, 1 + second)
    assert f.frame.call_count == 1 and restarted.preview_capture_revision == 4
    assert f.events[-1][1]["status"] == "preview"


@pytest.mark.parametrize(
    "unready",
    [
        "missing",
        "malformed",
        "disconnected",
        "off",
        "hidden",
        "stale",
        "stale-voice",
        "missing-state",
    ],
)
def test_unready_startup_cannot_arm_persisted_capture_after_reconnection(
    tmp_path: Path, unready: str
) -> None:
    """Unusable/off input is not camera readiness or fresh permission to take pixels."""
    f = booth_fixture(tmp_path)
    restarted = PhotoBooth(
        lambda kind, value: f.events.append((kind, value)), f.path, f.directory
    )
    value = write_state(
        f.path, stage="keepsake", prop=None, photo="capture", style="postcard"
    )
    if unready == "missing":
        f.path.unlink()
    elif unready == "malformed":
        f.path.write_text("{")
    else:
        if unready == "disconnected":
            value["connected"] = False
        elif unready == "off":
            value["state"]["protocol_enabled"] = False
        elif unready == "hidden":
            value["presentation"] = {"provider": "christmas-games"}
        elif unready == "stale":
            value["updated_at"] = 0
        elif unready == "stale-voice":
            value["voiceUpdatedAt"] = 0
        elif unready == "missing-state":
            value["state"] = None
        f.path.write_text(json.dumps(value))
    restarted.process_frame(f.frame, 100, 0)
    assert not f.frame.called
    write_state(
        f.path, stage="keepsake", prop=None, photo="capture", style="postcard", now=101
    )
    for second in range(4):
        restarted.process_frame(f.frame, 101 + second, 1 + second)
    assert not f.frame.called and restarted.preview is None
    assert f.events[-1][1]["status"] == "unavailable"
    write_state(
        f.path,
        revision=5,
        capture_revision=5,
        stage="keepsake",
        prop=None,
        photo="capture",
        style="postcard",
        now=106,
    )
    for second in range(4):
        restarted.process_frame(f.frame, 106 + second, 6 + second)
    assert f.frame.call_count == 1 and restarted.preview_capture_revision == 5
    assert f.events[-1][1]["status"] == "preview"


@pytest.mark.parametrize("lang", ["en-US", "fr-FR", "en-CA", "fr-CA"])
def test_opt_in_postcard_is_frozen_preview_and_exact_explicit_save(
    tmp_path: Path, lang: str
) -> None:
    """One borrowed photo is framed once; saving commits its identical preview bytes."""
    f = booth_fixture(tmp_path)
    write_state(
        f.path,
        stage="keepsake",
        prop=None,
        photo="capture",
        style="postcard",
        lang=lang,
    )
    original = f.frame.return_value.copy()
    capture(f)
    encoded = base64.b64decode(f.events[-1][1]["image"])
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    assert f.frame.call_count == 1
    assert np.array_equal(f.frame.return_value, original)
    assert decoded.shape == (808, 1160, 3)
    assert f.booth.preview_style == "postcard"
    assert not f.directory.exists()
    write_state(
        f.path,
        revision=5,
        stage="keepsake",
        prop=None,
        photo="save",
        now=105,
        style="postcard",
        lang=lang,
    )
    f.booth.process_frame(f.frame, 105, 5)
    assert f.frame.call_count == 1
    assert list(f.directory.iterdir())[0].read_bytes() == encoded
    assert f.events[-1][1]["status"] == "saved"


def test_style_cannot_reframe_old_capture_or_save_mismatched_preview(
    tmp_path: Path,
) -> None:
    """A different style needs a new capture revision, never posthoc editing on save."""
    f = booth_fixture(tmp_path)
    capture(f)
    write_state(
        f.path,
        revision=5,
        stage="keepsake",
        prop=None,
        photo="save",
        now=105,
        style="postcard",
    )
    f.booth.process_frame(f.frame, 105, 5)
    assert f.events[-1][1]["status"] == "unavailable"
    assert not f.directory.exists() and not f.booth.preview
    assert f.frame.call_count == 1


def test_style_change_during_countdown_requires_new_capture_revision(
    tmp_path: Path,
) -> None:
    """A same-revision style mutation cannot turn plain-photo consent into a postcard."""
    f = booth_fixture(tmp_path)
    f.booth.process_frame(f.frame, 100, 0)
    write_state(f.path, stage="keepsake", prop=None, photo="capture", style="postcard")
    f.booth.process_frame(f.frame, 103, 3)
    f.booth.process_frame(f.frame, 104, 4)
    assert not f.frame.called and not f.booth.preview
    write_state(
        f.path,
        revision=5,
        capture_revision=5,
        stage="keepsake",
        prop=None,
        photo="capture",
        now=105,
        style="postcard",
    )
    f.booth.process_frame(f.frame, 105, 5)
    f.booth.process_frame(f.frame, 108, 8)
    assert f.frame.call_count == 1 and f.booth.preview_style == "postcard"


def test_style_change_while_compositing_discards_inflight_pixels(
    tmp_path: Path,
) -> None:
    """The final source check binds the composite to the still-current opted-in style."""
    f = booth_fixture(tmp_path)
    write_state(f.path, stage="keepsake", prop=None, photo="capture", style="postcard")
    f.booth.process_frame(f.frame, 100, 0)

    def change_style() -> np.ndarray:
        """Simulate a newer conflicting source arriving during the capture callback."""
        write_state(
            f.path, stage="keepsake", prop=None, photo="capture", style="plain", now=103
        )
        return pixels()

    f.booth.process_frame(change_style, 103, 3)
    assert not f.booth.preview and not f.directory.exists()


def test_invalid_postcard_timezone_fails_photo_without_worker_exception(
    tmp_path: Path,
) -> None:
    """Missing IANA data cannot silently stamp a wrong date or save a replacement."""
    f = booth_fixture(tmp_path)
    f.booth.timezone = "Mars/NorthPole"
    write_state(f.path, stage="keepsake", prop=None, photo="capture", style="postcard")
    capture(f)
    assert f.events[-1][1]["status"] == "unavailable"
    assert not f.booth.preview and not f.directory.exists()


def test_postcard_to_plain_retake_uses_new_revision_and_original_photo_shape(
    tmp_path: Path,
) -> None:
    """Switching style needs a new explicit capture, never an edit of old bytes."""
    f = booth_fixture(tmp_path)
    write_state(f.path, stage="keepsake", prop=None, photo="capture", style="postcard")
    capture(f)
    postcard_bytes = f.booth.preview
    write_state(
        f.path,
        revision=5,
        capture_revision=5,
        stage="keepsake",
        prop=None,
        photo="capture",
        style="plain",
        now=105,
    )
    f.booth.process_frame(f.frame, 105, 5)
    assert f.booth.preview is None
    f.booth.process_frame(f.frame, 108, 8)
    assert f.frame.call_count == 2 and f.booth.preview_style == "plain"
    assert f.booth.preview != postcard_bytes
    decoded = cv2.imdecode(np.frombuffer(f.booth.preview, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (675, 1200, 3)
    ok, expected = cv2.imencode(
        ".jpg",
        cv2.resize(f.frame.return_value, (1200, 675)),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    assert ok and f.booth.preview == expected.tobytes()


def test_legacy_style_defaults_plain_and_invalid_language_is_closed(
    tmp_path: Path,
) -> None:
    """Older producer states retain plain behavior; malformed locale cannot drive text."""
    f = booth_fixture(tmp_path)
    view = experience_view(f.path, 100)
    assert view and view.photo_style == "plain" and view.lang == "en-US"
    value = json.loads(f.path.read_text())
    del value["state"]["lang"]
    f.path.write_text(json.dumps(value))
    assert experience_view(f.path, 100).lang == "en-US"
    value["state"]["lang"] = ["fr-FR"]
    f.path.write_text(json.dumps(value))
    assert experience_view(f.path, 100) is None


def test_photo_waits_for_listening_and_does_not_capture_over_speech(
    tmp_path: Path,
) -> None:
    """The countdown begins after Voice is fresh and quiet, not during Santa speech."""
    f = booth_fixture(tmp_path, voice="speaking")
    f.booth.process_frame(f.frame, 100, 0)
    f.booth.process_frame(f.frame, 104, 4)
    assert not f.frame.called and not f.events
    write_state(f.path, stage="keepsake", prop=None, photo="capture", now=104)
    f.booth.process_frame(f.frame, 104, 4)
    f.booth.process_frame(f.frame, 106, 6)
    assert not f.frame.called
    f.booth.process_frame(f.frame, 107, 7)
    assert f.frame.call_count == 1 and f.events[-1][1]["status"] == "preview"


@pytest.mark.parametrize("action", ["save", "discard", "none"])
def test_photo_never_captures_without_explicit_capture(
    tmp_path: Path, action: str
) -> None:
    """A save after worker restart cannot recover an absent preview or take a photo."""
    f = booth_fixture(tmp_path)
    write_state(f.path, stage="keepsake", prop=None, photo=action)
    f.booth.process_frame(f.frame, 100, 0)
    f.booth.process_frame(f.frame, 104, 4)
    assert not f.frame.called and not f.directory.exists()
    assert f.booth.preview is None
    if action == "save":
        assert f.events[-1][1]["status"] == "unavailable"


def test_discard_exit_stale_disable_and_replayed_capture_drop_unsaved_pixels(
    tmp_path: Path,
) -> None:
    """Expiry/off cannot revive a previously consumed capture command."""
    f = booth_fixture(tmp_path)
    capture(f)
    f.booth.process_frame(f.frame, 130, 30)
    assert f.booth.preview is None and not f.directory.exists()
    write_state(f.path, stage="keepsake", prop=None, photo="capture", now=130)
    f.booth.process_frame(f.frame, 130, 30)
    f.booth.process_frame(f.frame, 134, 34)
    assert f.frame.call_count == 1
    write_state(
        f.path,
        revision=5,
        stage="keepsake",
        prop=None,
        photo="capture",
        now=135,
        capture_revision=5,
    )
    for second in range(4):
        f.booth.process_frame(f.frame, 135 + second, 35 + second)
    assert f.frame.call_count == 2 and f.booth.preview
    write_state(
        f.path,
        revision=6,
        stage="keepsake",
        prop=None,
        photo="discard",
        now=140,
        capture_revision=5,
    )
    f.booth.process_frame(f.frame, 140, 40)
    assert not f.booth.preview and f.events[-1][1]["status"] == "discarded"
    assert not f.directory.exists()


def test_new_episode_cannot_save_old_preview_and_generic_revision_republishes_preview(
    tmp_path: Path,
) -> None:
    """A repeat keeps the same preview; another chapter does not inherit its pixels."""
    f = booth_fixture(tmp_path)
    capture(f)
    write_state(f.path, revision=5, stage="keepsake", prop=None, photo="none", now=104)
    f.booth.process_frame(f.frame, 104, 4)
    assert f.events[-1][1]["status"] == "preview" and f.events[-1][1]["revision"] == 5
    value = write_state(
        f.path, revision=6, stage="keepsake", prop=None, photo="save", now=105
    )
    value["state"]["experience"]["chapter"] = 2
    f.path.write_text(json.dumps(value))
    f.booth.process_frame(f.frame, 105, 5)
    assert not f.booth.preview and not f.directory.exists()
    assert f.events[-1][1]["status"] == "unavailable"


def test_late_state_change_during_capture_discards_encoded_image(
    tmp_path: Path,
) -> None:
    """A scene that exits during full-frame materialization receives no late preview."""
    f = booth_fixture(tmp_path)
    f.booth.process_frame(f.frame, 100, 0)

    def late_frame() -> np.ndarray:
        value = write_state(f.path, stage="keepsake", prop=None, photo="capture")
        value["state"]["protocol_enabled"] = False
        f.path.write_text(json.dumps(value))
        return pixels()

    f.booth.process_frame(late_frame, 103, 3)
    assert not f.booth.preview
    assert not any(value["status"] == "preview" for _, value in f.events)
    assert not f.directory.exists()


def test_photo_directory_symlink_or_shared_permissions_are_rejected(
    tmp_path: Path,
) -> None:
    """Saving cannot follow a configured link into a gallery or overwrite existing files."""
    f = booth_fixture(tmp_path)
    capture(f)
    target = tmp_path / "gallery"
    target.mkdir()
    f.directory.symlink_to(target, target_is_directory=True)
    write_state(f.path, revision=5, stage="keepsake", prop=None, photo="save", now=104)
    f.booth.process_frame(f.frame, 104, 4)
    assert not list(target.iterdir()) and f.events[-1][1]["status"] == "unavailable"
    f.directory.unlink()
    f.directory.mkdir(mode=0o755)
    f.directory.chmod(0o755)
    write_state(f.path, revision=6, stage="keepsake", prop=None, photo="save", now=105)
    f.booth.process_frame(f.frame, 105, 5)
    assert (
        not list(f.directory.iterdir()) and f.events[-1][1]["status"] == "unavailable"
    )


def test_gateway_dispatches_only_fixed_expected_revision_bound_command(
    tmp_path: Path,
) -> None:
    """Admission acknowledges quickly; a sole queued owner call performs the action."""
    path = tmp_path / "state.json"
    write_state(path)
    tasks = []
    call = Mock(return_value={"ok": True})
    gateway = ExperienceGateway(
        path,
        client_factory=lambda: SimpleNamespace(call=call),
        clock=lambda: 100,
        submit=tasks.append,
    )
    payload = {"version": 1, "revision": 4, "prop": "letter", "observed_at": 100}
    assert gateway.process(payload) == {"accepted": True, "retry": False}
    assert not call.called
    tasks.pop()()
    call.assert_called_once_with(
        "POST",
        "/v1/ask",
        {"text": "Christmas prop letter revision 4", "lang": "en-US", "speak": True},
        timeout=5.0,
    )
    assert gateway.process(payload) == {"accepted": True, "retry": False}
    assert not tasks
    gateway.close()


@pytest.mark.parametrize(
    "patch",
    [
        {"prop": "https://example.invalid"},
        {"prop": "gift"},
        {"prop": []},
        {"revision": 3},
        {"revision": True},
        {"version": True},
        {"observed_at": 96},
        {"observed_at": 103},
        {"observed_at": float("nan")},
        {"text": "arbitrary command"},
        {"actor": "other"},
        {"image": "pixels"},
    ],
)
def test_gateway_rejects_wrong_stale_forged_or_extended_observations(
    tmp_path: Path, patch: dict
) -> None:
    """The observation cannot supply speech, actors, routes, credentials or images."""
    path = tmp_path / "state.json"
    write_state(path)
    tasks = []
    gateway = ExperienceGateway(path, submit=tasks.append, clock=lambda: 100)
    payload = {"version": 1, "revision": 4, "prop": "letter", "observed_at": 100}
    assert gateway.process(payload | patch) == {"accepted": False, "retry": False}
    assert not tasks
    gateway.close()


def test_gateway_rechecks_state_before_late_dispatch_and_bounds_pending_work(
    tmp_path: Path,
) -> None:
    """A queued prop cannot affect a newer chapter or build an unbounded Voice queue."""
    path = tmp_path / "state.json"
    write_state(path)
    tasks = []
    call = Mock(return_value={"ok": True})
    gateway = ExperienceGateway(
        path,
        client_factory=lambda: SimpleNamespace(call=call),
        clock=lambda: 100,
        submit=tasks.append,
    )
    assert gateway.process(
        {"version": 1, "revision": 4, "prop": "letter", "observed_at": 100}
    )["accepted"]
    write_state(path, revision=5)
    assert gateway.process(
        {"version": 1, "revision": 5, "prop": "letter", "observed_at": 100}
    ) == {"accepted": False, "retry": True}
    tasks.pop()()
    assert not call.called
    assert gateway.process(
        {"version": 1, "revision": 5, "prop": "letter", "observed_at": 100}
    )["accepted"]
    gateway.close()
    tasks.pop()()
    assert not call.called


def test_skipped_retake_cannot_save_an_older_preview(tmp_path: Path) -> None:
    """A save for capture 11 cannot commit capture 4 when polling skipped the retake."""
    f = booth_fixture(tmp_path)
    capture(f)
    assert f.booth.preview_capture_revision == 4
    write_state(
        f.path,
        revision=12,
        stage="keepsake",
        prop=None,
        photo="save",
        now=104,
        capture_revision=11,
    )
    f.booth.process_frame(f.frame, 104, 4)
    assert f.booth.preview is None and not f.directory.exists()
    assert f.frame.call_count == 1
    assert f.events[-1][1] == {
        "version": 1,
        "revision": 12,
        "capture_revision": 11,
        "status": "unavailable",
    }


def test_direct_letter_to_star_transition_does_not_inherit_previous_prop_latch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family can exchange the two expected props without a blank camera frame."""
    path = tmp_path / "state.json"
    write_state(path)
    gateway = SimpleNamespace(lstat=lambda: SimpleNamespace(st_mode=stat.S_IFSOCK))
    scanner = ExperiencePropScanner(path, gateway)
    send = Mock(return_value=True)
    monkeypatch.setattr(scanner, "_send", send)
    scanner.scan(lambda: pixels("THALOVANT:PROP:LETTER"), 100, 0)
    write_state(path, revision=5, prop="star")
    scanner.scan(lambda: pixels("THALOVANT:PROP:STAR"), 101, 1)
    assert send.call_count == 2
    assert send.call_args.args == (5, "star", 101)
    write_state(path, revision=6, prop="star")
    scanner.scan(lambda: pixels("THALOVANT:PROP:STAR"), 102, 2)
    assert send.call_count == 2


def test_gateway_import_needs_no_camera_or_numpy_dependency() -> None:
    """The Voice-side process imports only lightweight state validation."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['cv2']=None; sys.modules['numpy']=None; import tools.experience_gateway",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("bad", [True, -1, 5, [], "4"])
def test_photo_capture_target_must_be_a_matching_safe_revision(
    tmp_path: Path, bad: object
) -> None:
    """Capture can only target its own native action revision, never a loose flag."""
    path = tmp_path / "state.json"
    value = write_state(path, stage="keepsake", photo="capture", prop=None)
    value["state"]["experience"]["photo_capture_revision"] = bad
    path.write_text(json.dumps(value))
    assert experience_view(path, 100) is None
