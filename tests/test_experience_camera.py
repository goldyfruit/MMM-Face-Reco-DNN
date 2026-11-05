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
