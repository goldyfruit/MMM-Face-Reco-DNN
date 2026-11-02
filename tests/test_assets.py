# SPDX-License-Identifier: Apache-2.0
"""Verify that shipped public assets match their source manifests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_detector_matches_official_model_digest() -> None:
    """Do not ship an LFS pointer, corrupted weights or an unrecorded model."""
    manifest = json.loads((ROOT/'model/yunet-manifest.json').read_text())
    data = (ROOT/'model'/manifest['file']).read_bytes()
    assert len(data) > 200000
    assert hashlib.sha256(data).hexdigest() == manifest['sha256']


def test_public_fixture_matches_its_provenance() -> None:
    """Keep test imagery reproducible and independent of any user gallery."""
    manifest = json.loads((ROOT/'tests/fixtures/provenance.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((ROOT/'tests/fixtures'/name).read_bytes()).hexdigest() == entry['sha256']
