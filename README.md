Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Add opt-in dated North Pole postcards with unchanged explicit photo consent.

# MMM-Face-Reco-DNN — CPU/ISP performance fork

This fork preserves existing dlib128 face galleries while adding an optional lightweight YuNet detector and a synchronized Picamera2 low-resolution processing stream. It also fixes per-face confidence labels, annotated enrollment images and ambiguous multi-face enrollment.

## Quick configuration

```javascript
{
  module: 'MMM-Face-Reco-DNN',
  config: {
    pythonPath: '/path/to/existing/face-venv/bin/python3',
    method: 'yunet',
    yunetScoreThreshold: 0.8,
    cameraLores: true,
    cameraLoresFormat: 'RGB888',
    cameraFrameRate: 15,
    cameraAutofocus: 'default',
    cameraExposureMode: 'normal',
    opencvThreads: 2,
    checkInterval: 500,
    processWidth: 500,
    resolution: [1920, 1080]
  }
}
```

Existing settings, HOG/CNN/Haar paths and gallery format remain supported. New camera/model choices are opt-in; default method remains the legacy `dnn`/HOG path. CPU face descriptors are not GPU accelerated by this change.

See [installation and settings](docs/index.md), [measured benchmark and limits](docs/performance.md), [quick facts](QUICK_FACTS.md) and [audit](AUDIT.md).

## Validation

Run `python -m pytest tests --cov=tools`, `npm test` and `npm run lint`. Tests use synthetic arrays and one public-domain NASA image. No user face images or embeddings belong in this repository.

## Licenses

Original upstream code remains MIT (`LICENSE`). New independently authored helpers/tests marked Apache-2.0 use `LICENSE-APACHE-2.0`. The bundled official YuNet model is MIT (`model/LICENSE.YuNet`); its source revision and SHA256 are recorded in `model/yunet-manifest.json`. NASA fixture provenance is in `tests/fixtures/provenance.json`.

Upstream: https://github.com/nischi/MMM-Face-Reco-DNN .

## Camera control and worker recovery

Version 2.2.0 preserves existing focus by default (`cameraAutofocus:'default'`). Optional `'continuous'` requests continuous autofocus/normal range when advertised. Glass or foreground objects can attract autofocus, so verify it at the actual standing distance. Optional `'manual'` accepts `cameraLensPosition` in dioptres only when the camera advertises manual focus and the requested position is in range; `1.0` is approximately one metre and `0` is infinity. No position is forced by default. Normal exposure remains the default; optional `'short'` is a room-dependent motion/noise tradeoff, not a guaranteed accuracy gain. [CameraStream](tools/utils/camera.py) exposes bounded numeric metadata for local diagnosis without a new server or retained images.

If the Python worker fails or closes, the mirror immediately clears stale face sessions, logout timers and preview; this bypasses the normal disappearance delay. Recovery waits for the old process to close, then attempts at most3 early retries with1/2/4second delays by default. Explicit stop/pause is respected; queued activity from a paused or closing worker is discarded. Plain stdin commands are sent with the installed PythonShell text formatter while stdout remains JSON. [The Python worker](tools/recognition.py) reports readiness only after a processed capture request was released, including after resume when no face or preview is present. See [configuration](docs/index.md), [audit](AUDIT.md) and [tests](tests/test_node_helper.cjs).

Optional `cameraAutofocus:'face'` targets autofocus at detected face windows instead of the camera's default metering region in a reflective scene. It begins at a supported manual seed (`cameraLensPosition`, or 1.0 dioptre), uses bounded windowed scans and retains a safe lens baseline when no usable face/window control is available. Face selection uses detected geometry, never a person's identity. During a scan/restoration transition, detection and preview continue while identity changes and dataset extension wait for settled frames; disappearance still permits normal logout. This is opt-in and requires local focus/recognition validation.

Face AF verifies the manual seed from a fresh frame before scanning, bounds each scan to six seconds and applies a global 30-second failure cooldown. Restoration must be confirmed by lens metadata; an unverified restoration after two seconds faults while faces are present. Physical face-window AF validation is pending.

## 2.2.1 fresh presence metadata

[Recognition worker `run`](tools/recognition.py) now emits a separate `presence` snapshot after each inference; [the helper](node_helper.js) validates it and [the frontend](MMM-Face-Reco-DNN.js) publishes `FACE_RECOGNITION_PRESENCE`. Its exact six fields are `version`, `epoch`, `sequence`, `observed_at`, `available` and `labels`. Labels represent current matches, preserving duplicates so two faces with the same label remain a group. They are separate from cached login/logout sessions and their disappearance delay.

The observation timestamp is taken when a capture becomes available, before recognition, so slow processing cannot make an old frame appear newly observed. Consumers must enforce timestamp freshness and stable presence themselves; missing signals after pause or worker failure are not proof that anyone remains present. No pixels, face coordinates, scores, gallery contents, spoken commands or authentication grants are added to this signal. Recognition remains personalization, not authentication. Optional Santa greeting integration is explicitly enabled in its separate Christmas module; this Face module does not select profiles or speak names.

Validation for this release: 217 Python tests under Python 3.13.11, all 23 Node helper/frontend tests, JavaScript lint and targeted Python lint/format checks pass. Tests use synthetic frames, identities and fake clocks; no camera or private gallery was read for this change. See [the contract](docs/index.md#fresh-presence-contract) and [audit](AUDIT.md).

## 2.3.0 Christmas props and requested keepsakes

The existing camera worker can recognize four printed Christmas Protocol stamps:
`THALOVANT:PROP:LETTER`, `THALOVANT:PROP:STAR`, `THALOVANT:PROP:BELL`, and
`THALOVANT:PROP:GIFT`. It scans only the prop the active authenticated experience
expects. Voice alternatives remain available. The source-owned
[`experience_gateway.py`](tools/experience_gateway.py) admits revision-bound
metadata to the existing owner-authorized Thalovant Voice control client; no
new camera handle or arbitrary-command receiver is introduced.

At the keepsake scene, say “take Christmas photo”, then “save Christmas photo”,
“discard Christmas photo”, or “retake Christmas photo”. French uses “prends la
photo de Noël”, “garde la photo de Noël”, “supprime la photo de Noël”, or
“reprends la photo de Noël”. A minimum three-second countdown starts when Voice
is quiet. The unannotated JPEG preview stays in memory until an explicit save.
Save creates a private file in `christmasPhotoDirectory`, or the worker user's
`~/.local/share/thalovant/christmas-photos` by default. A container needs a
persistent writeable mount there. The saved image is separate from enrollment.

[Setup and fixed gateway](docs/index.md#christmas-props-and-keepsakes-230)
describes mount paths, UID ownership and the current-action/capture-revision
contract. Local tests use synthetic pixels; no physical camera or family photo
was used during implementation.

## 2.4.0 North Pole postcard preview

At the keepsake scene, choose “take Christmas postcard” or “prends la carte
postale de Noël” for an ivory, evergreen and gold North Pole frame with the
local capture date. A plain photo remains a separate choice. Both use the
existing countdown, preview, retake, save and discard flow; only an explicit
save writes the exact JPEG you previewed. The date is fixed at capture, even
if you save after midnight.

A restarted camera worker consumes an already-pending capture without taking
another photo. Choose a fresh retake after the camera is ready; ordinary
capture after an observed ready scene works as before.

[`render_postcard`](tools/utils/postcard.py) adds original vector artwork and
English or French lettering outside the complete photo, using existing OpenCV
and NumPy. It adds no camera, font package or face enrollment. The optional
`christmasPhotoTimezone` uses an installed IANA timezone and defaults to
`America/Toronto`; install system timezone data if the camera image lacks it.
The native producer and display bridge must support optional
`experience.photo_style`. Older states without that field retain plain photos.
See [the postcard contract](docs/index.md#optional-north-pole-postcard).
