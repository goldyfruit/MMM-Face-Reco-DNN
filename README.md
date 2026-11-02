Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

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
