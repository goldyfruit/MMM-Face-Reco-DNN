Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

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
    cameraFrameRate: 15,
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
