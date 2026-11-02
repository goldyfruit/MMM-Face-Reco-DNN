Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

# Installation and configuration

Install this fork's reviewed branch and run `npm ci`. Use a working Python camera environment containing Picamera2, system libcamera, OpenCV with `FaceDetectorYN`, NumPy, dlib and face_recognition. The tested Pi environment uses Python3.13, OpenCV4.10.0, NumPy2.2.4, dlib20.0.1 and face_recognition1.3.0. Keep the sensor-specific libcamera/Python binding ABI aligned; this fork does not install firmware or change voltage settings.

The existing `face-recognition-models` package depends on legacy `pkg_resources`; the tested environment retains its explicit setuptools80.9.0 compatibility pin. This remains dependency debt, not a NumPy2 incompatibility in the tested fork.

| Setting | Default | Purpose |
|---|---|---|
| method | dnn | Legacy dnn/HOG, haar or optional yunet |
| yunetScoreThreshold | 0.8 | Detection score cutoff; not a calibrated probability of identity |
| yunetNmsThreshold | 0.3 | YuNet box suppression |
| cameraLores | false | Process an ISP low-resolution stream |
| cameraLoresFormat | YUV420 | Full-range portable YUV; RGB888 recommended on Pi5 |
| cameraFrameRate | null | Optional sensor cap;15 FPS tested on Pi5 |
| detectionUpsample | 1 | Preserve legacy HOG/CNN behaviour;0 may miss small faces |
| identityMargin | 0 | Optional competing-identity distance gap; calibrate before enabling |
| previewQuality | 95 | JPEG quality for preview, independent of enrollment images |
| opencvThreads | 0 | Leave defaults at0; explicit1–64 controls OpenCV worker threads |

`CameraStream.capture()` in [camera.py](../tools/utils/camera.py) owns and releases each camera request. `CameraFrame.full_frame()` materializes a synchronized full-resolution image only when needed. [RecognitionEngine.recognize](../tools/recognition.py) retains dlib128 embeddings. [FaceMatcher.match](../tools/utils/vision.py) validates and caches the gallery. No automatic gallery regeneration occurs.

`tools/encode.py` still writes `encodings`/`names` pickle keys. [encode_dataset](../tools/utils/enrollment.py) skips unreadable/no-face/multi-face images, preserves top-level identity labels in nested folders, and replaces a valid gallery atomically only after usable enrollment. Review automatically collected photos before retraining; this is personalization, not liveness/authentication.

Back up the current module, user config, face gallery and dataset before switching branches. Preserve local lockfile changes and untracked tools. Keep original MIT notices and model attribution. For results and boundaries see [performance](performance.md), [audit](../AUDIT.md), [FAQ](../FAQ.md), [maintenance](../MAINTENANCE_REPORT.md).

## Advertised camera controls

| Setting | Default | Supported choices / behaviour |
|---|---|---|
| cameraAutofocus | default | Preserve existing focus; optional continuous, manual or face-window AF |
| cameraLensPosition | null | Optional finite nonnegative dioptres with manual/face AF and an advertised in-range control; 1.0 is about one metre, 0 is infinity |
| cameraExposureMode | normal | normal or short; only an advertised AeExposureMode value is applied |
| workerRestartDelay | 1000 | Integer 100–30000 ms base; exponential backoff capped at 30000 ms |
| workerRestartLimit | 3 | Integer 0–10; 0 disables automatic retry |

[CameraStream](../tools/utils/camera.py#L199) checks advertised controls and preserves unsupported/fixed-focus behavior. Continuous AF requests Normal range when available but remains opt-in: it selected a nearer focus and recognition regressed in the mirror trial. Foreground/reflections are a plausible explanation, not established by lens metadata alone. Normal exposure remains default; optional short exposure needs a real-room blur/noise comparison. Scalar camera metadata excludes pixels and identity names.

### Opt-in face-window autofocus

Set `cameraAutofocus:'face'` to target detected geometry instead of default metering. [FaceFocusController](../tools/utils/focus.py#L121) maps rotated/resized face boxes through the same capture's sensor crop, selects the largest valid face deterministically and requires advertised window metering/trigger support. Recognition is not required. The mode begins from an advertised manual seed (`cameraLensPosition`, or 1.0 dioptre) and verifies its LensPosition on a fresh frame before requesting a scan; unsupported ROI AF retains a safe baseline rather than silently switching to global continuous AF.

`camera.face_focus_status` provides bounded image-free state/geometry diagnostics. Pending scans/restoration defer identity changes and dataset extension while detection/preview continue. No-face cancellation permits normal logout. Scans time out after six seconds; failure imposes a global 30-second cooldown even when the ROI changes. A restoration unverified by metadata after two seconds faults the worker while faces are present instead of publishing unsettled identities. Physical focus/recognition validation is pending; do not infer success from mocked controls.

### Worker lifecycle

[run](../tools/recognition.py#L152) renews readiness after a processed/released capture at startup and resume, including no-face/no-preview operation. Activation generations also reconcile identity snapshots so the same known face reopens its session after quick pause/resume, while stale in-flight identity/enrollment results are rejected. The actual PythonShell text stdin formatter preserves plain commands with JSON stdout.

[Helper recovery](../node_helper.js#L27) clears stale sessions/preview immediately through [frontend reset](../MMM-Face-Reco-DNN.js#L405), waits for old-process closure, and bounds retries. Explicit stop/pause suppresses retries and queued activity; a process that ran at least 60 seconds gets a fresh early-failure budget. Normal multi-user logout callbacks bind users independently and emit logout after expiry. This remains personalization, not authentication/liveness.

Current validation: 210 Python tests pass under actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests; 22 Node tests and lint/format checks pass. See [camera tests](../tests/test_camera.py), [runtime tests](../tests/test_runtime.py) and [audit](../AUDIT.md). Primary AF semantics are documented in [libcamera control definitions](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml); [Picamera2 configuration](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py) shares the main crop with preview lores when aspect preservation is disabled.
