Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

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

`CameraStream.capture()` in [camera.py](../tools/utils/camera.py) owns and releases each camera request. `CameraFrame.full_frame()` materializes a synchronized full-resolution image only when needed. [RecognitionEngine.recognize](../tools/recognition.py) retains dlib128 embeddings. [FaceMatcher.match](../tools/utils/vision.py) validates and caches the gallery. No automatic gallery regeneration occurs.

`tools/encode.py` still writes `encodings`/`names` pickle keys. [encode_dataset](../tools/utils/enrollment.py) skips unreadable/no-face/multi-face images, preserves top-level identity labels in nested folders, and replaces a valid gallery atomically only after usable enrollment. Review automatically collected photos before retraining; this is personalization, not liveness/authentication.

Back up the current module, user config, face gallery and dataset before switching branches. Preserve local lockfile changes and untracked tools. Keep original MIT notices and model attribution. For results and boundaries see [performance](performance.md), [audit](../AUDIT.md), [FAQ](../FAQ.md), [maintenance](../MAINTENANCE_REPORT.md).
