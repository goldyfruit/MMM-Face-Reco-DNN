Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

# Quick facts

| Field | Value |
|---|---|
| Package | mmm-face-reco-dnn |
| Version | 2.2.0; source of truth package.json |
| JavaScript entry | MMM-Face-Reco-DNN.js / node_helper.js |
| Python entry | tools/recognition.py:main; tools/encode.py:main |
| Gallery | Existing names/encodings pickle with128-value dlib descriptors |
| Optional detector | YuNet2023mar, OpenCV FaceDetectorYN |
| Camera | Picamera2 main plus optional ISP lores, lazy full-resolution frames |
| Tests | pytest tests; npm test; npm run lint |

[CameraStream](tools/utils/camera.py), [FaceDetector/FaceMatcher](tools/utils/vision.py), [RecognitionEngine](tools/recognition.py), [enrollment](tools/utils/enrollment.py).

| Follow-up control | Behaviour |
|---|---|
| cameraAutofocus | default leaves focus unchanged; optional continuous/manual/face controls are capability checked |
| cameraLensPosition | null by default; finite nonnegative dioptres with manual AF or as a face-AF seed, within the advertised lens range |
| cameraExposureMode | normal by default; optional short is opt-in |
| workerRestartDelay | 1000ms base by default; bounded exponential backoff,30s maximum |
| workerRestartLimit | 3 early retries by default;0 disables, maximum10 |
| Worker readiness | `tools/recognition.py:run` sends ready after a successful request release at startup and after resume |
| Face-window AF | Opt-in same-request face ROI; bounded scans; identity/enrollment transitions wait for settled input |
| Camera diagnostics | `CameraFrame.metadata` and `CameraStream.latest_metadata`: detached finite scalar allowlist |
| Current validation | 210 Python tests, 22 Node tests, JavaScript lint and targeted Python formatting |
