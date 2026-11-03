Last Edit: Codex (GPT-6) - 2026-11-03 - Motive: Document tested fresh per-frame presence metadata and release 2.2.1 without changing authentication or cached login behavior.

# Quick facts

| Field | Value |
|---|---|
| Package | mmm-face-reco-dnn |
| Version | 2.2.1; source of truth package.json |
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
| Current validation | 217 Python tests on Python 3.13.11, 23 Node tests, JavaScript lint and targeted Python lint/format; 94% overall coverage |
| Fresh presence | `FACE_RECOGNITION_PRESENCE`: exact6fields version/epoch/sequence/observed_at/available/labels; capture-enter timestamp, per-inference duplicate labels preserved; no authorization mapping |
| Fresh presence sources | `tools/recognition.py:run`; `node_helper.js:validPresence`; `MMM-Face-Reco-DNN.js:socketNotificationReceived` |
| Presence limits | ≤8 labels, ≤100 JavaScript code units each, Unicode controls rejected; unsettled/over-cap unavailable; consumers must expire missing snapshots |
