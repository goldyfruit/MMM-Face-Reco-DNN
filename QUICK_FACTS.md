Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Add opt-in dated North Pole postcards with unchanged explicit photo consent.

# Quick facts

| Field | Value |
|---|---|
| Package | mmm-face-reco-dnn |
| Version | 2.4.0 source; source of truth package.json; activation is separate |
| JavaScript entry | MMM-Face-Reco-DNN.js / node_helper.js |
| Python entry | tools/recognition.py:main; tools/encode.py:main |
| Gallery | Existing names/encodings pickle with128-value dlib descriptors |
| Optional detector | YuNet2023mar, OpenCV FaceDetectorYN |
| Camera | Picamera2 main plus optional ISP lores, lazy full-resolution frames |
| Tests | pytest tests; npm test; npm run lint |

| Postcard source (2.4.0) | Value |
|---|---|
| Opt-in | Native keepsake capture with optional experience.photo_style=postcard; absent/plain preserves ordinary photo |
| Artwork | tools/utils/postcard.py:render_postcard; OpenCV/NumPy vector mat, seal and English/French caption |
| Date | Trusted capture observed_at, stdlib ZoneInfo, christmasPhotoTimezone default America/Toronto; system IANA data required |
| Storage | Same single encoded preview JPEG saved only by existing explicit save; no automatic files or enrollment |
| Replay boundary | Episode identity + photo_capture_revision + photo_style; a new style needs a new capture |
| Worker startup | First observed already-pending capture is consumed as unavailable; fresh explicit retake required; ready/off observation arms normal future capture |
| Bounds | Complete composite <=1200 pixels per edge; JPEG<=500000 bytes; public event<700000 bytes |
| Activation | Source prepared and synthetic-tested; no 2.4.0 deployment or physical photo acceptance by this change |

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
| Current validation | 307 Python tests on Python 3.13.11, 25 Node tests, ESLint and CI-pinned Ruff 0.12.4; 91% overall coverage, postcard compositor 100% |
| Fresh presence | `FACE_RECOGNITION_PRESENCE`: exact6fields version/epoch/sequence/observed_at/available/labels; capture-enter timestamp, per-inference duplicate labels preserved; no authorization mapping |
| Fresh presence sources | `tools/recognition.py:run`; `node_helper.js:validPresence`; `MMM-Face-Reco-DNN.js:socketNotificationReceived` |
| Presence limits | ≤8 labels, ≤100 JavaScript code units each, Unicode controls rejected; unsettled/over-cap unavailable; consumers must expire missing snapshots |

## Local lost-letter stamps

`LetterScanner` borrows an existing `CameraFrame.full_frame()` only when the
Hub display is armed. It sends the revision, language and time to a local
Unix socket and keeps no frame files.
[Worker integration](tools/recognition.py) · [Stamp gate](tools/utils/letter_marker.py)

## Christmas camera extension (2.3.0)

| Field | Value |
|---|---|
| Camera ownership | Existing CameraStream and current captured.full_frame only; no second handle |
| Fixed props | THALOVANT:PROP:LETTER / STAR / BELL / GIFT; expected prop only, at most 2 scans per second |
| Gates | Fresh connected episode, protocol_enabled true, current expected prop, fresh listening Voice, no unrelated presentation/video |
| Gateway | tools/experience_gateway.py; owner ControlClient.from_file; fixed POST /v1/ask only; one pending request |
| Socket | /christmas-greetings/experience.sock; private Unix socket, exact4field observations; no image payload |
| Photo trigger | Explicit native capture/retake only at keepsake; at least3second quiet countdown |
| Photo identity | Current episode identity plus exact native photo_capture_revision; old preview cannot satisfy newer save |
| Preview | In-memory unannotated bounded JPEG; no photo file until explicit save |
| Save | christmasPhotoDirectory absolute/private owned path; default worker ~/.local/share/thalovant/christmas-photos; file0600 / folder0700 |
| Status | CHRISTMAS_PHOTO_STATUS; current revision/capture_revision, fixed status, optional countdown seconds or bounded base64 JPEG |
| Validation | 279 Python tests, 25 Node tests, JavaScript lint and targeted Python Ruff checks pass |
| Hardware acceptance | Not exercised; tests use synthetic frames and stub Voice client |

## Live installation - 2026-11-05

- Installed component version: `2.3.0`.
- Coordinated native Protocol/bridge: `0.10.1`; Companion `0.6.0`; Games `0.2.0`; mirror `0.17.0`; camera `2.3.0`.
- Music activation requires fresh enabled Protocol; pause/off stays quiet.
- Voice service and calibrated camera settings are preserved.
