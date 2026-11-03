Last Edit: Codex (GPT-6) - 2026-11-03 - Motive: Document armed local lost-letter stamp detection.

# Evidence-based audit

## Fixed with tests

- Last-face distance reused in every preview label: [annotate](tools/recognition.py), [test_annotations_are_separate_and_use_per_face_distance](tests/test_recognition.py).
- Multi-face photos enrolled under one identity: [encode_dataset](tools/utils/enrollment.py), [enrollment tests](tests/test_encode.py).
- Preview annotation could contaminate saved training pixels: [CameraFrame.full_frame](tools/utils/camera.py) and [dataset tests](tests/test_recognition.py).
- Camera/request leaks on failed processing or trigger-thread startup: [run](tools/recognition.py), [runtime tests](tests/test_runtime.py).
- Invalid/mismatched galleries and out-of-bounds boxes: [FaceMatcher/FaceDetector](tools/utils/vision.py), [vision tests](tests/test_vision.py).

## Remaining limits

Private pickle galleries are trusted local input; do not load arbitrary remote pickle files. Existing automatically collected photos can contain prior mistakes and are not re-labelled by this fork. AEC/audio tuning is outside this module. This is not liveness detection or secure authentication. One public portrait with transformations cannot establish broad recognition accuracy. dlib embedding cost remains CPU-bound. The dependency `face-recognition-models` still needs the documented legacy pkg_resources compatibility dependency. See [performance](docs/performance.md) for measured scope and [suggestions](SUGGESTIONS.md) for next work.

## Earlier fork validation

Actual tests were run before writing this audit:110Python and3Node tests passed; lint passed. Final totals may grow as asset/CI checks are added.

Final pre-deployment checks:112 Python tests and3 Node tests pass; JavaScript lint passes. Official-model and public-fixture checksum checks pass. CI covers Python3.10/3.13 and Node24.

Final colour review found that libcamera sYCC is full-range while OpenCV I420 conversion assumes limited range. The portable decoder now preserves full Rec601 range; Pi5 can select direct RGB888 lores. New tests preserve gray32 as32 (the old conversion produced19). Real RGB camera canary passed with mean main/lores difference0.643 byte levels, untouched1920x1080 main image and restart. Final tests:119 Python,3 Node; lint passes. User confirmed recognition at normal standing distance before this colour-consistency correction.

Python format verification was added to CI using Ruff0.12.4; formatting-only changes preserve behaviour.119 Python tests,3 Node tests and JavaScript lint pass afterward.

Follow-up YuNet-specific scaling benchmark:4threads used65–68ms CPU at22.8–25.0ms wall time;2threads used41–43ms CPU at22.7–25.4ms wall time on the same public scene. Added opt-in opencvThreads (0 preserves defaults); Pi deployment selects2 for voice headroom. This is lower detector CPU cost at similar latency, not a universal OpenCV setting recommendation.

## 2.2.0 camera and session reliability review

- [Helper availability/recovery](node_helper.js#L27) immediately clears stale browser sessions/preview on worker failure, waits for actual closure before replacement, and limits early retries. Paused/draining workers cannot forward queued activity. Normal helper logs use counts and generic errors instead of identity labels or private gallery paths; explicitly enabled legacy debug logs can still contain user classes.
- [Frontend session reset](MMM-Face-Reco-DNN.js#L405) bypasses the ordinary disappearance delay after failure/pause. Multi-user logout timers now bind each user independently, avoid duplicate orphan timers, and emit logout after actual expiry.
- [Command monitoring and worker lifecycle](tools/recognition.py#L129) renew readiness and reconcile identities after every activation generation, including rapid stop/start with the same known face. In-flight pre-pause identity/enrollment results are discarded. Tests invoke the installed PythonShell text formatter/send path, preserving plain stdin commands and JSON stdout.
- [Camera settings](tools/utils/camera.py#L199) preserve existing focus by default. Continuous/manual/face modes and normal/short exposure are capability checked; optional lens position requires a finite nonnegative value and an advertised range. The continuous-AF mirror trial selected a nearer focus and recognition regressed; foreground/reflections are a plausible cause, not proven by lens metadata alone.
- [Face-window geometry](tools/utils/focus.py#L45), [FaceFocusController](tools/utils/focus.py#L121) and the [same-request adapter](tools/utils/camera.py#L379) target detected geometry independently of identity. Pending scans/restoration suppress identity/enrollment commits while preview continues. No-face cancellation permits normal logout. Scans are bounded to six seconds; a failed scan imposes a global 30-second cooldown even if the ROI moves. Resume verifies the manual seed from a fresh frame before scanning. Missing/wrong lens metadata after two seconds of restoration faults the worker while faces are present rather than treating input as settled.

Current validation: 210 Python tests pass under actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests; 22 Node tests, JavaScript lint, targeted Ruff and Python formatting pass. See [camera tests](tests/test_camera.py), [runtime tests](tests/test_runtime.py) and [Node/frontend tests](tests/test_node_helper.cjs). The source tests use synthetic data and clocks; physical face-window AF validation is pending and no new recognition accuracy/FPS gain is claimed. Pre-existing unrelated Ruff findings in the legacy picture utility/test import remain outside this change.

Primary AF references: [libcamera public control definitions](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml) define sensor-pixel AF windows, Windows metering and explicit Auto-mode triggers. [Picamera2 configuration](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py) shares the main crop with preview lores when aspect preservation is disabled. [Pi 5 per-stream crop discussion](https://lists.libcamera.org/pipermail/libcamera-devel/2024-October/045497.html) distinguishes stream-zero ScalerCrop from per-stream ScalerCrops. The adapter uses actual current-request geometry.

## 2.2.1 current-frame presence review — 2026-11-03

- Fixed post-inference timestamp freshness: [`run`](tools/recognition.py) records observation time before recognition; `test_presence_timestamp_precedes_slow_inference` in [runtime tests](tests/test_runtime.py) simulates ten seconds of processing without allowing a fresh timestamp on old pixels.
- Current-frame `matches` retain repeated labels, unlike `FrameResult.names` used for legacy sessions. The parameterized `test_presence_preserves_duplicate_face_count_and_bounds_payload` covers empty, two same-label, eight and nine-face results. Focus transitions suppress available identities through `test_presence_unavailable_during_autofocus_transition`.
- [Helper `validPresence`](node_helper.js) requires six own fields, finite nonnegative observation time, worker epoch, positive safe sequence and at most eight bounded labels. Extra fields and Unicode control/format/surrogate labels are rejected. [Node/frontend tests](tests/test_node_helper.cjs) verify this separate metadata path without changing cached session semantics.

Verification completed before this record: 217 Python tests on Python 3.13.11 (94% overall / 97% recognition-worker coverage), 23 Node tests, JavaScript lint and targeted Ruff lint/format pass. This metadata is observational and unauthenticated: consumers must enforce short freshness expiry, independent stability, exact private mappings and their own authorization. Clock jumps fail only in a consumer that checks age. A paused/dead worker can stop sending; the Face frontend does not synthesize camera evidence. No inference accuracy, face liveness or private hardware result is established by these synthetic tests.

## Armed letter stamps

`LetterScanner.scan()` checks a fresh armed rehearsal and idle Voice before
requesting full-resolution pixels. Only the two fixed stamp values can produce
an event. The socket test checks that the event contains only the revision,
language and capture time. The latch test covers repeated frames and removing
the stamp before a new take. [Marker tests](tests/test_letter_marker.py)
