Last Edit: Codex (GPT-6) - 2026-11-03 - Motive: Document armed local lost-letter stamp detection.

# Maintenance log —2026-11-02

Forked upstream d7f01321d919140c049ee55094f46837de0bb668. Added optional YuNet and an ISP lores camera path, lazy same-frame main images, cached validated gallery matching and no-op preprocessing. Fixed cross-face score labels, annotation contamination, ambiguous multi-face enrollment, failed enrollment replacement and worker lifecycle cleanup.

## Earlier fork validation

Validation before this record:110 Python tests passed,86% overall coverage including unchanged camera utility code; recognition worker97% coverage. Three Node tests pass and JavaScript lint passes. Actual Pi public-fixture benchmark and camera canary pass; see [performance](docs/performance.md). Additional asset-integrity tests and CI accompany the fork; final totals are updated after final checks.

## Transparency Report

AI model: Codex, GPT-6 family; exact runtime identifiers unavailable. Actions: source review, helper/runtime/test implementation, official-model provenance verification, public-fixture benchmarking and local hardware canary. Three review/implementation agents worked on separate camera, vision and enrollment areas. Human oversight: user requested a fork and improvement; no live user images/embeddings were uploaded or added to Git. Runtime rollout is verified separately before claiming deployment success.

Final pre-deployment checks:112 Python tests and3 Node tests pass; JavaScript lint passes. Official-model and public-fixture checksum checks pass. CI covers Python3.10/3.13 and Node24.

Final colour review found that libcamera sYCC is full-range while OpenCV I420 conversion assumes limited range. The portable decoder now preserves full Rec601 range; Pi5 can select direct RGB888 lores. New tests preserve gray32 as32 (the old conversion produced19). Real RGB camera canary passed with mean main/lores difference0.643 byte levels, untouched1920x1080 main image and restart. Final tests:119 Python,3 Node; lint passes. User confirmed recognition at normal standing distance before this colour-consistency correction.

Python format verification was added to CI using Ruff0.12.4; formatting-only changes preserve behaviour.119 Python tests,3 Node tests and JavaScript lint pass afterward.

Follow-up YuNet-specific scaling benchmark:4threads used65–68ms CPU at22.8–25.0ms wall time;2threads used41–43ms CPU at22.7–25.4ms wall time on the same public scene. Added opt-in opencvThreads (0 preserves defaults); Pi deployment selects2 for voice headroom. This is lower detector CPU cost at similar latency, not a universal OpenCV setting recommendation.

## 2.2.0 advertised camera controls and bounded recovery

Added optional advertised-control-aware autofocus and normal/short exposure. Existing focus is preserved by default. Manual/face modes accept an optional finite nonnegative lens position within the advertised range. The continuous-AF mirror trial selected a nearer lens position and the user reported recognition regression; a working manual baseline was retained while face-window AF was developed. A later local in-room trial verified face-window AF scanning/locking and usable PDAF; the user confirmed correct recognition at the normal position and after small distance changes. No private media or gallery data was exported. This is a deployment check, not a population-accuracy benchmark.

The opt-in [FaceFocusController](tools/utils/focus.py#L121) maps detected faces through the same request's crop/rotation and requires window metering/trigger support. It verifies the manual seed on a fresh frame before scanning, bounds scans to six seconds and applies a global 30-second failure cooldown. Restoration must be confirmed by lens metadata; after two seconds without confirmation, a worker with visible faces faults instead of publishing potentially blurred identities. Detection/preview continue during bounded transitions, while identity events and enrollment wait. Disappearance permits normal logout.

Worker failures immediately reset sessions/preview and use bounded close-before-restart recovery. Explicit pause/stop suppresses retries and rejects queued old events. Activation generations renew readiness and restore the same known face after quick pause/resume; stale in-flight results cannot train or reopen a session. The actual PythonShell formatter/send path is tested. Frontend delayed logout now handles multiple users and duplicate notifications correctly. Normal failure/session logs avoid private labels.

Validation: 210 Python tests pass under actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests. All 22 Node tests, JavaScript lint, targeted Ruff and Python formatting checks pass. Tests use synthetic arrays/identities, public fixtures, fake clocks and mocked controls; physical face-window focus validation is recorded below. [Audit/source references](AUDIT.md), [camera tests](tests/test_camera.py) and [runtime tests](tests/test_runtime.py) record the scope.

### Transparency Report

AI model: Codex, GPT-6 family; exact runtime identifier unavailable. Actions: source/control inspection, face-window geometry and bounded recovery implementation, stale-session/timer fixes, unit/integration testing and documentation. Camera and runtime work used separate implementation agents with an independent lifecycle/geometry review. Human oversight: the user requested continued improvement and camera/config review; the user participated in the authorized physical comparison and confirmed recognition with automatic face-targeted focus. No private face images/embeddings were uploaded or added to Git by this source work.

Native validation on 2026-11-02 passed with the user present: all 30 frames contained a face; all 25 settled frames were accepted. The face-window run recorded three scanning and 25 focused frames, 24 locked-ROI samples, and positive PDAF confidence in 267 samples without format errors. Median matching distance was 0.4255 versus 0.539 in the preceding manual-focus run. This is a sequential same-room comparison, not a population accuracy result. Production now uses face-window AF seeded at 1.0 dioptre; the gallery and identity thresholds are unchanged. The correction targets AF metering/ROI behavior; it does not establish a lens-motor defect. Aggregate evidence: workspace `outputs/mm01-face-af-validated-20261102.json`; no private images or identity names are included here.

## 2.2.1 fresh-presence review — 2026-11-03

Added a separate exact-six-field current-frame metadata path in [`run`](tools/recognition.py), the [Node helper](node_helper.js) and [frontend notification adapter](MMM-Face-Reco-DNN.js). It exports raw match-label multiplicity rather than deduplicated login names, preventing a same-label group from appearing to be one person. Pending autofocus and over-cap groups produce unavailable empty snapshots. Legacy login/logout timing and gallery matching remain unchanged; no additional captures are started.

Independent review found that timestamping after inference could make a slow result appear fresh. The timestamp is now taken immediately after entering the captured request and before inference. A synthetic ten-second inference regression test verifies that the original observation time survives. Helper admission requires all six own fields, bounded strings, safe positive sequence and finite timestamp; Unicode control/format/surrogate labels and extra fields are rejected. The downstream optional Unix-socket consumer was reviewed separately; this source adds no socket listener, authentication mapping or spoken text.

Actual verification: 217 Python tests pass under Python 3.13.11 with 94% overall coverage and 97% recognition-worker coverage; all 23 Node tests and JavaScript lint pass. Targeted Ruff lint and formatting pass for `tools/recognition.py` and `tests/test_runtime.py`. [Runtime tests](tests/test_runtime.py) cover empty/multiple/duplicate/over-cap faces, autofocus unavailability and processing-delay timestamps; [Node tests](tests/test_node_helper.cjs) cover bounded admission, multiplicity and independent frontend forwarding. Package and lockfile now agree on 2.2.1. No new physical camera/gallery test or recognition accuracy/FPS gain is claimed. Earlier hardware evidence above is preserved as historical validation.

### Transparency Report — fresh presence

AI model: Codex, GPT-6. Actions: independent review, timestamp correction, helper validation, synthetic regression tests, release metadata and documentation. Human oversight: user requested Christmas mirror activities and personalized Santa greetings; root agent coordinates deployment separately. No live camera/gallery access, private media export, commit, push or rollout was performed by this review.

## 2026-11-03 — Armed letter stamp scanner

Added a bounded two-language OpenCV scan within the existing camera capture
loop. Checks cover fixed markers, stale states, transient gateway replies and
the removal latch between takes. The full local suite passes 228 Python and
23 Node tests. Synthetic print scans establish decoder behavior. Room lighting
and the physical prop still require a rehearsal.

### Transparency Report

Codex (GPT-6) wrote `LetterScanner`, integrated the existing capture and ran
the synthetic tests. The owner requested local camera recognition for the prop.
The scanner receives only frame memory from the existing camera worker.
[Worker](tools/recognition.py) · [Tests](tests/test_letter_marker.py)
