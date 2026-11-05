Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Require a valid ready scene before arming photo capture after startup or reconnection.

# Maintenance log —2026-11-02

## 2026-11-05 — Startup readiness correction (2.4.1 source)

After a synthetic reproduction of disconnected startup followed by an old
persisted capture, [PhotoBooth.process_frame](tools/utils/experience_camera.py)
now marks readiness only after a valid active `ExperienceView`. None/off/stale
or hidden views no longer arm a later capture. The first valid already-pending
capture remains unavailable until a new explicit retake. No public wire fields,
photo store, gateway behavior or camera calibration were changed.

Verification: all 314 Python tests (91% overall coverage), 25 Node tests,
JavaScript lint, Ruff 0.12.4 format checks and targeted Python lint pass.
Eight invalid-startup cases exercise reconnect rejection and a later successful
retake; two valid-ready cases retain normal capture. Package/lock version is
2.4.1. Work is on `fix/postcard-startup-ready` from origin/dev 02557d9; no commit,
push or deployment by this correction. The pre-existing performance doc edit
remains untouched.

### Transparency Report — valid startup readiness

- **AI model:** Codex (GPT-6).
- **Actions:** Reproduced the startup gap, moved the readiness gate behind valid
  state admission, added regression tests and updated version/documentation.
- **Oversight:** Parent agent identified the concrete reconnection path; user
  authorized the postcard feature. Automated synthetic verification completed;
  no live camera, saved image, native action or service intervention used.

## 2026-11-05 — Opt-in North Pole postcards (2.4.0 source)

Added [render_postcard](tools/utils/postcard.py), an OpenCV/NumPy-only ivory,
evergreen and gold frame with original vector postal artwork, English/French
caption and trusted local capture date. [PhotoBooth.process_frame](tools/utils/experience_camera.py)
renders only an explicit postcard capture, encodes once, and keeps the same
JPEG bytes for preview and explicit save. Plain-photo encoding remains on its
existing path. The optional source `photo_style` defaults to plain and is bound
to the capture revision; changing style cannot reframe or save older pixels.
The existing shared camera callback, Voice gate, private storage and enrollment
boundaries are preserved. `christmasPhotoTimezone` defaults to America/Toronto
and validates installed IANA names through stdlib ZoneInfo.

Verification: 307 Python tests pass on Python 3.13.11, OpenCV 4.10.0 and NumPy
2.2.4; 91% overall coverage and 100% compositor coverage. All 25 Node tests,
ESLint, CI-pinned Ruff 0.12.4 format checks, targeted Python lint and whitespace
checks pass. [Compositor tests](tests/test_postcard.py) cover full-image retention,
French accent, dimensions and local midnight; [PhotoBooth regressions](tests/test_experience_camera.py)
cover byte-identical save, style/countdown races, legacy defaults, retake back to
the unchanged plain encoding, and unavailable timezone behavior. English/French
synthetic previews received visual review. Package and lockfile agree on 2.4.0.
No physical camera/photo, face gallery, live service, commit, push or deployment
was used for this source change. The user's `docs/performance.md` edit is intact.

Final review reproduced a startup replay: a fresh worker observing an old
persisted capture with fresh heartbeats performed a new countdown and preview.
Added a startup observation guard to consume that action as unavailable without
pixels. A later explicit revision captures normally, and a prior ready/off scene
arms normal capture. Four regressions cover restart, heartbeat/reset suppression
and normal readiness; the final complete suite is 307 Python / 25 Node tests.

### Transparency Report — postcard polish

- **AI model:** Codex (GPT-6).
- **Actions taken:** Implemented the opt-in compositor, strict locale/style gate,
  capture-style guards, private timezone config, tests, version and documentation.
- **Oversight:** User approved postcard polish; parent agent reviewed the EN/FR
  synthetic design. Source and automated tests were checked locally. Physical
  photo acceptance and release/deployment are separate coordinated work.

## 2026-11-02 — Base fork

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

## 2026-11-05 - Christmas props and explicit keepsakes (2.3.0)

Added fixed expected-prop scanning, a source-owned fixed Voice gateway,
revision/capture-bound keepsake countdown and in-memory previews, explicit
private saves/discards, worker integration and bounded helper/frontend status.
All new camera uses share the existing CameraStream request. Camera calibration
and the user's existing `docs/performance.md` changes were preserved.

Verification: 279 Python tests and 25 Node tests passed; JavaScript lint,
new/modified Python helper Ruff checks and git diff whitespace checks pass.
All frames, state files and owner requests used synthetic fixtures or stubs.
No physical camera, family images, private gallery, Voice credential or current
runtime was accessed or activated by this implementation.

### Transparency report

- **AI model:** Codex (GPT-6).
- **Actions taken:** Implemented the fixed camera/gateway/state helpers,
  integrated existing worker/helper/frontend paths, added regression tests,
  documented source-owned setup and updated package version metadata.
- **Oversight:** User authorized the proposed Christmas experience. Automated
  checks and source review performed; physical room acceptance and device
  activation remain separate. No new biometric enrollment was performed.

## 2026-11-05 - Live activation 2.3.0

The live camera module and fixed prop gateway are installed. One existing camera worker runs with preserved focus, lens, exposure, frame rate, dataset and gallery settings. The private persistent photo mount is owned by UID 1000 with mode 0700; gateway socket is mode 0600. Startup error counts are zero. No photo capture was requested during rollout.

### Transparency report

Codex (GPT-6) used the user-provided mirror SSH address, existing operator
upgrade paths, private rollback backups and fixed silent owner-Voice requests.
No extra recording/listen turn or photo capture was requested by the checks;
the normal wake-word service remained running. User authorization covers
implementation and installation of the reviewed Christmas experience.
