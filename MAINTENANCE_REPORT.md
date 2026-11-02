Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

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

Validation: 210 Python tests pass under actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests. All 22 Node tests, JavaScript lint, targeted Ruff and Python formatting checks pass. Tests use synthetic arrays/identities, public fixtures, fake clocks and mocked controls; physical face-window focus validation remains separate. [Audit/source references](AUDIT.md), [camera tests](tests/test_camera.py) and [runtime tests](tests/test_runtime.py) record the scope.

### Transparency Report

AI model: Codex, GPT-6 family; exact runtime identifier unavailable. Actions: source/control inspection, face-window geometry and bounded recovery implementation, stale-session/timer fixes, unit/integration testing and documentation. Camera and runtime work used separate implementation agents with an independent lifecycle/geometry review. Human oversight: the user requested continued improvement and camera/config review; the user participated in the authorized physical comparison and confirmed recognition with automatic face-targeted focus. No private face images/embeddings were uploaded or added to Git by this source work.
