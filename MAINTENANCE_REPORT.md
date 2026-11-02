Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Maintenance log —2026-11-02

Forked upstream d7f01321d919140c049ee55094f46837de0bb668. Added optional YuNet and an ISP lores camera path, lazy same-frame main images, cached validated gallery matching and no-op preprocessing. Fixed cross-face score labels, annotation contamination, ambiguous multi-face enrollment, failed enrollment replacement and worker lifecycle cleanup.

Validation before this record:110 Python tests passed,86% overall coverage including unchanged camera utility code; recognition worker97% coverage. Three Node tests pass and JavaScript lint passes. Actual Pi public-fixture benchmark and camera canary pass; see [performance](docs/performance.md). Additional asset-integrity tests and CI accompany the fork; final totals are updated after final checks.

## Transparency Report

AI model: Codex, GPT-6 family; exact runtime identifiers unavailable. Actions: source review, helper/runtime/test implementation, official-model provenance verification, public-fixture benchmarking and local hardware canary. Three review/implementation agents worked on separate camera, vision and enrollment areas. Human oversight: user requested a fork and improvement; no live user images/embeddings were uploaded or added to Git. Runtime rollout is verified separately before claiming deployment success.

Final pre-deployment checks:112 Python tests and3 Node tests pass; JavaScript lint passes. Official-model and public-fixture checksum checks pass. CI covers Python3.10/3.13 and Node24.

Final colour review found that libcamera sYCC is full-range while OpenCV I420 conversion assumes limited range. The portable decoder now preserves full Rec601 range; Pi5 can select direct RGB888 lores. New tests preserve gray32 as32 (the old conversion produced19). Real RGB camera canary passed with mean main/lores difference0.643 byte levels, untouched1920x1080 main image and restart. Final tests:119 Python,3 Node; lint passes. User confirmed recognition at normal standing distance before this colour-consistency correction.

Python format verification was added to CI using Ruff0.12.4; formatting-only changes preserve behaviour.119 Python tests,3 Node tests and JavaScript lint pass afterward.
