Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Maintenance log —2026-11-02

Forked upstream d7f01321d919140c049ee55094f46837de0bb668. Added optional YuNet and an ISP lores camera path, lazy same-frame main images, cached validated gallery matching and no-op preprocessing. Fixed cross-face score labels, annotation contamination, ambiguous multi-face enrollment, failed enrollment replacement and worker lifecycle cleanup.

Validation before this record:110 Python tests passed,86% overall coverage including unchanged camera utility code; recognition worker97% coverage. Three Node tests pass and JavaScript lint passes. Actual Pi public-fixture benchmark and camera canary pass; see [performance](docs/performance.md). Additional asset-integrity tests and CI accompany the fork; final totals are updated after final checks.

## Transparency Report

AI model: Codex, GPT-6 family; exact runtime identifiers unavailable. Actions: source review, helper/runtime/test implementation, official-model provenance verification, public-fixture benchmarking and local hardware canary. Three review/implementation agents worked on separate camera, vision and enrollment areas. Human oversight: user requested a fork and improvement; no live user images/embeddings were uploaded or added to Git. Runtime rollout is verified separately before claiming deployment success.

Final pre-deployment checks:112 Python tests and3 Node tests pass; JavaScript lint passes. Official-model and public-fixture checksum checks pass. CI covers Python3.10/3.13 and Node24.
