Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Proposed follow-up work

| Problem/opportunity | Proposal | Estimated impact |
|---|---|---|
| Accuracy not validated across people/lighting | Local held-out genuine/impostor evaluation; calibrate tolerance and margin without uploading images | Establish actual false acceptance/rejection rates |
| Old dependency resource API | Upstream face-recognition-models migration to importlib.resources | Remove deprecated runtime dependency |
| Embedding still CPU-heavy | Benchmark a separately versioned embedding model only with deliberate gallery migration | Potential speed gain; incompatible descriptors require retraining |
| Preview and detection share cadence | Separate capped preview cadence after measuring CPU/WebSocket cost | Smoother preview, with additional work/traffic |
| Existing automatic dataset candidates may be wrong | Add opt-in local review/quality tooling, preserving original images | Better future enrollment data |

Sources: [RecognitionEngine](tools/recognition.py), [FaceMatcher](tools/utils/vision.py), [CameraStream](tools/utils/camera.py), [encode_dataset](tools/utils/enrollment.py).
