Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

# Proposed follow-up work

| Problem/opportunity | Proposal | Estimated impact |
|---|---|---|
| Accuracy not validated across people/lighting | Local held-out genuine/impostor evaluation; calibrate tolerance and margin without uploading images | Establish actual false acceptance/rejection rates |
| Old dependency resource API | Upstream face-recognition-models migration to importlib.resources | Remove deprecated runtime dependency |
| Embedding still CPU-heavy | Benchmark a separately versioned embedding model only with deliberate gallery migration | Potential speed gain; incompatible descriptors require retraining |
| Preview and detection share cadence | Separate capped preview cadence after measuring CPU/WebSocket cost | Smoother preview, with additional work/traffic |
| Existing automatic dataset candidates may be wrong | Add opt-in local review/quality tooling, preserving original images | Better future enrollment data |

Sources: [RecognitionEngine](tools/recognition.py), [FaceMatcher](tools/utils/vision.py), [CameraStream](tools/utils/camera.py), [encode_dataset](tools/utils/enrollment.py).

## Camera and recovery follow-up

| Opportunity | Proposed next step | Estimated impact |
|---|---|---|
| Reflections/default AF metering selected a near focus in the mirror trial | Compare opt-in face-window AF at several standing distances and lighting conditions, preserving identity thresholds | Establish actual focus acquisition and recognition reliability |
| Long exposures can blur a moving subject | Inspect local scalar exposure/focus metadata and compare optional short exposure in the real room | Measure the motion-blur versus gain/noise tradeoff |
| A driver may ignore graceful termination | Keep sessions cleared and diagnose locally before replacing the camera owner | Avoid competing processes and stale private views |
| Legacy debug mode logs identity classes | Keep normal diagnostics based on counts/scalars; redesign debug output before wider collection | Reduce accidental identity disclosure |

Use [FaceFocusController](tools/utils/focus.py#L121) and [CameraStream.face_focus_status](tools/utils/camera.py#L373) for bounded image-free focus diagnostics. A failed scan has a global 30-second cooldown; an unverified restoration faults while faces are present. Test those failure behaviors locally as well as successful acquisition. Source tests do not prove physical autofocus accuracy.
