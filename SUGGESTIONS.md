Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Record live Christmas experience 2.3.0 activation.

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

## Fresh-presence consumer follow-up

| Opportunity | Proposed next step | Estimated impact |
|---|---|---|
| Match labels can be cached by unrelated consumers | Consume `FACE_RECOGNITION_PRESENCE` with short timestamp expiry, epoch/sequence ordering and stable single-face checks; retain no raw labels in logs | Prevent greetings from delayed logout state and same-label groups |
| Clock or inference delays can invalidate current-frame evidence | Fail closed when capture-enter observation time is old/future; compare delayed-inference synthetic tests with bounded aggregate local timing if needed | Avoid stale or queued greetings without collecting images |
| A label is not an authenticated identity | Keep greeting mappings cosmetic, explicit and private; retain native Voice authorization for all actions | Prevent personalization from becoming an implicit permission grant |

References: [`run` and `FrameResult`](tools/recognition.py), [helper `validPresence`](node_helper.js), [fresh-presence runtime tests](tests/test_runtime.py) and [consumer contract](docs/index.md#fresh-presence-contract).

## Room calibration for letter stamps

Use `LetterScanner` with a 9–11 cm printed stamp at the filming position.
A local rehearsal can establish the best distance and angle for the existing
lens settings. [Synthetic decoder tests](tests/test_letter_marker.py)

## 2026-11-05 - Christmas camera implementation follow-through

- **Implemented:** fixed expected props and explicit keepsake previews now use
  the existing camera owner and the [source-owned gateway](tools/experience_gateway.py).
  Retain [race and state regressions](tests/test_experience_camera.py) on updates.
- **Opportunity:** physical stamp size, mirror reflections and low winter light
  affect recognition. **Proposed action:** rehearse LETTER then STAR at normal
  standing distance using the existing lens/exposure configuration. **Impact:**
  verifies the physical interaction without guessing calibration changes.
- **Opportunity:** unsaved photos intentionally disappear after worker restart.
  **Proposed action:** explain unavailable status and use explicit retake, as the
  [photo contract](docs/index.md#explicit-photo-actions-and-preview-contract) does.
  **Impact:** prevents silent replacement or stale-image saves.

## Follow-up after live activation - 2026-11-05

The authorized runtime/module/gateway activation is complete. The remaining
opportunity is a normal-distance family rehearsal: check instruction
readability, microphone recognition, a printed prop and a requested keepsake.
This measures the actual room without changing the existing calibration.
