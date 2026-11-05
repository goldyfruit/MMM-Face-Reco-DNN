Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Add opt-in dated North Pole postcards with unchanged explicit photo consent.

# Installation and configuration

Install this fork's reviewed branch and run `npm ci`. Use a working Python camera environment containing Picamera2, system libcamera, OpenCV with `FaceDetectorYN`, NumPy, dlib and face_recognition. The tested Pi environment uses Python3.13, OpenCV4.10.0, NumPy2.2.4, dlib20.0.1 and face_recognition1.3.0. Keep the sensor-specific libcamera/Python binding ABI aligned; this fork does not install firmware or change voltage settings.

The existing `face-recognition-models` package depends on legacy `pkg_resources`; the tested environment retains its explicit setuptools80.9.0 compatibility pin. This remains dependency debt, not a NumPy2 incompatibility in the tested fork.

| Setting | Default | Purpose |
|---|---|---|
| method | dnn | Legacy dnn/HOG, haar or optional yunet |
| yunetScoreThreshold | 0.8 | Detection score cutoff; not a calibrated probability of identity |
| yunetNmsThreshold | 0.3 | YuNet box suppression |
| cameraLores | false | Process an ISP low-resolution stream |
| cameraLoresFormat | YUV420 | Full-range portable YUV; RGB888 recommended on Pi5 |
| cameraFrameRate | null | Optional sensor cap;15 FPS tested on Pi5 |
| detectionUpsample | 1 | Preserve legacy HOG/CNN behaviour;0 may miss small faces |
| identityMargin | 0 | Optional competing-identity distance gap; calibrate before enabling |
| previewQuality | 95 | JPEG quality for preview, independent of enrollment images |
| opencvThreads | 0 | Leave defaults at0; explicit1–64 controls OpenCV worker threads |

`CameraStream.capture()` in [camera.py](../tools/utils/camera.py) owns and releases each camera request. `CameraFrame.full_frame()` materializes a synchronized full-resolution image only when needed. [RecognitionEngine.recognize](../tools/recognition.py) retains dlib128 embeddings. [FaceMatcher.match](../tools/utils/vision.py) validates and caches the gallery. No automatic gallery regeneration occurs.

`tools/encode.py` still writes `encodings`/`names` pickle keys. [encode_dataset](../tools/utils/enrollment.py) skips unreadable/no-face/multi-face images, preserves top-level identity labels in nested folders, and replaces a valid gallery atomically only after usable enrollment. Review automatically collected photos before retraining; this is personalization, not liveness/authentication.

Back up the current module, user config, face gallery and dataset before switching branches. Preserve local lockfile changes and untracked tools. Keep original MIT notices and model attribution. For results and boundaries see [performance](performance.md), [audit](../AUDIT.md), [FAQ](../FAQ.md), [maintenance](../MAINTENANCE_REPORT.md).

## Advertised camera controls

| Setting | Default | Supported choices / behaviour |
|---|---|---|
| cameraAutofocus | default | Preserve existing focus; optional continuous, manual or face-window AF |
| cameraLensPosition | null | Optional finite nonnegative dioptres with manual/face AF and an advertised in-range control; 1.0 is about one metre, 0 is infinity |
| cameraExposureMode | normal | normal or short; only an advertised AeExposureMode value is applied |
| workerRestartDelay | 1000 | Integer 100–30000 ms base; exponential backoff capped at 30000 ms |
| workerRestartLimit | 3 | Integer 0–10; 0 disables automatic retry |

[CameraStream](../tools/utils/camera.py#L199) checks advertised controls and preserves unsupported/fixed-focus behavior. Continuous AF requests Normal range when available but remains opt-in: it selected a nearer focus and recognition regressed in the mirror trial. Foreground/reflections are a plausible explanation, not established by lens metadata alone. Normal exposure remains default; optional short exposure needs a real-room blur/noise comparison. Scalar camera metadata excludes pixels and identity names.

### Opt-in face-window autofocus

Set `cameraAutofocus:'face'` to target detected geometry instead of default metering. [FaceFocusController](../tools/utils/focus.py#L121) maps rotated/resized face boxes through the same capture's sensor crop, selects the largest valid face deterministically and requires advertised window metering/trigger support. Recognition is not required. The mode begins from an advertised manual seed (`cameraLensPosition`, or 1.0 dioptre) and verifies its LensPosition on a fresh frame before requesting a scan; unsupported ROI AF retains a safe baseline rather than silently switching to global continuous AF.

`camera.face_focus_status` provides bounded image-free state/geometry diagnostics. Pending scans/restoration defer identity changes and dataset extension while detection/preview continue. No-face cancellation permits normal logout. Scans time out after six seconds; failure imposes a global 30-second cooldown even when the ROI changes. A restoration unverified by metadata after two seconds faults the worker while faces are present instead of publishing unsettled identities. Physical focus/recognition validation is pending; do not infer success from mocked controls.

### Worker lifecycle

[run](../tools/recognition.py#L152) renews readiness after a processed/released capture at startup and resume, including no-face/no-preview operation. Activation generations also reconcile identity snapshots so the same known face reopens its session after quick pause/resume, while stale in-flight identity/enrollment results are rejected. The actual PythonShell text stdin formatter preserves plain commands with JSON stdout.

[Helper recovery](../node_helper.js#L27) clears stale sessions/preview immediately through [frontend reset](../MMM-Face-Reco-DNN.js#L405), waits for old-process closure, and bounds retries. Explicit stop/pause suppresses retries and queued activity; a process that ran at least 60 seconds gets a fresh early-failure budget. Normal multi-user logout callbacks bind users independently and emit logout after expiry. This remains personalization, not authentication/liveness.

Current validation: 210 Python tests pass under actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests; 22 Node tests and lint/format checks pass. See [camera tests](../tests/test_camera.py), [runtime tests](../tests/test_runtime.py) and [audit](../AUDIT.md). Primary AF semantics are documented in [libcamera control definitions](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml); [Picamera2 configuration](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py) shares the main crop with preview lores when aspect preservation is disabled.

## Fresh presence contract

Release 2.2.1 adds `FACE_RECOGNITION_PRESENCE` independently of legacy login/logout events. [`run`](../tools/recognition.py) sends the `presence` payload; [helper `validPresence`](../node_helper.js) validates it, and [the frontend](../MMM-Face-Reco-DNN.js) forwards the fixed metadata notification.

```json
{"version":1,"epoch":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sequence":1,"observed_at":1800000000.0,"available":true,"labels":["synthetic-subject"]}
```

The exact six fields are mandatory. `epoch` is a fresh 32-character lowercase hexadecimal worker UUID; `sequence` increases from one for each inference in that worker. `observed_at` is UNIX seconds recorded when the captured request is entered, before inference, so processing latency remains visible to the consumer. The helper accepts only finite nonnegative timestamps and positive safe integer sequences. Match labels are nonempty strings of at most 100 JavaScript code units, with Unicode control/format/surrogate characters rejected; at most eight entries leave the worker. Consumers can enforce stricter bounds.

Each settled inference publishes its current matches, including duplicate labels. A two-face frame remains two entries even if both match the same label; consumers must not deduplicate this into a single-person greeting. A settled no-face frame is available with an empty list. During pending autofocus or over-cap groups, availability is false with an empty list. Pause, shutdown and failed workers can stop publishing; consumers must expire missing/old snapshots and never use cached login state as a replacement.

No image, face geometry, descriptor, distance, gallery, spoken text, profile selection or authorization is included. This is not liveness detection or authentication. The separate optional Christmas greeting consumer opts in and uses its own freshness, single-face stability, private exact-label mapping and cooldown checks; existing native Voice permissions remain independent. This feature adds no network listener or greeting request client to the Face module.

Validation uses [synthetic worker tests](../tests/test_runtime.py) and [Node/frontend admission tests](../tests/test_node_helper.cjs): 217 Python tests on Python 3.13.11, 23 Node tests, JavaScript lint and targeted Python lint/format pass. No private camera/gallery was accessed for this feature.

## Local lost-letter stamp

`run()` gives `LetterScanner.scan()` a lazy frame callback while the existing
camera request is live. The scanner requires a fresh public rehearsal state,
an idle Voice projection and the pre-existing local socket mount. No additional
camera handle is opened. [Recognition worker](../tools/recognition.py) ·
[Stamp scanner](../tools/utils/letter_marker.py)

## Christmas props and keepsakes (2.3.0)

`ExperiencePropScanner.scan()` and `PhotoBooth.process_frame()` in
[experience_camera.py](../tools/utils/experience_camera.py) receive the same lazy
`CameraFrame.full_frame()` callback while [the worker](../tools/recognition.py)
still owns the existing camera request. They open no camera, change no focus or
exposure setting, and do not use face recognition to authorize an action.
[The lightweight state gate](../tools/utils/experience_state.py) requires fresh
connected public state, fresh Voice evidence, `protocol_enabled:true` and an
active episode. An unrelated presentation, video, paused scene or stale state
cannot trigger camera work. The fixed stamp scanner additionally requires
Voice listening and the exact current `expected_prop`.

### Shared mounts and optional photo destination

Keep the existing display state directory mounted read-only at
`/christmas-state`, and mount a shared gateway directory at
`/christmas-greetings` in the camera worker. The Voice owner service needs the
same public file and shared socket directory under its own paths. The fixed
socket is `/christmas-greetings/experience.sock` inside the camera container.
Use matching numeric UID ownership for the Voice owner and camera worker;
the source gateway creates a mode-0600 Unix socket and does not expose a
network listener or distribute Voice credentials to the mirror.

Set `christmasPhotoDirectory:'/christmas-photos'` in the existing Face module
configuration if using a persistent container photo volume. The directory must
be absolute, owned by the worker and private (mode 0700). It is created only
when a save is requested; saved JPEGs use mode 0600 and unique local filenames.
The default is the worker user's `~/.local/share/thalovant/christmas-photos`.
Do not point this directory at the face enrollment gallery. Existing camera
calibration and `extendDataset` behavior are independent and unchanged.

### Source-owned fixed prop gateway

Run this explicitly using the existing **Voice owner's Python environment**
where `thalovant_voice.control.ControlClient` is already installed:

```sh
/path/to/voice/python /path/to/MMM-Face-Reco-DNN/tools/experience_gateway.py \
  --state-file /actual/public/christmas-state/state.json \
  --socket /actual/shared/christmas-greetings/experience.sock
```

The `--state-file` and `--socket` paths are host/service paths corresponding
to the camera's read-only state and shared socket mounts. The gateway imports
only standard-library state validation; it needs neither OpenCV nor NumPy.
Its owner-default `ControlClient.from_file()` reads the already provisioned
local Voice endpoint. Keep the command in the normal owner-managed service
supervisor if automatic restart is needed; this source change does not install
or activate a system service.

The camera sends exactly `{version:1,revision,prop,observed_at}`. `process()` in
[experience_gateway.py](../tools/experience_gateway.py) rejects stale, wrong,
unknown or extended observations and rechecks the current episode before the
sole allowed request:

```python
client.call('POST', '/v1/ask', {
    'text': f'Christmas prop {prop} revision {revision}',
    'lang': 'en-US', 'speak': True,
}, timeout=5.0)
```

`prop` can only be letter/star/bell/gift. The current revision and expected prop
are independently checked by the native Protocol skill. Admission returns a
small immediate Unix-socket acknowledgement before a potentially slow owner
call, so the camera request is not held for network speech. Only one request
may be pending. A held prop is consumed once; exchanging a letter for the next
expected star immediately re-arms detection. Repeating the same held prop does
not re-arm on heartbeat/revision changes. If Voice is unavailable, remove and
present the prop again or use its normal voice opening command.

### Explicit photo actions and preview contract

The native Protocol producer emits optional `photo_action` values
none/capture/save/discard at the keepsake scene. A capture/retake advances the
state revision and records `photo_capture_revision` equal to that action's
revision. Save/discard retain that capture identity. Generic hints/repeats clear
the action to none without requesting another capture.

`PhotoBooth` holds the preview JPEG only in memory. Countdown begins only with
fresh listening evidence and is sampled at the existing worker interval, so
capture occurs on the next available frame after at least three seconds.
Frame materialization and publication recheck episode/revision/capture identity.
A save commits bytes only when their in-memory `preview_capture_revision`
exactly matches the native capture target. Skipping a retake revision while
polling can therefore never save an older photo. A restart cannot save a lost
preview. Exit, off, stale state, pause and discard release uncommitted bytes.

A new worker's first observed action may be an old persisted capture with fresh
bridge heartbeats. `PhotoBooth.process_frame()` consumes that revision and emits
unavailable without requesting pixels. Repeated heartbeats and reset cannot
revive it; a new explicit capture revision is required. Observing a ready
non-capture scene or off state first arms ordinary future capture. This guard
does not create a durable photo store or recover lost preview bytes.

`Print.printJson('christmas_photo', payload)` is validated by
[node_helper.js](../node_helper.js) before the [frontend](../MMM-Face-Reco-DNN.js)
forwards `CHRISTMAS_PHOTO_STATUS`. Payload keys are version/revision/status,
optional capture_revision, countdown-only seconds (1-3), and preview-only image
(base64 JPEG). Status is countdown/preview/saved/discarded/unavailable. JPEG
bytes are capped at 500000 and the entire public event below 700000 bytes.
No filename, label, gallery data, transcript or control credential is sent.
The [Christmas renderer](../../MMM-Thalovant-ChristmasProtocol/docs/index.md)
checks the known sender and current action/capture revision before showing it.

### Optional North Pole postcard

`ExperienceView` in [experience_state.py](../tools/utils/experience_state.py)
accepts optional keepsake-only `photo_style:'plain'|'postcard'`. An absent field
means plain. The authenticated native producer offers “take Christmas postcard”
and “prends la carte postale de Noël” separately from ordinary photo capture.
See [Christmas Protocol: ExperienceDirector](../../thalovant-skill-christmas-protocol/docs/experience.md)
and [its bridge contract](../../thalovant-skill-christmas-protocol/docs/bridge.md).
The camera never chooses a new style when saving an existing preview.

On explicit postcard capture, [PhotoBooth.process_frame](../tools/utils/experience_camera.py)
calls [render_postcard](../tools/utils/postcard.py) once on the shared current
frame. The compositor preserves the complete photo with proportional resizing,
adds an ivory mat, fine evergreen/gold lines and original postal artwork outside
it, and renders `NORTH POLE` or `PÔLE NORD`. Locale prefixes `fr` select the French
caption and numeric date order; other well-formed source locales use English.
The circumflex is drawn as a vector, so no new fonts, external images, Pillow
or font downloads are required at runtime.

The trusted capture timestamp is converted with stdlib `zoneinfo` using optional
`christmasPhotoTimezone:'America/Toronto'`. Set another available IANA zone such
as `Europe/Paris` in the existing module config if needed. The system timezone
database must contain it; invalid/unavailable configured zones fail argument
validation. The host's ambient timezone is not used to guess the family date.
Both date and artwork are frozen into the preview JPEG before publication and
the exact bytes are saved after explicit consent. A style change needs another
capture revision; mismatched previews and style changes during capture are
discarded. All existing identity, freshness, listening and retake guards remain.

The complete postcard stays below the existing 1200-pixel dimension and
500000-byte JPEG limits. Rendering runs only on explicit postcard capture,
not on every frame, polling update or save. Plain capture uses the existing
encoding path. [Postcard tests](../tests/test_postcard.py) and
[PhotoBooth tests](../tests/test_experience_camera.py) use synthetic pixels and
fixed dates, including Toronto/Paris midnight boundaries and exact preview/save
byte equality. Source tests and synthetic visual review do not measure a real
camera's exposure, sharpness or room lighting.

[Camera/gateway regressions](../tests/test_experience_camera.py),
[worker ownership checks](../tests/test_runtime.py), and
[helper/frontend tests](../tests/test_node_helper.cjs) cover the implementation
with synthetic pixels, fake clocks and stub owner clients. No physical room
acceptance or current-device activation is claimed.
