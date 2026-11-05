Last Edit: Codex (GPT-6) - 2026-11-05 - Motive: Require a valid ready scene before arming photo capture after startup or reconnection.

# FAQ

## How do I choose a North Pole postcard instead of a plain photo?

At the active keepsake scene, say “take Christmas postcard” or “prends la carte
postale de Noël”. Plain photo capture remains available. Both choices use the
same countdown and in-memory preview; save/discard remain explicit. Choosing
another style requires a new capture, so a save never silently decorates an
older plain photo. [PhotoBooth](tools/utils/experience_camera.py) binds the
preview to its capture revision and style. The native skill and display bridge
need matching optional `photo_style` support; an older state defaults to plain.

## Which date and language appear on the postcard?

[render_postcard](tools/utils/postcard.py) uses the original capture timestamp
in `christmasPhotoTimezone` (default `America/Toronto`), not the save date or
host timezone. Source locales beginning with `fr`, including `fr-CA`, use
`PÔLE NORD` and day/month/year; others use English and year/month/day. The
existing OpenCV renderer draws the French accent directly, without a new font
dependency. A configured IANA zone must exist in the system timezone database.
The composed image stays in memory until explicit save, which writes exactly
the previewed bytes. See [the postcard setup](docs/index.md#optional-north-pole-postcard)
and [fixed-date tests](tests/test_postcard.py).

## Must I regenerate my face gallery?
No. `RecognitionEngine` uses the same dlib128 descriptor format and `FaceMatcher` accepts the existing names/encodings structure. Back it up first; this fork does not silently re-encode it.

## Does this prove recognition accuracy improved for my family?
No. Public fixture tests show detection counts and timing only. Correctness tests fix real labeling/enrollment bugs. Test your own camera lighting and held-out photos locally; never upload private galleries.

## Why are group training photos skipped?
`encode_dataset` cannot infer which face belongs to the folder label. Skipping ambiguous photos prevents accidentally training another person under that identity. Empty/failed enrollment keeps the previous gallery.

## Does lores reduce dataset resolution?
No. `CameraFrame.full_frame()` gets untouched full-resolution pixels from the same request on a new single-subject login; the processing/preview stream is separate. Group frames are not automatically enrolled as one identity.

## What if the ISP stream is unsupported?
Configuration fails clearly; set `cameraLores:false` for the legacy main-stream path. No silent camera/backend substitution occurs.

## Can I lower the matching tolerance or enable identityMargin?
`FaceMatcher.match()` supports both, but calibrate them locally on held-out genuine/impostor examples. Lower values are not an automatic improvement; false rejection can increase.

## Why did a failed camera leave a person logged in?

The previous helper handled only one error path and did not reset browser sessions. `worker_availability()` in [node_helper.js](node_helper.js) now reports failure on spawn/Python/close events. `reset_sessions()` in [MMM-Face-Reco-DNN.js](MMM-Face-Reco-DNN.js) clears sessions, pending logout timers and the preview immediately. It applies the configured no-face visibility state rather than waiting15seconds. [Recovery tests](tests/test_node_helper.cjs) exercise failure, duplicate/stale events, paused operation and explicit shutdown. This remains personalization, not secure authentication.

## Does recovery endlessly restart a broken camera?

No. It waits for actual process closure, then defaults to3 early retries with1/2/4second backoff. A process that ran at least60seconds starts a fresh recovery budget when it later closes. `workerRestartLimit:0` disables retries. Normal ranges are0–10retries and100–30000ms base delay; invalid settings use safe defaults. After the limit, check the local camera/configuration and reload the module. A process that cannot close is not overlapped by a second camera owner.

## Which focus mode should I use?

The default leaves existing camera focus unchanged. Optional `cameraAutofocus:'continuous'` requests advertised Continuous/Normal controls, but glass and foreground objects can attract autofocus. A trial on the mirror focused closer and the user reported failed recognition, so continuous AF is not the generic default. For a fixed standing distance, optional `'manual'` with `cameraLensPosition:1.0` requests roughly one metre; `0` is infinity. [CameraStream.__init__](tools/utils/camera.py) checks advertised manual mode and the lens range before applying the position. An explicit lens position is accepted only with manual or face-targeted AF. Optional `cameraAutofocus:'face'` starts from a supported seed then requests autofocus within a detected face window, rather than default metering in a reflective scene. Targeting does not depend on identity recognition. Validate focus at the actual distance and lighting without changing identity thresholds; mocked controls cannot establish image sharpness.

## Should I select short exposure for every room?

No. `cameraExposureMode:'short'` asks the advertised AE profile to favour shorter exposure, which may require more gain/noise in a dim room. Normal remains the default. `CameraFrame.metadata` and `CameraStream.latest_metadata` expose only allowlisted finite timing/focus/exposure fields for local diagnosis; images and identities are excluded.

## Why is readiness shown again after resume?

External pause immediately clears stale sessions and preview, and the helper discards queued events while paused or closing. Resume becomes available only after another successfully processed capture, even with no faces and preview disabled. [The runtime tests](tests/test_runtime.py) include quick stop/start between captures with both known and empty scenes; the same known face is announced again after the frontend clears its session. [The installed PythonShell formatter test](tests/test_node_helper.cjs) verifies plain `stop`/`start` stdin lines with JSON stdout; relying on JSON-mode's default stdin formatter would quote commands and silently prevent pausing.

## Can an autofocus scan trigger a stranger greeting?

Opt-in face AF defers new identity login/logout and dataset extension while the controller reports a pending scan or lens restoration. Detection and preview continue, and the existing identity is held only during the bounded transition. A no-face result cancels scanning and allows normal logout, so disappearance does not keep a ghost session. Timeout/failure requests restoration of a safe lens baseline and imposes a global 30-second failure cooldown, even if the ROI moves. Resume waits for a fresh frame confirming the seed. If restoration metadata is missing or wrong after two seconds while a face is present, the worker faults instead of classifying the unverified input. Tests cover known face → blurred unknown during scan → settled known face without an intervening guest greeting. This remains personalization, not authentication.

## Can another module tell who is present right now without stale login sessions?

Use `FACE_RECOGNITION_PRESENCE`, emitted by [the Face frontend](MMM-Face-Reco-DNN.js) from the separately validated per-inference snapshots in [the helper](node_helper.js). Do not treat `USERS_LOGIN`, cached users or delayed logout as current-frame evidence. The [fresh-presence contract](docs/index.md#fresh-presence-contract) uses a new random worker epoch, positive increasing sequence and capture-enter wall timestamp. An empty settled frame has `available:true, labels:[]`; an unsettled focus transition or more than eight matches has `available:false, labels:[]`. Two same-label matches remain two entries. A paused or failed worker may send nothing; consumers need their own short expiry.

## Does the presence signal authorize an action or identify a child profile?

No. It contains bounded match labels only and does not create any authentication mapping. It carries no image, gallery, face coordinates or confidence values. A separate explicitly enabled consumer may use a private exact-label allowlist for a cosmetic greeting, with its own freshness, stability and cooldown checks. Voice actions still require their existing authenticated connection. The Face module does not turn observed labels into commands, speech or permissions.

## How should I hold the North Pole stamp?

Arm the lost-letter rehearsal, then hold the printed square still and flat
toward the camera. Keep its white border clear. Lower the prop for a beat
between takes so `LetterScanner.scan()` observes two empty samples.
[Scanner implementation](tools/utils/letter_marker.py)

## How do I use a physical Christmas prop?

Print one of the fixed stamps LETTER/STAR/BELL/GIFT with the QR contents
`THALOVANT:PROP:LETTER`, `THALOVANT:PROP:STAR`, `THALOVANT:PROP:BELL`, or
`THALOVANT:PROP:GIFT`. Hold the currently requested prop toward the camera.
The active episode and revision authorize its effect; arbitrary QR text is
ignored. [Scanner and PhotoBooth](tools/utils/experience_camera.py) reuse the
existing camera request. [Gateway setup](docs/index.md#christmas-props-and-keepsakes-230)
requires the owner Voice service and shared socket mount. Voice commands remain
available when a printed stamp cannot be read.

## Does the keepsake flow save photos automatically?

No. “Take Christmas photo” starts a quiet countdown and an in-memory preview.
Only “save Christmas photo” writes that exact preview into the private configured
photo folder. Retake replaces it; discard, exit, Protocol off, worker pause and
stale state release uncommitted bytes. The saved photo is not face enrollment.
An old preview cannot satisfy a save for a newer capture revision. A worker
restart loses its unsaved preview and asks for another capture rather than
silently taking or saving a replacement.

If a new camera worker first sees an already-pending capture, it marks that
revision unavailable and suppresses its heartbeats. Use a fresh retake after
the camera is ready. An old persisted action is not treated as a new request.
[`test_worker_restart_consumes_persisted_capture_until_new_explicit_retake`](tests/test_experience_camera.py)
also checks that a reset cannot revive the blocked action.

## Why did the photo say unavailable?

Version 2.4.1 deliberately treats startup/reconnection as unready until a valid
active scene is observed. Missing, malformed, disconnected, off, stale and
presentation-hidden input cannot arm the camera. If its first valid view is an
already-pending capture, use a fresh retake. A worker that already observed a
valid non-capture scene admits the next explicit request normally. See
`test_unready_startup_cannot_arm_persisted_capture_after_reconnection` in
[PhotoBooth tests](tests/test_experience_camera.py).

There may be no preview for the current capture revision, the scene may have
changed during capture, encoding may have failed, or the photo destination may
not be an owned private absolute directory. Retake while the keepsake scene is
active; configure a persistent writable photo mount for saves. Camera calibration
is not changed by the feature. See [photo contract](docs/index.md#explicit-photo-actions-and-preview-contract)
and [regressions](tests/test_experience_camera.py).

## Is the connected experience installed on the mirror?

The coordinated release was activated on mm01 on 2026-11-05. Use the configured
wake phrase with “start Christmas family adventure”. Camera props need the
matching printed card at the requested step; saving a keepsake is explicit.
Use “pause Christmas adventure” or “disable Christmas Protocol” for quiet.
The final deployment evidence distinguishes observed player state from
physical hearing/recognition.
