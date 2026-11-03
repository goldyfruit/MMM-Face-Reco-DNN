Last Edit: Codex (GPT-6) - 2026-11-03 - Motive: Document tested fresh per-frame presence metadata and release 2.2.1 without changing authentication or cached login behavior.

# FAQ

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
