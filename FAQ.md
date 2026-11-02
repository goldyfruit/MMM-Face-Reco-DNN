Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

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
