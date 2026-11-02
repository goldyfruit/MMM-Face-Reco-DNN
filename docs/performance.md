Last Edit: Codex - 2026-11-02 - Motive: Document advertised camera controls and tested worker/session recovery.

# Performance evidence and limits

On the tested Pi5 at2.8GHz, Python3.13/OpenCV4.10, seven repeats per scene:

| Scene | Legacy HOG median | YuNet median |
|---|---:|---:|
| Public portrait500x500 | 152.3ms | 56.2ms |
| Dim portrait | 152.7ms | 60.9ms |
| Small crop on500x281 frame | 84.7ms | 26.1ms |
| Two faces on500x281 frame | 84.9ms | 25.4ms |
| Blank frame | 85.5ms | 28.0ms |

Both detectors returned the expected face counts on all nine public test scenes at YuNet cutoff0.8. Cutoff0.9 missed the smallest crop. This is approximately2.5–3.5x faster detection in this bounded test, not whole-pipeline speed or population-level accuracy. It uses one public-domain NASA image and transformed scenes, not a diverse held-out dataset. Per-face dlib descriptor computation remains CPU work.

Reproduce with `python tools/benchmark.py --repeats 7 --output benchmark-results.json` in a runtime with dlib/face_recognition. [benchmark.scenarios](../tools/benchmark.py) constructs all scenes. The model is OpenCV Zoo's2023mar YuNet, chosen for OpenCV4.x compatibility; source/checksum/license are bundled.

The real Arducam64MP canary verified512x288 YUV420 processing,500x281 BGR detector input, untouched1920x1080 main pixels, capture-request release and stop/restart at15 FPS. It saved no images. First capture included~771ms startup; subsequent capture-plus-detection samples were~50–69ms, including waiting for the next sensor frame. No people were present in that canary, so it did not establish live identity accuracy.

Accuracy-related fixes have deterministic tests: independent face scores, safe box bounds, ambiguous enrollment rejection and annotation isolation. Optional identity margin is disabled by default to avoid uncalibrated false rejections. Existing face galleries stay private and unchanged.

Primary references: [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [public NASA fixture provenance](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut), [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).

Raw public benchmark: [pi5-detection.json](benchmarks/pi5-detection.json).

Live deployment at500ms polling measured23.5% of one CPU core over30 seconds,60.9 C and no throttling. This is a scene-dependent sample, not a controlled family-accuracy comparison. Existing gallery bytes stayed unchanged; user confirmed correct recognition at normal standing distance.

Final Pi5 rollout uses RGB888 lores directly, avoiding a full-/limited-range YUV mismatch. The real-camera RGB canary measured mean main/lores difference0.643 byte levels after resizing, verified full-resolution pixels and restart, and saved no images. Portable YUV420 now uses full-range Rec601 conversion with regression tests.

## YuNet thread scaling

On the same500x281 public scene, nine repeats after warm-up:4threads measured22.8–25.0ms median wall time and65–68ms CPU time;2threads measured22.7–25.4ms wall and41–43ms CPU. One thread measured27.6ms wall/26.8ms CPU. The Pi uses explicit `opencvThreads:2` to reduce detector CPU consumption by about36% with similar wall latency. Default0 retains library behaviour for other users. Reproduce with `python tools/benchmark.py --opencv-threads 2`.

## Camera-control and recovery follow-up

Version 2.2.0 does not claim another detector FPS or recognition-accuracy gain. Existing focus remains the generic default. A continuous-AF mirror trial moved to a reported lens position of about 5.79 rather than the prior 1.0 and user recognition regressed; foreground/reflections are a plausible cause. Optional face-window AF targets detected geometry using the actual capture crop/rotation instead of relying on default camera metering. It does not require recognition to acquire a target. A subsequent local trial observed real scanning-to-focus transitions and usable PDAF. The user confirmed recognition at normal standing distance and after small distance changes; vendor lens values are not measurements of the person's distance.

The [FaceFocusController](../tools/utils/focus.py#L121) bounds each scan to six seconds and imposes a global 30-second cooldown after failure, even if the face ROI moves. It verifies a fresh manual-seed frame before scanning/resuming. Restoration requires matching lens metadata; if unverified after two seconds while faces remain visible, the worker faults rather than treating the frame as settled. Identity changes/enrollment wait through bounded scans/restoration; preview continues and no-face cancellation allows normal logout. Optional short AE remains a room-dependent blur/noise tradeoff.

Worker failure/pause immediately clears stale sessions and preview rather than waiting for the ordinary 15-second disappearance grace. Fake-clock tests verify bounded 1/2/4-second early retries, no replacement before process closure, stop/pause semantics and correct multi-user logout timing. Activation generations renew readiness and identity snapshots after quick stop/start, including the same known face. The installed PythonShell formatter test verifies plain stdin commands with JSON stdout.

Current source validation: 210 Python tests on actual Python 3.10.22 and 3.13.11, including 80 camera/controller and 50 runtime tests, plus 22 Node tests and lint/format checks. [Camera tests](../tests/test_camera.py), [runtime tests](../tests/test_runtime.py) and [Node tests](../tests/test_node_helper.cjs) use synthetic data and controls. The separate in-room hardware check passed; those unit results alone do not establish physical focus or general recognition accuracy. No private media or gallery data was exported.
