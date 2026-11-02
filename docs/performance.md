Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Performance evidence and limits

On mm01 Pi5 at2.8GHz, Python3.13/OpenCV4.10, seven repeats per scene:

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
