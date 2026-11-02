Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Evidence-based audit

## Fixed with tests

- Last-face distance reused in every preview label: [annotate](tools/recognition.py), [test_annotations_are_separate_and_use_per_face_distance](tests/test_recognition.py).
- Multi-face photos enrolled under one identity: [encode_dataset](tools/utils/enrollment.py), [enrollment tests](tests/test_encode.py).
- Preview annotation could contaminate saved training pixels: [CameraFrame.full_frame](tools/utils/camera.py) and [dataset tests](tests/test_recognition.py).
- Camera/request leaks on failed processing or trigger-thread startup: [run](tools/recognition.py), [runtime tests](tests/test_runtime.py).
- Invalid/mismatched galleries and out-of-bounds boxes: [FaceMatcher/FaceDetector](tools/utils/vision.py), [vision tests](tests/test_vision.py).

## Remaining limits

Private pickle galleries are trusted local input; do not load arbitrary remote pickle files. Existing automatically collected photos can contain prior mistakes and are not re-labelled by this fork. AEC/audio tuning is outside this module. This is not liveness detection or secure authentication. One public portrait with transformations cannot establish broad recognition accuracy. dlib embedding cost remains CPU-bound. The dependency `face-recognition-models` still needs the documented legacy pkg_resources compatibility dependency. See [performance](docs/performance.md) for measured scope and [suggestions](SUGGESTIONS.md) for next work.

Actual tests were run before writing this audit:110Python and3Node tests passed; lint passed. Final totals may grow as asset/CI checks are added.

Final pre-deployment checks:112 Python tests and3 Node tests pass; JavaScript lint passes. Official-model and public-fixture checksum checks pass. CI covers Python3.10/3.13 and Node24.

Final colour review found that libcamera sYCC is full-range while OpenCV I420 conversion assumes limited range. The portable decoder now preserves full Rec601 range; Pi5 can select direct RGB888 lores. New tests preserve gray32 as32 (the old conversion produced19). Real RGB camera canary passed with mean main/lores difference0.643 byte levels, untouched1920x1080 main image and restart. Final tests:119 Python,3 Node; lint passes. User confirmed recognition at normal standing distance before this colour-consistency correction.

Python format verification was added to CI using Ruff0.12.4; formatting-only changes preserve behaviour.119 Python tests,3 Node tests and JavaScript lint pass afterward.
