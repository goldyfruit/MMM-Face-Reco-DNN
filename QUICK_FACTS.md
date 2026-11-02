Last Edit: Codex - 2026-11-02 - Motive: Document the tested Pi camera and face-detection fork.

# Quick facts

| Field | Value |
|---|---|
| Package | mmm-face-reco-dnn |
| Version | 2.1.0; source of truth package.json |
| JavaScript entry | MMM-Face-Reco-DNN.js / node_helper.js |
| Python entry | tools/recognition.py:main; tools/encode.py:main |
| Gallery | Existing names/encodings pickle with128-value dlib descriptors |
| Optional detector | YuNet2023mar, OpenCV FaceDetectorYN |
| Camera | Picamera2 main plus optional ISP lores, lazy full-resolution frames |
| Tests | pytest tests; npm test; npm run lint |

[CameraStream](tools/utils/camera.py), [FaceDetector/FaceMatcher](tools/utils/vision.py), [RecognitionEngine](tools/recognition.py), [enrollment](tools/utils/enrollment.py).
