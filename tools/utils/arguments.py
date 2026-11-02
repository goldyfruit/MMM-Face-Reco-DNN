"""Validated command-line settings shared by enrollment and recognition."""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

# Retain imports by legacy standalone scripts that put tools/ on sys.path.
if __package__ == "utils":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.utils.helper import Helper


class Arguments:
    """Keep the legacy settings API while accepting explicit test arguments."""

    args: dict[str, Any] = {}

    @staticmethod
    def get(name: str) -> Any:
        """Return a parsed setting by its historical name."""
        return Arguments.args[name]

    @staticmethod
    def prepareEncodingArguments(argv: Sequence[str] | None = None) -> None:
        """Parse enrollment inputs without starting a camera."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--dataset", default="../dataset/")
        parser.add_argument("-e", "--encodings", default="../model/encodings.pickle")
        parser.add_argument(
            "-d", "--detection-method", choices=("hog", "cnn"), default="hog"
        )
        Arguments.args = vars(parser.parse_args(argv))

    @staticmethod
    def prepareRecognitionArguments(argv: Sequence[str] | None = None) -> None:
        """Parse legacy options and opt-in CPU/ISP detector improvements."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", "--cascade", default="../model/haarcascade_frontalface_default.xml"
        )
        parser.add_argument("-e", "--encodings", default="../model/encodings.pickle")
        parser.add_argument(
            "-r", "--rotateCamera", type=int, choices=(-1, 0, 1, 2), default=-1
        )
        parser.add_argument(
            "-m", "--method", choices=("dnn", "haar", "yunet"), default="dnn"
        )
        parser.add_argument(
            "-d", "--detectionMethod", choices=("hog", "cnn"), default="hog"
        )
        parser.add_argument("-i", "--interval", type=int, default=2000)
        parser.add_argument("-o", "--output", type=int, choices=(0, 1), default=0)
        parser.add_argument("-omm", "--outputmm", type=int, choices=(0, 1), default=0)
        parser.add_argument(
            "-eds", "--extendDataset", type=Helper.str2bool, default=False
        )
        parser.add_argument("-ds", "--dataset", default="../dataset/")
        parser.add_argument("-t", "--tolerance", type=float, default=0.6)
        parser.add_argument("-br", "--brightness", type=float, default=0)
        parser.add_argument("-co", "--contrast", type=float, default=0)
        parser.add_argument("-res", "--resolution", default="1920,1080")
        parser.add_argument("-pw", "--processWidth", type=int, default=500)
        parser.add_argument(
            "-roon", "--run-only-on-notification", type=int, choices=(0, 1), default=0
        )
        parser.add_argument("--cameraLores", type=Helper.str2bool, default=False)
        parser.add_argument(
            "--cameraLoresFormat", choices=("YUV420", "RGB888"), default="YUV420"
        )
        parser.add_argument("--cameraFrameRate", type=float, default=None)
        parser.add_argument(
            "--opencvThreads",
            type=int,
            default=0,
            help="OpenCV worker threads (0 preserves the library default; maximum 64)",
        )
        parser.add_argument(
            "--detectionUpsample", type=int, choices=(0, 1, 2), default=1
        )
        parser.add_argument(
            "--yunetModel",
            default=str(
                Path(__file__).resolve().parents[2]
                / "model/face_detection_yunet_2023mar.onnx"
            ),
        )
        parser.add_argument("--yunetScoreThreshold", type=float, default=0.8)
        parser.add_argument("--yunetNmsThreshold", type=float, default=0.3)
        parser.add_argument("--yunetTopK", type=int, default=5000)
        parser.add_argument("--identityMargin", type=float, default=0.0)
        parser.add_argument("--previewQuality", type=int, default=95)
        parsed = vars(parser.parse_args(argv))
        if parsed["interval"] <= 0:
            parser.error("interval must be positive")
        if not 0 <= parsed["opencvThreads"] <= 64:
            parser.error("opencvThreads must be between 0 and 64")
        if not 1 <= parsed["previewQuality"] <= 100:
            parser.error("previewQuality must be between 1 and 100")
        if not 0 < parsed["tolerance"] <= 2 or not 0 <= parsed["identityMargin"] <= 2:
            parser.error("tolerance and identityMargin are outside their valid ranges")
        Arguments.args = parsed
