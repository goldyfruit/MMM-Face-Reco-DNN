"""Command-line gallery enrollment; importing this module does not run a job."""

from pathlib import Path
import sys

# Retain ``python tools/encode.py`` while using absolute package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.arguments import Arguments
from tools.utils.enrollment import encode_dataset


def main() -> int:
    """Enroll labelled images and report failure without erasing a valid gallery."""
    Arguments.prepareEncodingArguments()
    try:
        statistics = encode_dataset(
            dataset=Arguments.get("dataset"),
            output=Arguments.get("encodings"),
            detection_method=Arguments.get("detection_method"),
        )
    except (OSError, ValueError) as error:
        print(f"[ERROR] Enrollment failed: {error}", file=sys.stderr)
        return 1
    return 0 if statistics.output_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
