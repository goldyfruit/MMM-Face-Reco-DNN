"""Shared parsing helpers for the legacy command-line interface."""

import argparse


class Helper:
    """Convert JSON-like command-line values without Python truthiness bugs."""

    @staticmethod
    def str2bool(value: str | bool) -> bool:
        """Accept explicit boolean words and reject ambiguous strings."""
        if isinstance(value, bool):
            return value
        lowered = value.lower()
        if lowered in ('yes', 'true', 't', 'y', '1'):
            return True
        if lowered in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
