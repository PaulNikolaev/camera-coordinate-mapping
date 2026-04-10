"""Shared argparse helpers for project entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from solution.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_ROOT


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add the canonical dataset root argument to a parser."""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the unpacked coord_data directory (default: %(default)s).",
    )


def add_artifacts_dir_argument(
        parser: argparse.ArgumentParser,
        help_text: str,
) -> None:
    """Add the canonical artifacts directory argument to a parser."""
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=help_text,
    )


def add_output_metrics_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional metrics output path argument to a parser."""
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Where to save metrics.json (default: <artifacts-dir>/metrics.json).",
    )


def add_strict_argument(parser: argparse.ArgumentParser) -> None:
    """Add the strict filtering toggle to a parser."""
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict point-count filtering instead of allow_partial preparation.",
    )
