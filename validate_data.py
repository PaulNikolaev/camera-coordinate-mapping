"""Download, extract, and validate the dataset in one command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solution.config import DEFAULT_ARCHIVE_NAME, DEFAULT_DATA_ROOT, DEFAULT_DATASET_URL
from solution.validation import (
    DatasetDownloadError,
    DatasetValidationError,
    prepare_and_validate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download, extract, and validate the coordinate-mapping dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the unpacked coord_data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DATASET_URL,
        help="Google Drive sharing URL for the dataset archive.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path(DEFAULT_ARCHIVE_NAME),
        help="Where to store the downloaded ZIP archive (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded ZIP file after successful extraction and validation.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the archive and replace an existing coord_data directory.",
    )
    return parser


def main() -> int:
    """Run the end-to-end dataset preparation flow."""
    args = build_parser().parse_args()

    try:
        dataset = prepare_and_validate_dataset(
            data_root=args.data_root,
            dataset_url=args.url,
            archive_path=args.archive_path,
            keep_archive=args.keep_archive,
            force_download=args.force_download,
        )
    except (DatasetDownloadError, DatasetValidationError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1

    print(
        "Dataset is ready. "
        f"Validated {len(dataset.train.sessions)} train sessions and "
        f"{len(dataset.val.sessions)} val sessions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
