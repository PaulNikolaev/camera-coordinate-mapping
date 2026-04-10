"""Project-wide configuration constants for dataset IO and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

SourceName = Literal["top", "bottom"]

IMAGE_WIDTH: Final[int] = 3200
IMAGE_HEIGHT: Final[int] = 1800

EXPECTED_SPLITS: Final[tuple[str, str]] = ("train", "val")
VALID_SOURCES: Final[tuple[SourceName, SourceName]] = ("top", "bottom")
METADATA_DIR_NAMES: Final[tuple[str, ...]] = ("__MACOSX",)

SPLIT_FILENAME: Final[str] = "split.json"
DEFAULT_DATA_ROOT: Final[Path] = Path("coord_data")
DEFAULT_ARTIFACTS_DIR: Final[Path] = Path("artifacts")
DEFAULT_ARCHIVE_NAME: Final[str] = "test-task.zip"
DEFAULT_DATASET_URL: Final[str] = (
    "https://drive.google.com/file/d/1VVxx4I6T8xdtJnUnzsO-PnSnniGUnOwp/view?usp=drive_link"
)
MIN_POINTS_PER_FRAME_PAIR: Final[int] = 17
MAX_POINTS_PER_FRAME_PAIR: Final[int] = 22

COORDS_FILENAME_BY_SOURCE: Final[dict[str, str]] = {
    "top": "coords_top.json",
    "bottom": "coords_bottom.json",
}
COORDS_FILENAME_CANDIDATES_BY_SOURCE: Final[dict[str, tuple[str, ...]]] = {
    "top": ("coords_top.json", "coodrs_top.json"),
    "bottom": ("coords_bottom.json",),
}
