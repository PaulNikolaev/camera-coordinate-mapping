"""Dataset loading helpers for coordinate-mapping sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solution.config import (
    COORDS_FILENAME_CANDIDATES_BY_SOURCE,
    COORDS_FILENAME_BY_SOURCE,
    EXPECTED_SPLITS,
    SPLIT_FILENAME,
    SourceName,
)


@dataclass(frozen=True, slots=True)
class CoordinatePoint:
    """A single labeled point in image pixel coordinates."""

    number: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FramePairAnnotation:
    """A labeled correspondence between a door2 frame and one side-camera frame."""

    session_rel_path: str
    source: SourceName
    door2_image_path: Path
    source_image_path: Path
    door2_points: tuple[CoordinatePoint, ...]
    source_points: tuple[CoordinatePoint, ...]


@dataclass(frozen=True, slots=True)
class SessionAnnotations:
    """Annotations collected for one session directory."""

    session_rel_path: str
    top_pairs: tuple[FramePairAnnotation, ...]
    bottom_pairs: tuple[FramePairAnnotation, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Validated sessions that belong to one split."""

    name: str
    session_paths: tuple[str, ...]
    sessions: tuple[SessionAnnotations, ...]


@dataclass(frozen=True, slots=True)
class CoordinateDataset:
    """Loaded dataset object keyed by train/val split."""

    data_root: Path
    train: DatasetSplit
    val: DatasetSplit

    def get_split(self, split_name: str) -> DatasetSplit:
        """Return a split object by its canonical name."""
        if split_name == "train":
            return self.train
        if split_name == "val":
            return self.val
        raise KeyError(f"Unknown split: {split_name!r}. Expected one of {EXPECTED_SPLITS}.")


def read_split_file(data_root: Path) -> dict[str, list[str]]:
    """Read `split.json` from the dataset root."""
    split_path = data_root / SPLIT_FILENAME
    with split_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("split.json must contain a JSON object with 'train' and 'val' lists.")

    result: dict[str, list[str]] = {}
    for split_name in EXPECTED_SPLITS:
        split_entries = raw_data.get(split_name)
        if not isinstance(split_entries, list):
            raise ValueError(f"split.json field '{split_name}' must be a list of session paths.")

        normalized_entries: list[str] = []
        for entry in split_entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"split.json field '{split_name}' must contain non-empty string paths."
                )
            normalized_entries.append(entry.replace("\\", "/"))

        result[split_name] = normalized_entries

    return result


def build_session_path(data_root: Path, session_rel_path: str) -> Path:
    """Build an absolute path to a session from the dataset root."""
    return data_root / Path(session_rel_path)


def load_dataset(data_root: Path) -> CoordinateDataset:
    """Load train/val sessions and annotation JSON files into memory."""
    split_map = read_split_file(data_root)

    train_sessions = tuple(load_session_annotations(data_root, session) for session in split_map["train"])
    val_sessions = tuple(load_session_annotations(data_root, session) for session in split_map["val"])

    return CoordinateDataset(
        data_root=data_root,
        train=DatasetSplit(
            name="train",
            session_paths=tuple(split_map["train"]),
            sessions=train_sessions,
        ),
        val=DatasetSplit(
            name="val",
            session_paths=tuple(split_map["val"]),
            sessions=val_sessions,
        ),
    )


def load_session_annotations(data_root: Path, session_rel_path: str) -> SessionAnnotations:
    """Load both top and bottom annotation files for one session."""
    session_path = build_session_path(data_root, session_rel_path)
    top_pairs = load_annotation_pairs(data_root, session_rel_path, session_path, "top")
    bottom_pairs = load_annotation_pairs(data_root, session_rel_path, session_path, "bottom")

    return SessionAnnotations(
        session_rel_path=session_rel_path,
        top_pairs=top_pairs,
        bottom_pairs=bottom_pairs,
    )


def load_annotation_pairs(
    data_root: Path,
    session_rel_path: str,
    session_path: Path,
    source: SourceName,
) -> tuple[FramePairAnnotation, ...]:
    """Load one source annotation file (`coords_top.json` or `coords_bottom.json`)."""
    annotation_path = resolve_annotation_path(session_path, source)

    with annotation_path.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)

    if not isinstance(raw_records, list):
        raise ValueError(f"{annotation_path} must contain a JSON array of frame-pair records.")

    return tuple(
        parse_frame_pair_record(
            data_root=data_root,
            session_rel_path=session_rel_path,
            source=source,
            raw_record=raw_record,
        )
        for raw_record in raw_records
    )


def parse_frame_pair_record(
    data_root: Path,
    session_rel_path: str,
    source: SourceName,
    raw_record: Any,
) -> FramePairAnnotation:
    """Convert one raw JSON record into a typed frame-pair annotation."""
    if not isinstance(raw_record, dict):
        raise ValueError(
            f"Annotation record in session '{session_rel_path}' and source '{source}' must be an object."
        )

    session_path = build_session_path(data_root, session_rel_path)
    door2_image_raw = _require_non_empty_str(raw_record, "file1_path")
    source_image_raw = _require_non_empty_str(raw_record, "file2_path")
    door2_points = parse_points(raw_record.get("image1_coordinates"), "image1_coordinates")
    source_points = parse_points(raw_record.get("image2_coordinates"), "image2_coordinates")

    return FramePairAnnotation(
        session_rel_path=session_rel_path,
        source=source,
        door2_image_path=resolve_session_image_path(
            session_path=session_path,
            raw_path=door2_image_raw,
            expected_camera_dir="door2",
        ),
        source_image_path=resolve_session_image_path(
            session_path=session_path,
            raw_path=source_image_raw,
            expected_camera_dir=source,
        ),
        door2_points=door2_points,
        source_points=source_points,
    )


def parse_points(raw_points: Any, field_name: str) -> tuple[CoordinatePoint, ...]:
    """Parse a list of point dicts into typed coordinates."""
    if not isinstance(raw_points, list):
        raise ValueError(f"Field '{field_name}' must be a list of coordinates.")

    points: list[CoordinatePoint] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            raise ValueError(f"Each item in '{field_name}' must be an object with number/x/y.")

        number = raw_point.get("number")
        x = raw_point.get("x")
        y = raw_point.get("y")

        if not isinstance(number, int):
            raise ValueError(f"Coordinate field '{field_name}.number' must be an integer.")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Coordinate field '{field_name}.x/.y' must be numeric.")

        points.append(CoordinatePoint(number=number, x=float(x), y=float(y)))

    return tuple(points)


def resolve_session_image_path(
    session_path: Path,
    raw_path: str,
    expected_camera_dir: str,
) -> Path:
    """Map a raw annotation path to the extracted JPG inside the current session."""
    normalized_raw_path = raw_path.replace("\\", "/")
    image_name = Path(normalized_raw_path).name

    if not image_name:
        raise ValueError(f"Annotation image path is empty: '{raw_path}'.")

    return session_path / expected_camera_dir / image_name


def resolve_annotation_path(session_path: Path, source: SourceName) -> Path:
    """Resolve the annotation file path, including known filename typos in the archive."""
    for candidate_name in COORDS_FILENAME_CANDIDATES_BY_SOURCE[source]:
        candidate_path = session_path / candidate_name
        if candidate_path.exists():
            return candidate_path

    expected_path = session_path / COORDS_FILENAME_BY_SOURCE[source]
    raise ValueError(f"Missing annotation file: '{expected_path}'.")


def _require_non_empty_str(raw_record: dict[str, Any], field_name: str) -> str:
    """Fetch a non-empty string field from a raw JSON object."""
    value = raw_record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return value
