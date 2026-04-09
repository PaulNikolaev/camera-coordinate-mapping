"""Dataset loading helpers for coordinate-mapping sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solution.config import (
    COORDS_FILENAME_CANDIDATES_BY_SOURCE,
    COORDS_FILENAME_BY_SOURCE,
    EXPECTED_SPLITS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_POINTS_PER_FRAME_PAIR,
    MIN_POINTS_PER_FRAME_PAIR,
    SPLIT_FILENAME,
    VALID_SOURCES,
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


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One matched point pair used for source->door2 training."""

    split_name: str
    session_rel_path: str
    source: SourceName
    point_number: int
    door2_image_path: Path
    source_image_path: Path
    x_src: float
    y_src: float
    x_door2: float
    y_door2: float


@dataclass(slots=True)
class TrainingSourceReport:
    """Aggregated preparation stats for one source camera."""

    source: SourceName
    records_seen: int = 0
    records_used: int = 0
    records_dropped: int = 0
    matched_samples: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source": self.source,
            "records_seen": self.records_seen,
            "records_used": self.records_used,
            "records_dropped": self.records_dropped,
            "matched_samples": self.matched_samples,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


@dataclass(slots=True)
class TrainingDataReport:
    """Serializable summary of data cleaning and sample preparation."""

    split_name: str
    strict: bool
    sessions_seen: int = 0
    sessions_dropped: int = 0
    records_seen: int = 0
    records_used: int = 0
    records_dropped: int = 0
    matched_samples: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    sources: dict[SourceName, TrainingSourceReport] = field(
        default_factory=lambda: {
            "top": TrainingSourceReport(source="top"),
            "bottom": TrainingSourceReport(source="bottom"),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "split_name": self.split_name,
            "strict": self.strict,
            "sessions_seen": self.sessions_seen,
            "sessions_dropped": self.sessions_dropped,
            "records_seen": self.records_seen,
            "records_used": self.records_used,
            "records_dropped": self.records_dropped,
            "matched_samples": self.matched_samples,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "sources": {
                source: source_report.to_dict()
                for source, source_report in self.sources.items()
            },
        }


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


def build_training_samples(
        data_root: Path,
        split_name: str = "train",
        strict: bool = False,
) -> tuple[tuple[TrainingSample, ...], TrainingDataReport]:
    """Build flat point-level training samples for one split.

    Args:
        data_root: Path to the unpacked `coord_data` dataset root.
        split_name: Which split from `split.json` to process.
        strict: When `True`, reject frame pairs whose point counts are outside
            the task-defined `17-22` range. When `False`, keep such records if
            the matched points are otherwise valid and only record the issue in
            the report.

    Returns:
        A tuple of `(samples, report)`, where `samples` is a flat collection of
        matched point pairs for both `top` and `bottom`, and `report` contains
        cleaning statistics suitable for logging or JSON serialization.
    """
    root = Path(data_root)
    split_map = read_split_file(root)

    if split_name not in split_map:
        raise ValueError(f"Unknown split: {split_name!r}. Expected one of {EXPECTED_SPLITS}.")

    report = TrainingDataReport(split_name=split_name, strict=strict)
    samples: list[TrainingSample] = []

    for session_rel_path in split_map[split_name]:
        report.sessions_seen += 1
        session_sample_count = 0
        session_path = build_session_path(root, session_rel_path)

        if not session_path.exists() or not session_path.is_dir():
            _increment_reason(report, None, "missing_session_directory")
            report.sessions_dropped += 1
            continue

        for source in VALID_SOURCES:
            raw_records = _load_annotation_records_for_training(
                report=report,
                session_path=session_path,
                source=source,
            )
            if raw_records is None:
                continue

            for raw_record in raw_records:
                report.records_seen += 1
                report.sources[source].records_seen += 1

                record_samples = _build_record_training_samples(
                    report=report,
                    split_name=split_name,
                    session_rel_path=session_rel_path,
                    session_path=session_path,
                    source=source,
                    raw_record=raw_record,
                    strict=strict,
                )
                if record_samples is None:
                    report.records_dropped += 1
                    report.sources[source].records_dropped += 1
                    continue

                report.records_used += 1
                report.sources[source].records_used += 1
                report.matched_samples += len(record_samples)
                report.sources[source].matched_samples += len(record_samples)
                session_sample_count += len(record_samples)
                samples.extend(record_samples)

        if session_sample_count == 0:
            report.sessions_dropped += 1

    return tuple(samples), report


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


def _load_annotation_records_for_training(
        report: TrainingDataReport,
        session_path: Path,
        source: SourceName,
) -> list[Any] | None:
    """Load one annotation file for lenient sample preparation."""
    annotation_path, used_typo = _try_resolve_annotation_path(session_path, source)
    if annotation_path is None:
        _increment_reason(report, source, "missing_annotation_file")
        return None

    if used_typo:
        _increment_reason(report, source, "annotation_filename_typo")

    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            raw_records = json.load(handle)
    except (OSError, json.JSONDecodeError):
        _increment_reason(report, source, "invalid_annotation_schema")
        return None

    if not isinstance(raw_records, list):
        _increment_reason(report, source, "invalid_annotation_schema")
        return None

    return raw_records


def _build_record_training_samples(
        report: TrainingDataReport,
        split_name: str,
        session_rel_path: str,
        session_path: Path,
        source: SourceName,
        raw_record: Any,
        strict: bool,
) -> tuple[TrainingSample, ...] | None:
    """Convert a raw annotation record into flat point-level training samples."""
    if not isinstance(raw_record, dict):
        _increment_reason(report, source, "invalid_record_schema")
        return None

    try:
        door2_image_raw = _require_non_empty_str(raw_record, "file1_path")
        source_image_raw = _require_non_empty_str(raw_record, "file2_path")
    except ValueError:
        _increment_reason(report, source, "invalid_record_schema")
        return None

    door2_points = _parse_points_for_training(
        report=report,
        source=source,
        raw_points=raw_record.get("image1_coordinates"),
        field_name="image1_coordinates",
    )
    source_points = _parse_points_for_training(
        report=report,
        source=source,
        raw_points=raw_record.get("image2_coordinates"),
        field_name="image2_coordinates",
    )
    if door2_points is None or source_points is None:
        return None

    if not door2_points or not source_points:
        _increment_reason(report, source, "empty_points")
        return None

    try:
        door2_image_path = resolve_session_image_path(
            session_path=session_path,
            raw_path=door2_image_raw,
            expected_camera_dir="door2",
        )
        source_image_path = resolve_session_image_path(
            session_path=session_path,
            raw_path=source_image_raw,
            expected_camera_dir=source,
        )
    except ValueError:
        _increment_reason(report, source, "invalid_record_schema")
        return None

    if not door2_image_path.is_file() or not source_image_path.is_file():
        _increment_reason(report, source, "missing_image_file")
        return None

    if not _points_in_range(door2_points) or not _points_in_range(source_points):
        _increment_reason(report, source, "point_out_of_range")
        return None

    if strict and (
            not _has_valid_point_count(door2_points) or not _has_valid_point_count(source_points)
    ):
        _increment_reason(report, source, "invalid_point_count")
        return None

    if not strict and (
            not _has_valid_point_count(door2_points) or not _has_valid_point_count(source_points)
    ):
        _increment_reason(report, source, "invalid_point_count")

    door2_points_by_number = _map_points_by_number(door2_points)
    source_points_by_number = _map_points_by_number(source_points)
    if door2_points_by_number is None or source_points_by_number is None:
        _increment_reason(report, source, "point_number_mismatch")
        return None

    if door2_points_by_number.keys() != source_points_by_number.keys():
        _increment_reason(report, source, "point_number_mismatch")
        return None

    samples = tuple(
        TrainingSample(
            split_name=split_name,
            session_rel_path=session_rel_path,
            source=source,
            point_number=point_number,
            door2_image_path=door2_image_path,
            source_image_path=source_image_path,
            x_src=source_points_by_number[point_number].x,
            y_src=source_points_by_number[point_number].y,
            x_door2=door2_points_by_number[point_number].x,
            y_door2=door2_points_by_number[point_number].y,
        )
        for point_number in sorted(door2_points_by_number)
    )

    if not samples:
        _increment_reason(report, source, "empty_points")
        return None

    return samples


def _parse_points_for_training(
        report: TrainingDataReport,
        source: SourceName,
        raw_points: Any,
        field_name: str,
) -> tuple[CoordinatePoint, ...] | None:
    """Parse points for sample preparation and convert schema issues into report entries."""
    try:
        return parse_points(raw_points, field_name)
    except ValueError:
        _increment_reason(report, source, "invalid_record_schema")
        return None


def _try_resolve_annotation_path(
        session_path: Path,
        source: SourceName,
) -> tuple[Path | None, bool]:
    """Return an existing annotation path and whether a typo fallback was used."""
    expected_name = COORDS_FILENAME_BY_SOURCE[source]
    for candidate_name in COORDS_FILENAME_CANDIDATES_BY_SOURCE[source]:
        candidate_path = session_path / candidate_name
        if candidate_path.exists():
            return candidate_path, candidate_name != expected_name
    return None, False


def _increment_reason(
        report: TrainingDataReport,
        source: SourceName | None,
        reason: str,
) -> None:
    """Increase one issue counter in the global and per-source reports."""
    report.reason_counts[reason] = report.reason_counts.get(reason, 0) + 1
    if source is None:
        return
    source_report = report.sources[source]
    source_report.reason_counts[reason] = source_report.reason_counts.get(reason, 0) + 1


def _map_points_by_number(
        points: tuple[CoordinatePoint, ...],
) -> dict[int, CoordinatePoint] | None:
    """Create a number->point mapping or return None on duplicates."""
    points_by_number: dict[int, CoordinatePoint] = {}
    for point in points:
        if point.number in points_by_number:
            return None
        points_by_number[point.number] = point
    return points_by_number


def _has_valid_point_count(points: tuple[CoordinatePoint, ...]) -> bool:
    """Check whether a frame pair stays within the task-defined point-count range."""
    return MIN_POINTS_PER_FRAME_PAIR <= len(points) <= MAX_POINTS_PER_FRAME_PAIR


def _points_in_range(points: tuple[CoordinatePoint, ...]) -> bool:
    """Check whether all points lie inside the expected image bounds."""
    return all(0.0 <= point.x <= IMAGE_WIDTH and 0.0 <= point.y <= IMAGE_HEIGHT for point in points)


def _require_non_empty_str(raw_record: dict[str, Any], field_name: str) -> str:
    """Fetch a non-empty string field from a raw JSON object."""
    value = raw_record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return value
