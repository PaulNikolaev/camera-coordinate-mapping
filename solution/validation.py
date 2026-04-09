"""Quick checks and full validation for the coordinate-mapping dataset."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from solution.config import (
    COORDS_FILENAME_BY_SOURCE,
    DEFAULT_ARCHIVE_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET_URL,
    EXPECTED_SPLITS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_POINTS_PER_FRAME_PAIR,
    METADATA_DIR_NAMES,
    MIN_POINTS_PER_FRAME_PAIR,
    SPLIT_FILENAME,
)
from solution.data import (
    CoordinateDataset,
    FramePairAnnotation,
    build_session_path,
    load_dataset,
    read_split_file,
    resolve_annotation_path,
)


class DatasetValidationError(ValueError):
    """Raised when the dataset layout or annotations are invalid."""


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset archive cannot be downloaded or extracted."""


def ensure_dataset_present(data_root: Path = DEFAULT_DATA_ROOT) -> None:
    """Perform a fast existence check before expensive work starts."""
    root = Path(data_root)
    split_path = root / SPLIT_FILENAME

    if not root.exists() or not root.is_dir():
        raise DatasetValidationError(
            f"Dataset root not found: '{root}'. Expected unpacked 'coord_data/'."
        )
    if not split_path.exists():
        raise DatasetValidationError(
            f"Missing split file: '{split_path}'. Expected 'coord_data/{SPLIT_FILENAME}'."
        )

    try:
        split_map = read_split_file(root)
    except ValueError as error:
        raise DatasetValidationError(str(error)) from error

    for split_name in EXPECTED_SPLITS:
        split_dir = root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            raise DatasetValidationError(
                f"Missing split directory: '{split_dir}'. Expected both train/ and val/."
            )
        if not split_map[split_name]:
            raise DatasetValidationError(f"split.json field '{split_name}' must not be empty.")


def load_and_validate_dataset(data_root: Path = DEFAULT_DATA_ROOT) -> CoordinateDataset:
    """Load the dataset once and run the full validation pass."""
    root = Path(data_root)
    ensure_dataset_present(root)

    try:
        dataset = load_dataset(root)
    except ValueError as error:
        raise DatasetValidationError(str(error)) from error

    validate_dataset(dataset)
    return dataset


def prepare_and_validate_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    dataset_url: str = DEFAULT_DATASET_URL,
    archive_path: Path | None = None,
    keep_archive: bool = False,
    force_download: bool = False,
) -> CoordinateDataset:
    """Download, normalize, and validate the dataset in one flow."""
    root = Path(data_root)
    archive_file = Path(archive_path) if archive_path is not None else root.parent / DEFAULT_ARCHIVE_NAME

    if force_download:
        remove_dataset_root(root)

    if root.exists() and not force_download:
        return load_and_validate_dataset(root)

    existing_root = discover_dataset_root(root.parent)
    if existing_root is not None and existing_root != root and not force_download:
        normalize_dataset_root(existing_root, root)
        return load_and_validate_dataset(root)

    downloaded_archive = False
    if force_download or not archive_file.exists():
        download_dataset_archive(dataset_url=dataset_url, archive_path=archive_file)
        downloaded_archive = True

    extract_dataset_archive(
        archive_path=archive_file,
        output_dir=root.parent,
        target_root=root,
    )
    dataset = load_and_validate_dataset(root)

    if downloaded_archive and not keep_archive and archive_file.exists():
        archive_file.unlink()

    return dataset


def download_dataset_archive(dataset_url: str, archive_path: Path) -> Path:
    """Download the dataset archive from a shared Google Drive link."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    direct_url = build_google_drive_download_url(dataset_url)

    try:
        with urlopen(direct_url) as response, archive_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    except Exception as error:  # noqa: BLE001
        raise DatasetDownloadError(
            f"Failed to download dataset archive from '{dataset_url}': {error}"
        ) from error

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise DatasetDownloadError(f"Downloaded archive is empty: '{archive_path}'.")

    return archive_path


def extract_dataset_archive(
    archive_path: Path,
    output_dir: Path,
    target_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Extract the archive and normalize the result into `coord_data/`."""
    if not archive_path.exists():
        raise DatasetDownloadError(f"Archive file does not exist: '{archive_path}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    remove_dataset_root(Path(target_root))

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(temp_dir)
        except zipfile.BadZipFile as error:
            raise DatasetDownloadError(
                f"Archive is not a valid ZIP file: '{archive_path}'."
            ) from error

        extracted_root = discover_dataset_root(temp_dir)
        if extracted_root is None:
            raise DatasetDownloadError(
                "Archive was extracted, but no dataset root with split.json/train/val was found."
            )

        normalize_dataset_root(extracted_root, Path(target_root))

    return Path(target_root)


def discover_dataset_root(search_root: Path) -> Path | None:
    """Find a directory that already looks like a valid dataset root."""
    candidates: list[Path] = []

    if looks_like_dataset_root(search_root):
        candidates.append(search_root)

    for child in search_root.iterdir():
        if child.name in METADATA_DIR_NAMES or not child.is_dir():
            continue
        if looks_like_dataset_root(child):
            candidates.append(child)

    if not candidates:
        return None
    if len(candidates) > 1:
        candidate_paths = ", ".join(str(path) for path in candidates)
        raise DatasetDownloadError(
            f"Found multiple dataset root candidates after extraction: {candidate_paths}."
        )

    return candidates[0]


def looks_like_dataset_root(path: Path) -> bool:
    """Check the minimum layout expected from the unpacked dataset root."""
    return (
        path.exists()
        and path.is_dir()
        and (path / SPLIT_FILENAME).is_file()
        and all((path / split_name).is_dir() for split_name in EXPECTED_SPLITS)
    )


def normalize_dataset_root(source_root: Path, target_root: Path) -> Path:
    """Move an extracted dataset root into the canonical `coord_data/` location."""
    if source_root == target_root:
        return target_root

    if target_root.exists():
        raise DatasetDownloadError(
            f"Cannot move dataset into '{target_root}': target path already exists."
        )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_root), str(target_root))
    return target_root


def remove_dataset_root(dataset_root: Path) -> None:
    """Delete the canonical dataset root if it already exists."""
    if dataset_root.exists():
        shutil.rmtree(dataset_root)


def build_google_drive_download_url(shared_url: str) -> str:
    """Convert a Google Drive sharing URL into a direct-download URL."""
    parsed = urlparse(shared_url)
    query_id = parse_qs(parsed.query).get("id", [None])[0]
    file_id = query_id or _extract_drive_file_id(parsed.path)

    if not file_id:
        raise DatasetDownloadError(
            f"Could not extract Google Drive file id from URL: '{shared_url}'."
        )

    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def validate_dataset(dataset: CoordinateDataset) -> None:
    """Validate all split/session/record constraints required by the task."""
    for split_name in EXPECTED_SPLITS:
        split = dataset.get_split(split_name)
        seen_sessions: set[str] = set()

        for session_rel_path, session in zip(split.session_paths, split.sessions, strict=True):
            if session_rel_path in seen_sessions:
                raise DatasetValidationError(
                    f"Duplicate session path in split '{split_name}': '{session_rel_path}'."
                )
            seen_sessions.add(session_rel_path)

            session_path = build_session_path(dataset.data_root, session_rel_path)
            if not session_path.exists() or not session_path.is_dir():
                raise DatasetValidationError(
                    f"Session directory listed in split '{split_name}' does not exist: '{session_path}'."
                )

            for source, frame_pairs in (("top", session.top_pairs), ("bottom", session.bottom_pairs)):
                try:
                    annotation_path = resolve_annotation_path(session_path, source)
                except ValueError as error:
                    raise DatasetValidationError(str(error)) from error
                if not frame_pairs:
                    raise DatasetValidationError(f"No frame pairs found in '{annotation_path}'.")

                for frame_pair in frame_pairs:
                    validate_frame_pair(frame_pair)


def validate_frame_pair(frame_pair: FramePairAnnotation) -> None:
    """Validate one frame-pair record and all labeled points inside it."""
    if not frame_pair.door2_image_path.exists():
        raise DatasetValidationError(
            f"Missing door2 image file: '{frame_pair.door2_image_path}'."
        )
    if not frame_pair.source_image_path.exists():
        raise DatasetValidationError(
            f"Missing {frame_pair.source} image file: '{frame_pair.source_image_path}'."
        )

    door2_numbers = {point.number for point in frame_pair.door2_points}
    source_numbers = {point.number for point in frame_pair.source_points}

    if len(door2_numbers) != len(frame_pair.door2_points):
        raise DatasetValidationError(
            f"Duplicate point numbers found in door2 annotations for '{frame_pair.door2_image_path}'."
        )
    if len(source_numbers) != len(frame_pair.source_points):
        raise DatasetValidationError(
            f"Duplicate point numbers found in {frame_pair.source} annotations for "
            f"'{frame_pair.source_image_path}'."
        )
    if door2_numbers != source_numbers:
        raise DatasetValidationError(
            "Point-number mismatch between image1_coordinates and image2_coordinates for "
            f"'{frame_pair.source_image_path}'."
        )

    validate_point_count(len(frame_pair.door2_points), "image1_coordinates", frame_pair)
    validate_point_count(len(frame_pair.source_points), "image2_coordinates", frame_pair)

    if len(frame_pair.door2_points) != len(frame_pair.source_points):
        raise DatasetValidationError(
            f"Point count mismatch for '{frame_pair.source_image_path}'."
        )

    for point in frame_pair.door2_points:
        validate_point_range(point.x, point.y, "door2", frame_pair.door2_image_path)
    for point in frame_pair.source_points:
        validate_point_range(point.x, point.y, frame_pair.source, frame_pair.source_image_path)


def validate_point_count(
    point_count: int,
    field_name: str,
    frame_pair: FramePairAnnotation,
) -> None:
    """Ensure each frame pair contains the task-defined variable point count."""
    if MIN_POINTS_PER_FRAME_PAIR <= point_count <= MAX_POINTS_PER_FRAME_PAIR:
        return

    raise DatasetValidationError(
        f"Unexpected number of labeled points in '{field_name}' for "
        f"'{frame_pair.source_image_path}': {point_count}. "
        f"Expected {MIN_POINTS_PER_FRAME_PAIR}-{MAX_POINTS_PER_FRAME_PAIR}."
    )


def validate_point_range(x: float, y: float, camera_name: str, image_path: Path) -> None:
    """Ensure point coordinates stay inside the expected image bounds."""
    if not 0.0 <= x <= IMAGE_WIDTH:
        raise DatasetValidationError(
            f"Point x-coordinate out of range for {camera_name} image '{image_path}': {x}."
        )
    if not 0.0 <= y <= IMAGE_HEIGHT:
        raise DatasetValidationError(
            f"Point y-coordinate out of range for {camera_name} image '{image_path}': {y}."
        )


def _extract_drive_file_id(path: str) -> str | None:
    """Extract the file id from `/file/d/<id>/...` style Google Drive URLs."""
    parts = [part for part in path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "d" and index + 1 < len(parts):
            return parts[index + 1]
    return None
