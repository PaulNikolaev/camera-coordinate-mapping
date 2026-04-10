"""Inference helpers for source-to-door2 coordinate mapping."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

from solution.config import IMAGE_HEIGHT, IMAGE_WIDTH, VALID_SOURCES, SourceName

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
ARTIFACT_SCHEMA_VERSION = "1"


class ArtifactLoadError(RuntimeError):
    """Raised when trained artifacts are missing or malformed."""


@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    """Loaded predictor state for both supported source cameras.

    Args:
        artifacts_dir: Directory containing `manifest.json` and model files.
        manifest_path: Path to the validated manifest file.
        manifest: Parsed manifest payload.
        models: Loaded models keyed by source name.
    """

    artifacts_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    models: dict[SourceName, Any]

    def predict(self, x: float, y: float, source: str) -> tuple[float, float]:
        """Map source-camera coordinates into the door2 frame.

        The input `(x, y)` must be numeric pixel coordinates in the source image
        space with bounds `[0, 3200] x [0, 1800]`. The `source` must be either
        `top` or `bottom`. The returned coordinates are always clipped into the
        door2 image frame `[0, 3200] x [0, 1800]`.

        Args:
            x: Source-image x coordinate in pixels.
            y: Source-image y coordinate in pixels.
            source: Source camera name, `top` or `bottom`.

        Returns:
            A `(x_door2, y_door2)` tuple in door2 pixel coordinates.

        Raises:
            TypeError: If `x` or `y` is not a real number.
            ValueError: If coordinates are outside the source frame or
                `source` is invalid.
            ArtifactLoadError: If the loaded model returns malformed output.
        """

        validated_source = _validate_source(source)
        x_value = _validate_coordinate(name="x", value=x, upper_bound=IMAGE_WIDTH)
        y_value = _validate_coordinate(name="y", value=y, upper_bound=IMAGE_HEIGHT)

        features = [[x_value / IMAGE_WIDTH, y_value / IMAGE_HEIGHT]]
        raw_prediction = self.models[validated_source].predict(features)
        return _normalize_prediction_output(raw_prediction)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one-off coordinate prediction."""
    parser = argparse.ArgumentParser(
        description="Predict door2 coordinates from top/bottom source coordinates."
    )
    parser.add_argument("x", type=float, help="Source-image x coordinate in pixels.")
    parser.add_argument("y", type=float, help="Source-image y coordinate in pixels.")
    parser.add_argument(
        "source",
        help="Source camera name. Supported values: top, bottom.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing manifest.json and saved source models.",
    )
    return parser


def load_artifacts(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> LoadedArtifacts:
    """Load and validate serialized models for `top` and `bottom`.

    Args:
        artifacts_dir: Directory containing `manifest.json` and model pickle files.

    Returns:
        Loaded models and the validated manifest payload.

    Raises:
        ArtifactLoadError: If the manifest or model files are missing or invalid.
    """

    artifacts_root = Path(artifacts_dir)
    manifest_path = artifacts_root / "manifest.json"
    manifest = _read_manifest(manifest_path)
    source_entries = _validate_manifest(manifest)

    models: dict[SourceName, Any] = {}
    for source in VALID_SOURCES:
        model_path_value = source_entries[source].get("model_path")
        if not isinstance(model_path_value, str) or not model_path_value.strip():
            raise ArtifactLoadError(
                f"Manifest entry for source '{source}' must define a non-empty 'model_path'."
            )

        model_path = artifacts_root / model_path_value
        if not model_path.is_file():
            raise ArtifactLoadError(
                f"Model file for source '{source}' was not found: '{model_path}'."
            )

        try:
            with model_path.open("rb") as handle:
                model = pickle.load(handle)
        except OSError as error:
            raise ArtifactLoadError(
                f"Failed to read model file for source '{source}': {error}."
            ) from error
        except (pickle.PickleError, AttributeError, EOFError, ImportError, ModuleNotFoundError) as error:
            raise ArtifactLoadError(
                f"Failed to deserialize model file for source '{source}': {error}."
            ) from error

        if not hasattr(model, "predict"):
            raise ArtifactLoadError(
                f"Model loaded for source '{source}' does not provide a 'predict' method."
            )

        models[source] = model

    return LoadedArtifacts(
        artifacts_dir=artifacts_root,
        manifest_path=manifest_path,
        manifest=manifest,
        models=models,
    )


def predict(x: float, y: float, source: str) -> tuple[float, float]:
    """Predict door2 coordinates using artifacts from the default directory.

    This public helper keeps the task contract stable as `predict(x, y, source)`.
    For custom artifact directories, load an explicit predictor with
    `load_artifacts(...)` and call its `.predict(...)` method.
    """

    return _load_default_artifacts().predict(x=x, y=y, source=source)


def main(argv: list[str] | None = None) -> int:
    """Run one prediction from CLI arguments."""
    args = build_parser().parse_args(argv)
    predictor = load_artifacts(args.artifacts_dir)
    x_door2, y_door2 = predictor.predict(x=args.x, y=args.y, source=args.source)
    print(
        json.dumps(
            {
                "input_source": args.source,
                "input_x": args.x,
                "input_y": args.y,
                "x_door2": x_door2,
                "y_door2": y_door2,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


@lru_cache(maxsize=1)
def _load_default_artifacts() -> LoadedArtifacts:
    """Load default artifacts once for the public `predict(...)` helper."""

    return load_artifacts(DEFAULT_ARTIFACTS_DIR)


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read one manifest JSON file from disk."""

    if not manifest_path.is_file():
        raise ArtifactLoadError(f"Artifact manifest was not found: '{manifest_path}'.")

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as error:
        raise ArtifactLoadError(f"Failed to read artifact manifest: {error}.") from error
    except json.JSONDecodeError as error:
        raise ArtifactLoadError(f"Artifact manifest is not valid JSON: {error}.") from error

    if not isinstance(payload, dict):
        raise ArtifactLoadError("Artifact manifest must contain a JSON object.")

    return payload


def _validate_manifest(manifest: dict[str, Any]) -> dict[SourceName, dict[str, Any]]:
    """Validate a manifest payload and return the per-source section."""

    schema_version = manifest.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactLoadError(
            "Unsupported artifact schema version. "
            f"Expected '{ARTIFACT_SCHEMA_VERSION}', got {schema_version!r}."
        )

    model_family = manifest.get("model_family")
    if model_family != "polynomial_ridge":
        raise ArtifactLoadError(
            f"Unsupported model family {model_family!r}. Expected 'polynomial_ridge'."
        )

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ArtifactLoadError("Artifact manifest must contain a 'sources' object.")

    validated_sources: dict[SourceName, dict[str, Any]] = {}
    for source in VALID_SOURCES:
        source_entry = sources.get(source)
        if not isinstance(source_entry, dict):
            raise ArtifactLoadError(
                f"Artifact manifest is missing a valid source entry for '{source}'."
            )
        validated_sources[source] = source_entry

    return validated_sources


def _validate_source(source: str) -> SourceName:
    """Validate and normalize the source-camera name."""

    if not isinstance(source, str):
        raise TypeError(f"'source' must be a string, got {type(source).__name__}.")
    if source not in VALID_SOURCES:
        allowed_sources = ", ".join(VALID_SOURCES)
        raise ValueError(f"Unsupported source {source!r}. Expected one of: {allowed_sources}.")
    return source


def _validate_coordinate(name: str, value: float, upper_bound: int) -> float:
    """Validate one numeric input coordinate within the image frame."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{name}' must be a real number, got {type(value).__name__}.")

    numeric_value = float(value)
    if not 0.0 <= numeric_value <= float(upper_bound):
        raise ValueError(
            f"'{name}' must be within [0, {upper_bound}], got {numeric_value}."
        )

    return numeric_value


def _normalize_prediction_output(raw_prediction: Any) -> tuple[float, float]:
    """Validate the model output shape and clip it into the door2 frame."""

    try:
        first_row = raw_prediction[0]
        raw_x = float(first_row[0])
        raw_y = float(first_row[1])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ArtifactLoadError(
            "Loaded model returned an invalid prediction format; expected [[x, y]]."
        ) from error

    clipped_x = min(max(raw_x, 0.0), float(IMAGE_WIDTH))
    clipped_y = min(max(raw_y, 0.0), float(IMAGE_HEIGHT))
    return clipped_x, clipped_y
