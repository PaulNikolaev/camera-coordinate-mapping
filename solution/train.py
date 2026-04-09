"""Training utilities for source-to-door2 coordinate mapping."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from solution.config import (
    DEFAULT_DATA_ROOT,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VALID_SOURCES,
    SourceName,
)
from solution.data import TrainingDataReport, TrainingSample, build_training_samples

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_RANDOM_SEED = 42
ARTIFACT_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class TrainedSourceArtifact:
    """Saved artifact metadata for one trained source model."""

    source: SourceName
    model_path: Path
    sample_count: int
    record_count: int


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Information about one completed training run."""

    artifacts_dir: Path
    manifest_path: Path
    training_report_path: Path
    source_artifacts: dict[SourceName, TrainedSourceArtifact]
    sample_count: int


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for baseline model training."""
    parser = argparse.ArgumentParser(
        description="Train polynomial Ridge baselines for top->door2 and bottom->door2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the unpacked coord_data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory where trained artifacts and reports will be saved.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=DEFAULT_POLYNOMIAL_DEGREE,
        help="PolynomialFeatures degree for both source models.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Ridge regularization strength for both source models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed recorded in artifacts for reproducibility.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict point-count filtering instead of allow_partial preparation.",
    )
    return parser


def train_and_save_models(
        data_root: Path = DEFAULT_DATA_ROOT,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        alpha: float = DEFAULT_RIDGE_ALPHA,
        seed: int = DEFAULT_RANDOM_SEED,
        strict: bool = False,
) -> TrainingRunResult:
    """Train top/bottom baseline models and persist their artifacts.

    Args:
        data_root: Dataset root with `split.json`, `train/`, and `val/`.
        artifacts_dir: Output directory for serialized models and JSON reports.
        degree: Polynomial feature degree used by both models.
        alpha: Ridge regularization strength.
        seed: Random seed recorded in the artifact manifest.
        strict: Whether to use strict or allow_partial sample preparation.

    Returns:
        Paths and counts for the generated training artifacts.

    Raises:
        ValueError: If there are no usable samples or invalid training params.
    """
    if degree < 1:
        raise ValueError(f"'degree' must be >= 1, got {degree}.")
    if alpha < 0.0:
        raise ValueError(f"'alpha' must be >= 0, got {alpha}.")

    random.seed(seed)

    samples, report = build_training_samples(
        data_root=Path(data_root),
        split_name="train",
        strict=strict,
    )
    if not samples:
        raise ValueError("No usable training samples were produced from the train split.")

    grouped_samples = _group_samples_by_source(samples)
    artifacts_root = Path(artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    source_artifacts: dict[SourceName, TrainedSourceArtifact] = {}
    for source in VALID_SOURCES:
        source_samples = grouped_samples[source]
        if not source_samples:
            raise ValueError(f"No usable training samples were produced for source '{source}'.")

        model = train_source_model(source_samples=source_samples, degree=degree, alpha=alpha)
        model_path = artifacts_root / f"{source}_model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)

        source_artifacts[source] = TrainedSourceArtifact(
            source=source,
            model_path=model_path,
            sample_count=len(source_samples),
            record_count=report.sources[source].records_used,
        )

    training_report_path = artifacts_root / "training_report.json"
    _write_json(training_report_path, report.to_dict())

    manifest_path = artifacts_root / "manifest.json"
    _write_json(
        manifest_path,
        build_artifact_manifest(
            report=report,
            source_artifacts=source_artifacts,
            degree=degree,
            alpha=alpha,
            seed=seed,
            strict=strict,
        ),
    )

    return TrainingRunResult(
        artifacts_dir=artifacts_root,
        manifest_path=manifest_path,
        training_report_path=training_report_path,
        source_artifacts=source_artifacts,
        sample_count=len(samples),
    )


def train_source_model(
        source_samples: tuple[TrainingSample, ...],
        degree: int,
        alpha: float,
) -> Pipeline:
    """Train one source-specific polynomial Ridge model."""
    features = [
        [sample.x_src / IMAGE_WIDTH, sample.y_src / IMAGE_HEIGHT]
        for sample in source_samples
    ]
    targets = [[sample.x_door2, sample.y_door2] for sample in source_samples]

    model = Pipeline(
        steps=[
            (
                "polynomial_features",
                PolynomialFeatures(degree=degree, include_bias=False),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(features, targets)
    return model


def build_artifact_manifest(
        report: TrainingDataReport,
        source_artifacts: dict[SourceName, TrainedSourceArtifact],
        degree: int,
        alpha: float,
        seed: int,
        strict: bool,
) -> dict[str, Any]:
    """Build a stable JSON manifest for downstream loading."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": "polynomial_ridge",
        "train_split": report.split_name,
        "strict": strict,
        "seed": seed,
        "model_params": {
            "degree": degree,
            "alpha": alpha,
        },
        "input_normalization": {
            "x_scale": IMAGE_WIDTH,
            "y_scale": IMAGE_HEIGHT,
            "range": [0.0, 1.0],
        },
        "output_coordinates": {
            "space": "door2",
            "x_scale": IMAGE_WIDTH,
            "y_scale": IMAGE_HEIGHT,
        },
        "sources": {
            source: {
                "model_path": artifact.model_path.name,
                "sample_count": artifact.sample_count,
                "record_count": artifact.record_count,
            }
            for source, artifact in source_artifacts.items()
        },
        "preparation_report_path": "training_report.json",
    }


def main(argv: list[str] | None = None) -> int:
    """Run baseline model training from CLI arguments."""
    args = build_parser().parse_args(argv)
    result = train_and_save_models(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        degree=args.degree,
        alpha=args.alpha,
        seed=args.seed,
        strict=args.strict,
    )
    print(
        "Training completed. "
        f"Saved {len(result.source_artifacts)} models and {result.sample_count} samples to "
        f"'{result.artifacts_dir}'."
    )
    return 0


def _group_samples_by_source(
        samples: tuple[TrainingSample, ...],
) -> dict[SourceName, tuple[TrainingSample, ...]]:
    """Group flat samples by source name."""
    grouped: dict[SourceName, list[TrainingSample]] = {
        "top": [],
        "bottom": [],
    }
    for sample in samples:
        grouped[sample.source].append(sample)

    return {source: tuple(grouped[source]) for source in VALID_SOURCES}


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting."""
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
