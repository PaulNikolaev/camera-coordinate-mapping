"""Training utilities for source-to-door2 coordinate mapping."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.ensemble import ExtraTreesRegressor

from solution.cli import add_artifacts_dir_argument, add_data_root_argument, add_strict_argument
from solution.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_ROOT,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODEL_FAMILY,
    VALID_SOURCES,
    SourceName,
)
from solution.data import TrainingDataReport, TrainingSample, build_training_samples
from solution.validation import ensure_dataset_present

DEFAULT_RANDOM_SEED = 42
DEFAULT_EXTRA_TREES_ESTIMATORS = 600
DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF = 2
DEFAULT_EXTRA_TREES_N_JOBS = 1
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
        description="Train ExtraTrees baselines for top->door2 and bottom->door2."
    )
    add_data_root_argument(parser)
    add_artifacts_dir_argument(
        parser,
        help_text="Directory where trained artifacts and reports will be saved.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_EXTRA_TREES_ESTIMATORS,
        help="Number of trees for both source models.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
        help="Minimum number of samples required in each leaf for both source models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed recorded in artifacts for reproducibility.",
    )
    add_strict_argument(parser)
    return parser


def train_and_save_models(
        data_root: Path = DEFAULT_DATA_ROOT,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        n_estimators: int = DEFAULT_EXTRA_TREES_ESTIMATORS,
        min_samples_leaf: int = DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
        seed: int = DEFAULT_RANDOM_SEED,
        strict: bool = False,
) -> TrainingRunResult:
    """Train top/bottom baseline models and persist their artifacts.

    Args:
        data_root: Dataset root with `split.json`, `train/`, and `val/`.
        artifacts_dir: Output directory for serialized models and JSON reports.
        n_estimators: Number of trees used by both ExtraTrees regressors.
        min_samples_leaf: Minimum number of samples per leaf node.
        seed: Random seed recorded in the artifact manifest.
        strict: Whether to use strict or allow_partial sample preparation.

    Returns:
        Paths and counts for the generated training artifacts.

    Raises:
        ValueError: If there are no usable samples or invalid training params.
    """
    _validate_training_params(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )

    data_root = Path(data_root)
    artifacts_root = Path(artifacts_dir)
    ensure_dataset_present(data_root)

    samples, report = build_training_samples(
        data_root=data_root,
        split_name="train",
        strict=strict,
    )
    if not samples:
        raise ValueError("No usable training samples were produced from the train split.")

    grouped_samples = _group_samples_by_source(samples)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    source_artifacts: dict[SourceName, TrainedSourceArtifact] = {}
    for source in VALID_SOURCES:
        source_samples = grouped_samples[source]
        if not source_samples:
            raise ValueError(f"No usable training samples were produced for source '{source}'.")

        model = train_source_model(
            source_samples=source_samples,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
        )
        model_path = artifacts_root / f"{source}_model.pkl"
        model_path.write_bytes(pickle.dumps(model))

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
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
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
        n_estimators: int,
        min_samples_leaf: int,
        seed: int,
) -> ExtraTreesRegressor:
    """Train one source-specific ExtraTrees regressor."""
    features = [_build_feature_row(sample.x_src, sample.y_src) for sample in source_samples]
    targets = [[sample.x_door2, sample.y_door2] for sample in source_samples]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=DEFAULT_EXTRA_TREES_N_JOBS,
    )
    model.fit(features, targets)
    return model


def build_artifact_manifest(
        report: TrainingDataReport,
        source_artifacts: dict[SourceName, TrainedSourceArtifact],
        n_estimators: int,
        min_samples_leaf: int,
        seed: int,
        strict: bool,
) -> dict[str, Any]:
    """Build a stable JSON manifest for downstream loading."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "train_split": report.split_name,
        "strict": strict,
        "seed": seed,
        "model_params": {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
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
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
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


def _build_feature_row(x_src: float, y_src: float) -> list[float]:
    """Build one normalized feature row from source-image coordinates."""
    return [x_src / IMAGE_WIDTH, y_src / IMAGE_HEIGHT]


def _validate_training_params(
        n_estimators: int,
        min_samples_leaf: int,
) -> None:
    """Validate baseline training hyperparameters."""
    if n_estimators < 1:
        raise ValueError(f"'n_estimators' must be >= 1, got {n_estimators}.")
    if min_samples_leaf < 1:
        raise ValueError(f"'min_samples_leaf' must be >= 1, got {min_samples_leaf}.")


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting."""
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
