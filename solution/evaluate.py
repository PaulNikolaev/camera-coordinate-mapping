"""Evaluation helpers for source-to-door2 coordinate mapping."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solution.cli import (
    add_artifacts_dir_argument,
    add_data_root_argument,
    add_output_metrics_argument,
    add_strict_argument,
)
from solution.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_ROOT,
    VALID_SOURCES,
    SourceName,
)
from solution.data import TrainingDataReport, build_training_samples
from solution.inference import load_artifacts
from solution.validation import ensure_dataset_present

DEFAULT_METRICS_FILENAME = "metrics.json"
METRICS_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class SourceEvaluationMetrics:
    """MED summary for one source camera."""

    source: SourceName
    med: float
    point_count: int
    record_count: int


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Information about one completed evaluation run."""

    artifacts_dir: Path
    metrics_path: Path
    source_metrics: dict[SourceName, SourceEvaluationMetrics]
    overall_med: float
    point_count: int
    record_count: int
    metrics: dict[str, Any]


@dataclass(slots=True)
class _DistanceAccumulator:
    """Mutable helper for MED aggregation."""

    total_distance: float = 0.0
    point_count: int = 0

    def add(self, distance: float) -> None:
        """Accumulate one Euclidean distance value."""

        self.total_distance += distance
        self.point_count += 1

    def mean(self) -> float:
        """Return the mean distance over all accumulated points."""

        if self.point_count == 0:
            raise ValueError("Cannot calculate MED without at least one matched point.")
        return self.total_distance / self.point_count


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for validation-split evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate MED on the validation split for top->door2 and bottom->door2."
    )
    add_data_root_argument(parser)
    add_artifacts_dir_argument(
        parser,
        help_text="Directory containing trained artifacts and the default metrics output.",
    )
    add_output_metrics_argument(parser)
    add_strict_argument(parser)
    return parser


def evaluate_and_save_metrics(
        data_root: Path = DEFAULT_DATA_ROOT,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        output_metrics: Path | None = None,
        strict: bool = False,
) -> EvaluationRunResult:
    """Evaluate saved models on the validation split and persist MED metrics.

    Args:
        data_root: Dataset root with `split.json`, `train/`, and `val/`.
        artifacts_dir: Directory containing `manifest.json` and source models.
        output_metrics: Explicit output path for `metrics.json`.
        strict: Whether to use strict or allow_partial sample preparation.

    Returns:
        A summary of the completed evaluation run and the persisted metrics.

    Raises:
        ValueError: If no usable validation samples are available.
        ArtifactLoadError: If inference artifacts cannot be loaded.
    """

    data_root = Path(data_root)
    artifacts_dir = Path(artifacts_dir)

    ensure_dataset_present(data_root)
    samples, report = build_training_samples(
        data_root=data_root,
        split_name="val",
        strict=strict,
    )
    if not samples:
        raise ValueError("No usable evaluation samples were produced from the val split.")

    predictor = load_artifacts(artifacts_dir)

    per_source_accumulators = {
        source: _DistanceAccumulator() for source in VALID_SOURCES
    }
    overall_accumulator = _DistanceAccumulator()

    for source in VALID_SOURCES:
        source_samples = tuple(sample for sample in samples if sample.source == source)
        if not source_samples:
            continue

        predictions = predictor.predict_batch(
            points=tuple((sample.x_src, sample.y_src) for sample in source_samples),
            source=source,
        )
        if len(predictions) != len(source_samples):
            raise ValueError(
                "Model returned a different number of predictions than evaluation samples."
            )

        accumulator = per_source_accumulators[source]
        for sample, prediction in zip(source_samples, predictions):
            predicted_x, predicted_y = prediction
            distance = math.hypot(
                predicted_x - sample.x_door2,
                predicted_y - sample.y_door2,
            )
            accumulator.add(distance)
            overall_accumulator.add(distance)

    source_metrics: dict[SourceName, SourceEvaluationMetrics] = {}
    for source in VALID_SOURCES:
        accumulator = per_source_accumulators[source]
        if accumulator.point_count == 0:
            raise ValueError(f"No usable evaluation samples were produced for source '{source}'.")

        source_metrics[source] = SourceEvaluationMetrics(
            source=source,
            med=accumulator.mean(),
            point_count=accumulator.point_count,
            record_count=report.sources[source].records_used,
        )

    metrics_path = _resolve_metrics_path(
        artifacts_dir=artifacts_dir,
        output_metrics=output_metrics,
    )
    metrics_payload = build_metrics_payload(
        report=report,
        predictor_manifest=predictor.manifest,
        source_metrics=source_metrics,
        overall_med=overall_accumulator.mean(),
    )
    _write_json(metrics_path, metrics_payload)

    return EvaluationRunResult(
        artifacts_dir=artifacts_dir,
        metrics_path=metrics_path,
        source_metrics=source_metrics,
        overall_med=metrics_payload["overall"]["med"],
        point_count=report.matched_samples,
        record_count=report.records_used,
        metrics=metrics_payload,
    )


def build_metrics_payload(
        report: TrainingDataReport,
        predictor_manifest: dict[str, Any],
        source_metrics: dict[SourceName, SourceEvaluationMetrics],
        overall_med: float,
) -> dict[str, Any]:
    """Build a stable JSON payload for validation metrics."""

    sessions_used = report.sessions_seen - report.sessions_dropped
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "metric_name": "MED",
        "distance_unit": "pixels",
        "split": report.split_name,
        "strict": report.strict,
        "model_family": predictor_manifest.get("model_family"),
        "artifacts_schema_version": predictor_manifest.get("schema_version"),
        "overall": {
            "med": overall_med,
            "point_count": report.matched_samples,
            "record_count": report.records_used,
            "session_count": sessions_used,
        },
        "sources": {
            source: {
                "med": source_metrics[source].med,
                "point_count": source_metrics[source].point_count,
                "record_count": source_metrics[source].record_count,
            }
            for source in VALID_SOURCES
        },
        "preparation_report": report.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run validation-split evaluation from CLI arguments."""

    args = build_parser().parse_args(argv)
    result = evaluate_and_save_metrics(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        output_metrics=args.output_metrics,
        strict=args.strict,
    )
    print(
        "Evaluation completed. "
        f"Saved MED metrics for {result.point_count} points to '{result.metrics_path}'."
    )
    return 0


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_metrics_path(
        artifacts_dir: Path,
        output_metrics: Path | None,
) -> Path:
    """Resolve where evaluation metrics should be saved."""
    if output_metrics is not None:
        return Path(output_metrics)
    return artifacts_dir / DEFAULT_METRICS_FILENAME
