"""End-to-end CLI flow for training and validation-split evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from solution.cli import (
    add_artifacts_dir_argument,
    add_data_root_argument,
    add_output_metrics_argument,
    add_strict_argument,
)
from solution.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_ROOT
from solution.evaluate import EvaluationRunResult, evaluate_and_save_metrics
from solution.train import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_EXTRA_TREES_ESTIMATORS,
    DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
    TrainingRunResult,
    train_and_save_models,
)


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Information about one completed train-and-evaluate run."""

    training: TrainingRunResult
    evaluation: EvaluationRunResult


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the one-command baseline pipeline."""
    parser = argparse.ArgumentParser(
        description="Train baseline models and evaluate MED on the validation split."
    )
    add_data_root_argument(parser)
    add_artifacts_dir_argument(
        parser,
        help_text="Directory where artifacts, reports, and metrics will be saved.",
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
    add_output_metrics_argument(parser)
    add_strict_argument(parser)
    return parser


def run_pipeline(
        data_root: Path = DEFAULT_DATA_ROOT,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        n_estimators: int = DEFAULT_EXTRA_TREES_ESTIMATORS,
        min_samples_leaf: int = DEFAULT_EXTRA_TREES_MIN_SAMPLES_LEAF,
        seed: int = DEFAULT_RANDOM_SEED,
        output_metrics: Path | None = None,
        strict: bool = False,
) -> PipelineRunResult:
    """Train artifacts and immediately evaluate them on the validation split."""
    data_root = Path(data_root)
    artifacts_dir = Path(artifacts_dir)

    training_result = train_and_save_models(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
        strict=strict,
    )
    evaluation_result = evaluate_and_save_metrics(
        data_root=data_root,
        artifacts_dir=training_result.artifacts_dir,
        output_metrics=output_metrics,
        strict=strict,
    )
    return PipelineRunResult(training=training_result, evaluation=evaluation_result)


def main(argv: list[str] | None = None) -> int:
    """Run the full baseline pipeline from CLI arguments."""
    args = build_parser().parse_args(argv)
    result = run_pipeline(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
        output_metrics=args.output_metrics,
        strict=args.strict,
    )
    print(
        "Pipeline completed. "
        f"Saved artifacts to '{result.training.artifacts_dir}' and metrics to "
        f"'{result.evaluation.metrics_path}'."
    )
    return 0
