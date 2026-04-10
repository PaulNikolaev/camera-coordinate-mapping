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
    DEFAULT_POLYNOMIAL_DEGREE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RIDGE_ALPHA,
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
    add_output_metrics_argument(parser)
    add_strict_argument(parser)
    return parser


def run_pipeline(
        data_root: Path = DEFAULT_DATA_ROOT,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        alpha: float = DEFAULT_RIDGE_ALPHA,
        seed: int = DEFAULT_RANDOM_SEED,
        output_metrics: Path | None = None,
        strict: bool = False,
) -> PipelineRunResult:
    """Train artifacts and immediately evaluate them on the validation split."""
    training_result = train_and_save_models(
        data_root=Path(data_root),
        artifacts_dir=Path(artifacts_dir),
        degree=degree,
        alpha=alpha,
        seed=seed,
        strict=strict,
    )
    evaluation_result = evaluate_and_save_metrics(
        data_root=Path(data_root),
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
        degree=args.degree,
        alpha=args.alpha,
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
