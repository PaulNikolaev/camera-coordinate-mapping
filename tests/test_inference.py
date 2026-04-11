"""Smoke tests for artifact loading and prediction contract."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from solution.inference import ArtifactLoadError, load_artifacts


class DummyModel:
    """Simple pickle-friendly model stub for inference tests."""

    def __init__(self, x_multiplier: float, y_multiplier: float) -> None:
        self.x_multiplier = x_multiplier
        self.y_multiplier = y_multiplier

    def predict(self, features: list[list[float]]) -> list[list[float]]:
        return [
            [x_value * self.x_multiplier, y_value * self.y_multiplier]
            for x_value, y_value in features
        ]


class FixedOutputModel:
    """Model stub that always returns the same coordinates."""

    def __init__(self, x_value: float, y_value: float) -> None:
        self.x_value = x_value
        self.y_value = y_value

    def predict(self, features: list[list[float]]) -> list[list[float]]:
        _ = features
        return [[self.x_value, self.y_value]]


class LoadArtifactsTests(unittest.TestCase):
    """Validate artifact loading and prediction behavior."""

    def test_load_and_predict_for_supported_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            self._write_artifact_bundle(
                artifacts_dir=artifacts_dir,
                top_model=DummyModel(x_multiplier=3200.0, y_multiplier=1800.0),
                bottom_model=DummyModel(x_multiplier=6400.0, y_multiplier=3600.0),
            )

            predictor = load_artifacts(artifacts_dir)

            self.assertEqual(predictor.predict(1600.0, 900.0, "top"), (1600.0, 900.0))
            self.assertEqual(predictor.predict(800.0, 450.0, "bottom"), (1600.0, 900.0))

    def test_prediction_is_clipped_to_door2_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            self._write_artifact_bundle(
                artifacts_dir=artifacts_dir,
                top_model=FixedOutputModel(x_value=-25.0, y_value=9999.0),
                bottom_model=FixedOutputModel(x_value=100.0, y_value=200.0),
            )

            predictor = load_artifacts(artifacts_dir)

            self.assertEqual(predictor.predict(10.0, 10.0, "top"), (0.0, 1800.0))

    def test_predict_batch_returns_all_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            self._write_artifact_bundle(
                artifacts_dir=artifacts_dir,
                top_model=DummyModel(x_multiplier=3200.0, y_multiplier=1800.0),
                bottom_model=DummyModel(x_multiplier=3200.0, y_multiplier=1800.0),
            )

            predictor = load_artifacts(artifacts_dir)

            self.assertEqual(
                predictor.predict_batch(
                    points=((100.0, 200.0), (1600.0, 900.0)),
                    source="top",
                ),
                ((100.0, 200.0), (1600.0, 900.0)),
            )

    def test_invalid_source_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            self._write_artifact_bundle(
                artifacts_dir=artifacts_dir,
                top_model=FixedOutputModel(x_value=1.0, y_value=2.0),
                bottom_model=FixedOutputModel(x_value=3.0, y_value=4.0),
            )

            predictor = load_artifacts(artifacts_dir)

            with self.assertRaisesRegex(ValueError, "Unsupported source"):
                predictor.predict(100.0, 200.0, "left")

    def test_invalid_coordinate_type_raises_type_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            self._write_artifact_bundle(
                artifacts_dir=artifacts_dir,
                top_model=FixedOutputModel(x_value=1.0, y_value=2.0),
                bottom_model=FixedOutputModel(x_value=3.0, y_value=4.0),
            )

            predictor = load_artifacts(artifacts_dir)

            with self.assertRaisesRegex(TypeError, "'x' must be a real number"):
                predictor.predict("100", 200.0, "top")  # type: ignore[arg-type]

    def test_missing_source_entry_raises_artifact_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            manifest_path = artifacts_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "model_family": "extra_trees",
                        "sources": {
                            "top": {
                                "model_path": "top_model.pkl",
                            }
                        },
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (artifacts_dir / "top_model.pkl").write_bytes(
                pickle.dumps(FixedOutputModel(x_value=1.0, y_value=2.0))
            )

            with self.assertRaisesRegex(ArtifactLoadError, "missing a valid source entry"):
                load_artifacts(artifacts_dir)

    @staticmethod
    def _write_artifact_bundle(
            artifacts_dir: Path,
            top_model: object,
            bottom_model: object,
    ) -> None:
        manifest = {
            "schema_version": "1",
            "model_family": "extra_trees",
            "sources": {
                "top": {
                    "model_path": "top_model.pkl",
                },
                "bottom": {
                    "model_path": "bottom_model.pkl",
                },
            },
        }
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (artifacts_dir / "top_model.pkl").write_bytes(pickle.dumps(top_model))
        (artifacts_dir / "bottom_model.pkl").write_bytes(pickle.dumps(bottom_model))


if __name__ == "__main__":
    unittest.main()
