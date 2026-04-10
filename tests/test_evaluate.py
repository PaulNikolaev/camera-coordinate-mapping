"""End-to-end tests for validation-split evaluation."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from solution.evaluate import evaluate_and_save_metrics


class IdentityPixelModel:
    """Model stub that maps normalized inputs back to pixel coordinates."""

    @staticmethod
    def predict(features: list[list[float]]) -> list[list[float]]:
        x_value, y_value = features[0]
        return [[x_value * 3200.0, y_value * 1800.0]]


class EvaluateModelsTests(unittest.TestCase):
    """Validate MED calculation and persisted metrics output."""

    def test_evaluate_uses_val_split_and_writes_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            data_root = workspace / "coord_data"
            artifacts_dir = workspace / "artifacts"
            self._write_dataset(data_root)
            self._write_artifacts(artifacts_dir)

            result = evaluate_and_save_metrics(
                data_root=data_root,
                artifacts_dir=artifacts_dir,
            )

            self.assertEqual(result.metrics_path, artifacts_dir / "metrics.json")
            self.assertEqual(result.point_count, 34)
            self.assertEqual(result.record_count, 2)
            self.assertAlmostEqual(result.overall_med, 0.0)
            self.assertAlmostEqual(result.source_metrics["top"].med, 0.0)
            self.assertAlmostEqual(result.source_metrics["bottom"].med, 0.0)
            self.assertEqual(result.source_metrics["top"].point_count, 17)
            self.assertEqual(result.source_metrics["bottom"].point_count, 17)
            self.assertEqual(result.source_metrics["top"].record_count, 1)
            self.assertEqual(result.source_metrics["bottom"].record_count, 1)

            metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["split"], "val")
            self.assertEqual(metrics_payload["overall"]["point_count"], 34)
            self.assertEqual(metrics_payload["overall"]["record_count"], 2)
            self.assertEqual(metrics_payload["overall"]["session_count"], 1)
            self.assertEqual(metrics_payload["sources"]["top"]["point_count"], 17)
            self.assertEqual(metrics_payload["sources"]["bottom"]["point_count"], 17)
            self.assertEqual(metrics_payload["preparation_report"]["split_name"], "val")
            self.assertEqual(metrics_payload["preparation_report"]["matched_samples"], 34)

    @staticmethod
    def _write_dataset(data_root: Path) -> None:
        split_payload = {
            "train": ["train/session_train"],
            "val": ["val/session_val"],
        }
        (data_root / "split.json").parent.mkdir(parents=True, exist_ok=True)
        (data_root / "split.json").write_text(
            json.dumps(split_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        EvaluateModelsTests._write_session(data_root, split_name="train", session_name="session_train")
        EvaluateModelsTests._write_session(data_root, split_name="val", session_name="session_val")

    @staticmethod
    def _write_session(data_root: Path, split_name: str, session_name: str) -> None:
        session_dir = data_root / split_name / session_name
        (session_dir / "door2").mkdir(parents=True, exist_ok=True)
        (session_dir / "top").mkdir(parents=True, exist_ok=True)
        (session_dir / "bottom").mkdir(parents=True, exist_ok=True)

        for image_name in ("door2_frame.jpg", "top_frame.jpg", "bottom_frame.jpg"):
            if image_name.startswith("door2"):
                (session_dir / "door2" / image_name).write_bytes(b"jpg")
            elif image_name.startswith("top"):
                (session_dir / "top" / image_name).write_bytes(b"jpg")
            else:
                (session_dir / "bottom" / image_name).write_bytes(b"jpg")

        top_record = {
            "file1_path": "door2/door2_frame.jpg",
            "file2_path": "top/top_frame.jpg",
            "image1_coordinates": EvaluateModelsTests._build_points(),
            "image2_coordinates": EvaluateModelsTests._build_points(),
        }
        bottom_record = {
            "file1_path": "door2/door2_frame.jpg",
            "file2_path": "bottom/bottom_frame.jpg",
            "image1_coordinates": EvaluateModelsTests._build_points(),
            "image2_coordinates": EvaluateModelsTests._build_points(),
        }

        (session_dir / "coords_top.json").write_text(
            json.dumps([top_record], ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (session_dir / "coords_bottom.json").write_text(
            json.dumps([bottom_record], ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _build_points() -> list[dict[str, float | int]]:
        return [
            {
                "number": index,
                "x": float(100 + (index * 10)),
                "y": float(200 + (index * 5)),
            }
            for index in range(1, 18)
        ]

    @staticmethod
    def _write_artifacts(artifacts_dir: Path) -> None:
        manifest = {
            "schema_version": "1",
            "model_family": "polynomial_ridge",
            "sources": {
                "top": {
                    "model_path": "top_model.pkl",
                },
                "bottom": {
                    "model_path": "bottom_model.pkl",
                },
            },
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifacts_dir / "top_model.pkl").write_bytes(pickle.dumps(IdentityPixelModel()))
        (artifacts_dir / "bottom_model.pkl").write_bytes(pickle.dumps(IdentityPixelModel()))


if __name__ == "__main__":
    unittest.main()
