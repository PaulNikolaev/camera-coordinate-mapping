"""Smoke test for the one-command training and evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from solution.pipeline import run_pipeline


class RunPipelineTests(unittest.TestCase):
    """Validate that the pipeline trains and evaluates from one entrypoint."""

    def test_run_pipeline_writes_artifacts_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            data_root = workspace / "coord_data"
            artifacts_dir = workspace / "artifacts"
            output_metrics = workspace / "reports" / "metrics.json"

            self._write_dataset(data_root)

            result = run_pipeline(
                data_root=data_root,
                artifacts_dir=artifacts_dir,
                n_estimators=50,
                min_samples_leaf=1,
                seed=7,
                output_metrics=output_metrics,
            )

            self.assertEqual(result.training.artifacts_dir, artifacts_dir)
            self.assertEqual(result.evaluation.metrics_path, output_metrics)
            self.assertTrue((artifacts_dir / "manifest.json").is_file())
            self.assertTrue((artifacts_dir / "training_report.json").is_file())
            self.assertTrue(output_metrics.is_file())

            metrics_payload = json.loads(output_metrics.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["split"], "val")
            self.assertEqual(set(metrics_payload["sources"]), {"top", "bottom"})
            self.assertLess(result.evaluation.overall_med, 1e-6)

    @staticmethod
    def _write_dataset(data_root: Path) -> None:
        split_payload = {
            "train": ["train/session_train"],
            "val": ["val/session_val"],
        }
        RunPipelineTests._write_json_file(data_root / "split.json", split_payload)

        for split_name, session_name in (
                ("train", "session_train"),
                ("val", "session_val"),
        ):
            RunPipelineTests._write_session(
                data_root=data_root,
                split_name=split_name,
                session_name=session_name,
            )

    @staticmethod
    def _write_session(data_root: Path, split_name: str, session_name: str) -> None:
        session_dir = data_root / split_name / session_name
        (session_dir / "door2").mkdir(parents=True, exist_ok=True)
        (session_dir / "top").mkdir(parents=True, exist_ok=True)
        (session_dir / "bottom").mkdir(parents=True, exist_ok=True)

        (session_dir / "door2" / "door2_frame.jpg").write_bytes(b"jpg")
        (session_dir / "top" / "top_frame.jpg").write_bytes(b"jpg")
        (session_dir / "bottom" / "bottom_frame.jpg").write_bytes(b"jpg")

        for source in ("top", "bottom"):
            record = RunPipelineTests._build_record(source=source)
            RunPipelineTests._write_json_file(session_dir / f"coords_{source}.json", [record])

    @staticmethod
    def _build_record(source: str) -> dict[str, object]:
        points = RunPipelineTests._build_points(x_offset=0.0, y_offset=0.0)
        return {
            "file1_path": "door2/door2_frame.jpg",
            "file2_path": f"{source}/{source}_frame.jpg",
            "image1_coordinates": points,
            "image2_coordinates": points,
        }

    @staticmethod
    def _build_points(x_offset: float, y_offset: float) -> list[dict[str, float | int]]:
        return [
            {
                "number": index,
                "x": float(100 + (index * 10)) + x_offset,
                "y": float(200 + (index * index)) + y_offset,
            }
            for index in range(1, 18)
        ]

    @staticmethod
    def _write_json_file(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
