from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from matchtrace.data.synthetic import (
    generate_all,
    generate_classification_dataset,
)
from matchtrace.evaluation.evaluate import evaluate_model
from matchtrace.inference.pipeline import (
    AnalysisPaths,
    VideoAnalyzer,
    analyze_video,
)
from matchtrace.service import _safe_child
from matchtrace.training.train import load_dataset, train_model
from matchtrace.utils.config import load_settings
from matchtrace.utils.fingerprint import file_sha256


class ProductionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.settings = copy.deepcopy(load_settings())
        cls.settings["paths"] = {
            "dataset": str(cls.root / "dataset.npz"),
            "demo_video": str(cls.root / "demo.mp4"),
            "demo_truth": str(cls.root / "truth.json"),
            "model": str(cls.root / "model.pt"),
            "evaluation": str(cls.root / "evaluation.json"),
            "output_video": str(cls.root / "output.mp4"),
            "run_summary": str(cls.root / "summary.json"),
            "log": str(cls.root / "run.jsonl"),
        }
        cls.settings["synthetic"].update(
            {
                "train_samples": 120,
                "validation_samples": 40,
                "test_samples": 40,
                "video_width": 320,
                "video_height": 180,
                "video_frames": 24,
                "players_per_team": 3,
            }
        )
        cls.settings["model"].update(
            {
                "epochs": 3,
                "batch_size": 32,
                "channels": 16,
                "torch_threads": 1,
            }
        )
        generate_all(cls.settings)
        cls.training = train_model(cls.settings)
        cls.evaluation = evaluate_model(cls.settings)
        cls.analysis = analyze_video(cls.settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_dataset_loading_is_complete_and_reproducible(self) -> None:
        arrays = load_dataset(self.settings["paths"]["dataset"])
        self.assertEqual(arrays["train_images"].shape, (120, 32, 32, 3))
        second = self.root / "second_dataset.npz"
        generate_classification_dataset(
            second,
            seed=int(self.settings["project"]["seed"]),
            image_size=32,
            train_samples=120,
            validation_samples=40,
            test_samples=40,
        )
        with np.load(second, allow_pickle=False) as duplicate:
            np.testing.assert_array_equal(
                arrays["train_images"],
                duplicate["train_images"],
            )
            np.testing.assert_array_equal(
                arrays["train_labels"],
                duplicate["train_labels"],
            )

    def test_training_and_evaluation_meet_synthetic_threshold(self) -> None:
        self.assertTrue(Path(self.training["checkpoint"]).is_file())
        self.assertGreaterEqual(
            self.training["best_validation_accuracy"],
            0.95,
        )
        self.assertGreaterEqual(self.evaluation["accuracy"], 0.95)

    def test_inference_preserves_frame_contract(self) -> None:
        output = Path(self.settings["paths"]["output_video"])
        self.assertTrue(output.is_file())
        self.assertEqual(self.analysis["frames"], 24)
        self.assertEqual(
            self.analysis["frames"],
            self.analysis["expected_frames"],
        )
        capture = cv2.VideoCapture(str(output))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 24)
        success, frame = capture.read()
        capture.release()
        self.assertTrue(success)
        self.assertEqual(frame.shape[:2], (180, 320))
        self.assertEqual(self.analysis["version"], "3.0.0")
        fingerprints = self.analysis["fingerprints"]
        self.assertEqual(
            fingerprints["output_sha256"],
            file_sha256(output),
        )
        self.assertEqual(len(fingerprints["settings_sha256"]), 64)
        self.assertEqual(len(fingerprints["model_sha256"]), 64)

    def test_api_path_guard_rejects_traversal(self) -> None:
        with self.assertRaises(ValueError):
            _safe_child(self.root, "../outside.mp4")

    def test_inference_rejects_overlapping_artifact_paths(self) -> None:
        source = self.settings["paths"]["demo_video"]
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            analyze_video(
                self.settings,
                input_path=source,
                output_path=source,
            )

    def test_analyzer_can_be_reused_without_leaking_track_state(self) -> None:
        analyzer = VideoAnalyzer(self.settings)
        first_paths = AnalysisPaths.from_settings(
            self.settings,
            self.settings["paths"]["demo_video"],
            self.root / "reuse_first.mp4",
            self.root / "reuse_first.json",
        )
        second_paths = AnalysisPaths.from_settings(
            self.settings,
            self.settings["paths"]["demo_video"],
            self.root / "reuse_second.mp4",
            self.root / "reuse_second.json",
        )

        first = analyzer.run(first_paths)
        second = analyzer.run(second_paths)

        self.assertEqual(first["frames"], second["frames"])
        self.assertEqual(
            first["distance_by_track_m"].keys(),
            second["distance_by_track_m"].keys(),
        )


if __name__ == "__main__":
    unittest.main()
