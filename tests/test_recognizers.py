from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.be.recognizers.config import RecognizerSettings
from src.be.recognizers.geometry import (
    extract_mt_sequence,
    extract_part_aware_sequence,
)
from src.be.recognizers.models import (
    BaselineTransformer,
    FELFStageOne,
    MorphTrajExpert,
)
from src.be.recognizers.runtime import BaselineRecognizer, FELFSLRRecognizer


class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.base_env = {
            "RECOGNIZER_DEVICE": "cpu",
            "RECOGNIZER_SEQUENCE_LENGTH": "40",
            "RECOGNIZER_THRESHOLD": "0.70",
            "RECOGNIZER_LABELS_FILE": "",
        }

    def test_geometric_feature_dimensions(self):
        sequence = np.zeros((40, 258), dtype=np.float32)
        left, right, global_features = extract_part_aware_sequence(sequence)
        morphology, trajectory, orientation = extract_mt_sequence(sequence)
        self.assertEqual(left.shape, (40, 165))
        self.assertEqual(right.shape, (40, 165))
        self.assertEqual(global_features.shape, (40, 23))
        self.assertEqual(morphology.shape, (40, 310))
        self.assertEqual(trajectory.shape, (40, 131))
        self.assertEqual(orientation.shape, (40, 40))
        self.assertTrue(np.isfinite(left).all())
        self.assertTrue(np.isfinite(trajectory).all())

    def test_invalid_mode_is_rejected(self):
        with patch.dict(
            os.environ,
            {**self.base_env, "SIGN_RECOGNIZER": "unknown"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "SIGN_RECOGNIZER"):
                RecognizerSettings.from_env()

    def test_baseline_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "baseline.pth"
            torch.save(
                BaselineTransformer(num_classes=19).state_dict(), checkpoint
            )
            with patch.dict(
                os.environ,
                {
                    **self.base_env,
                    "SIGN_RECOGNIZER": "baseline",
                    "BASELINE_CHECKPOINT": str(checkpoint),
                },
                clear=False,
            ):
                recognizer = BaselineRecognizer(
                    RecognizerSettings.from_env()
                )
                probabilities = recognizer.predict_proba(
                    np.zeros((40, 457), dtype=np.float32)
                )
            self.assertEqual(probabilities.shape, (19,))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_felf_stage_one_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "felf.pth"
            model = FELFStageOne(
                num_classes=19,
                branch_dim=16,
                layers=1,
                heads=4,
                feedforward_dim=64,
                dropout=0.0,
                lrg_weight=1.0,
                rf_weight=0.75,
            )
            torch.save(model.state_dict(), checkpoint)
            with patch.dict(
                os.environ,
                {
                    **self.base_env,
                    "SIGN_RECOGNIZER": "felf_slr",
                    "FELF_CHECKPOINT": str(checkpoint),
                    "FELF_REQUIRE_MT": "false",
                    "FELF_MT_CHECKPOINT": "",
                    "FELF_BRANCH_DIM": "16",
                    "FELF_NUM_LAYERS": "1",
                    "FELF_NUM_HEADS": "4",
                    "FELF_FF_DIM": "64",
                    "FELF_DROPOUT": "0.0",
                },
                clear=False,
            ):
                recognizer = FELFSLRRecognizer(
                    RecognizerSettings.from_env()
                )
                probabilities = recognizer.predict_proba(
                    np.zeros((40, 258), dtype=np.float32)
                )
            self.assertEqual(probabilities.shape, (19,))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_complete_felf_stage_two_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            felf_checkpoint = Path(directory) / "felf.pth"
            mt_checkpoint = Path(directory) / "mt.pth"
            torch.save(
                FELFStageOne(
                    num_classes=19,
                    branch_dim=16,
                    layers=1,
                    heads=4,
                    feedforward_dim=64,
                    dropout=0.0,
                    lrg_weight=1.0,
                    rf_weight=0.75,
                ).state_dict(),
                felf_checkpoint,
            )
            torch.save(
                MorphTrajExpert(
                    morph_dim=310,
                    trajectory_dim=131,
                    orientation_dim=40,
                    num_classes=19,
                    branch_dim=16,
                    dropout=0.0,
                ).state_dict(),
                mt_checkpoint,
            )
            with patch.dict(
                os.environ,
                {
                    **self.base_env,
                    "SIGN_RECOGNIZER": "felf_slr",
                    "FELF_CHECKPOINT": str(felf_checkpoint),
                    "FELF_REQUIRE_MT": "true",
                    "FELF_MT_CHECKPOINT": str(mt_checkpoint),
                    "FELF_BRANCH_DIM": "16",
                    "FELF_NUM_LAYERS": "1",
                    "FELF_NUM_HEADS": "4",
                    "FELF_FF_DIM": "64",
                    "FELF_DROPOUT": "0.0",
                    "FELF_MT_BRANCH_DIM": "16",
                    "FELF_MT_CONDITIONING": "none",
                },
                clear=False,
            ):
                recognizer = FELFSLRRecognizer(
                    RecognizerSettings.from_env()
                )
                probabilities = recognizer.predict_proba(
                    np.zeros((40, 258), dtype=np.float32)
                )
            self.assertEqual(probabilities.shape, (19,))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
