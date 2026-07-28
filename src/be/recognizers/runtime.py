"""Runtime adapters for the Baseline and FELF-SLR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .config import RecognizerSettings
from .geometry import (
    extract_mt_sequence,
    extract_part_aware_sequence,
    pack_raw_frame,
)
from .models import (
    BaselineTransformer,
    FELFStageOne,
    MorphTrajExpert,
    load_checkpoint,
    resolve_device,
)


class SignRecognizer(ABC):
    def __init__(self, settings: RecognizerSettings):
        self.settings = settings
        self.actions = np.asarray(settings.labels)
        self.actions_tts = np.asarray(settings.tts_labels)
        self.device = resolve_device(settings.device)

    @property
    def mode(self) -> str:
        return self.settings.mode

    @abstractmethod
    def prepare_frame(
        self,
        pose_landmarks: np.ndarray | None,
        left_hand_landmarks: np.ndarray | None,
        right_hand_landmarks: np.ndarray | None,
        *,
        legacy_features: np.ndarray,
    ) -> np.ndarray:
        """Convert one detected pose into this recognizer's frame input."""

    @abstractmethod
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Return one probability vector for a complete 40-frame sequence."""

    def status(self) -> dict[str, object]:
        return {
            **self.settings.public_dict(),
            "active_device": str(self.device),
            "labels": list(self.settings.labels),
        }


class BaselineRecognizer(SignRecognizer):
    def __init__(self, settings: RecognizerSettings):
        super().__init__(settings)
        self.model = BaselineTransformer(num_classes=len(self.actions))
        load_checkpoint(
            self.model,
            settings.baseline_checkpoint,
            self.device,
            strict=True,
        )
        self.model.to(self.device).eval()

    def prepare_frame(
        self,
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        *,
        legacy_features,
    ) -> np.ndarray:
        features = np.asarray(legacy_features, dtype=np.float32)
        if features.shape != (457,):
            raise ValueError(
                f"Baseline expects a 457-D frame, got {features.shape}."
            )
        return features

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        expected = (self.settings.sequence_length, 457)
        if sequence.shape != expected:
            raise ValueError(
                f"Baseline expects sequence {expected}, got {sequence.shape}."
            )
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    def status(self) -> dict[str, object]:
        return {
            **super().status(),
            "input": "legacy_457",
            "checkpoint": str(self.settings.baseline_checkpoint),
            "stage": "single_baseline",
        }


def _load_felf_stage_one(
    model: FELFStageOne,
    settings: RecognizerSettings,
    device: torch.device,
) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if settings.felf_checkpoint is not None:
        load_checkpoint(
            model, settings.felf_checkpoint, device, strict=False
        )
        loaded["combined"] = str(settings.felf_checkpoint)
        return loaded

    paths: list[tuple[str, torch.nn.Module, Path | None]] = [
        ("baseline", model.main, settings.felf_baseline_checkpoint),
        ("lrg", model.lrg, settings.felf_lrg_checkpoint),
        ("rf", model.rf, settings.felf_rf_checkpoint),
    ]
    missing = [name for name, _, path in paths if path is None]
    if missing:
        raise ValueError(
            "SIGN_RECOGNIZER=felf_slr requires FELF_CHECKPOINT or all of "
            "FELF_BASELINE_CHECKPOINT, FELF_LRG_CHECKPOINT, and "
            f"FELF_RF_CHECKPOINT. Missing: {', '.join(missing)}"
        )
    for name, expert, path in paths:
        assert path is not None
        load_checkpoint(expert, path, device, strict=True)
        loaded[name] = str(path)
    return loaded


class FELFSLRRecognizer(SignRecognizer):
    def __init__(self, settings: RecognizerSettings):
        super().__init__(settings)
        self.stage_one = FELFStageOne(
            num_classes=len(self.actions),
            branch_dim=settings.felf_branch_dim,
            layers=settings.felf_num_layers,
            heads=settings.felf_num_heads,
            feedforward_dim=settings.felf_ff_dim,
            dropout=settings.felf_dropout,
            lrg_weight=settings.felf_lrg_weight,
            rf_weight=settings.felf_rf_weight,
        )
        self.loaded_checkpoints = _load_felf_stage_one(
            self.stage_one, settings, self.device
        )
        self.stage_one.to(self.device).eval()

        self.mt_model: MorphTrajExpert | None = None
        if settings.felf_mt_checkpoint is not None:
            if settings.felf_mt_feature_version != "v1":
                raise ValueError(
                    "ViSTAR currently supports FELF_MT_FEATURE_VERSION=v1."
                )
            self.mt_model = MorphTrajExpert(
                morph_dim=310,
                trajectory_dim=131,
                orientation_dim=40,
                num_classes=len(self.actions),
                branch_dim=settings.felf_mt_branch_dim,
                dropout=settings.felf_dropout,
                conditioning=settings.felf_mt_conditioning,
            )
            load_checkpoint(
                self.mt_model,
                settings.felf_mt_checkpoint,
                self.device,
                strict=True,
            )
            self.mt_model.to(self.device).eval()
            self.loaded_checkpoints["mt"] = str(settings.felf_mt_checkpoint)
        elif settings.felf_require_mt:
            raise ValueError(
                "FELF_REQUIRE_MT=true requires FELF_MT_CHECKPOINT. Set the "
                "checkpoint trained for the configured ViSTAR labels, or set "
                "FELF_REQUIRE_MT=false to run Stage 1 only."
            )

    def prepare_frame(
        self,
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        *,
        legacy_features,
    ) -> np.ndarray:
        return pack_raw_frame(
            pose_landmarks, left_hand_landmarks, right_hand_landmarks
        )

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        left, right, global_features = extract_part_aware_sequence(sequence)
        left_tensor = torch.from_numpy(left).unsqueeze(0).to(self.device)
        right_tensor = torch.from_numpy(right).unsqueeze(0).to(self.device)
        global_tensor = (
            torch.from_numpy(global_features).unsqueeze(0).to(self.device)
        )
        with torch.inference_mode():
            stage_one = self.stage_one.forward_all(
                left_tensor, right_tensor, global_tensor
            )
            logits = stage_one["fused"]
            if self.mt_model is not None:
                morphology, trajectory, orientation = extract_mt_sequence(
                    sequence,
                    feature_version=self.settings.felf_mt_feature_version,
                )
                mt_logits = self.mt_model(
                    torch.from_numpy(morphology).unsqueeze(0).to(self.device),
                    torch.from_numpy(trajectory).unsqueeze(0).to(self.device),
                    torch.from_numpy(orientation).unsqueeze(0).to(self.device),
                )["fused"]
                logits = (
                    self.settings.felf_stage2_baseline_weight
                    * stage_one["main"]
                    + self.settings.felf_stage2_felf_weight
                    * stage_one["fused"]
                    + self.settings.felf_stage2_mt_weight * mt_logits
                )
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    def status(self) -> dict[str, object]:
        return {
            **super().status(),
            "input": "raw_pose_to_165_165_23",
            "stage": "stage2_with_mt" if self.mt_model else "stage1_felf",
            "checkpoints": self.loaded_checkpoints,
            "stage1_weights": {
                "lrg": self.settings.felf_lrg_weight,
                "rf": self.settings.felf_rf_weight,
            },
            "stage2_weights": {
                "baseline": self.settings.felf_stage2_baseline_weight,
                "felf": self.settings.felf_stage2_felf_weight,
                "mt": self.settings.felf_stage2_mt_weight,
            },
        }
