"""Inference-only Baseline and FELF-SLR model definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "RECOGNIZER_DEVICE=cuda was requested, but CUDA is unavailable."
        )
    if requested not in {"cpu", "cuda"}:
        raise ValueError("RECOGNIZER_DEVICE must be auto, cpu, or cuda.")
    return torch.device(requested)


def _extract_state_dict(payload) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint does not contain a state dictionary.")
    state = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized = str(key)
        for prefix in ("module.", "_orig_mod."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        state[normalized] = value
    if not state:
        raise ValueError("Checkpoint state dictionary is empty.")
    return state


def load_checkpoint(
    module: nn.Module,
    path: Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Recognizer checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = _extract_state_dict(payload)
    expected_keys = set(module.state_dict())
    matched_keys = expected_keys.intersection(state)
    coverage = len(matched_keys) / max(1, len(expected_keys))
    if coverage < 0.90:
        raise RuntimeError(
            f"Checkpoint {path} matches only {coverage:.1%} of the model state. "
            "Check the recognizer architecture and class-label configuration."
        )
    missing, unexpected = module.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key != "n_averaged"]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Incompatible checkpoint {path}: missing={missing}, "
            f"unexpected={unexpected}"
        )


class BaselinePositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_length: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, dimension)
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dimension)
        )
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer(
            "pe", encoding.unsqueeze(0).transpose(0, 1), persistent=True
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.dropout(source + self.pe[:source.size(0)])


class BaselineTransformer(nn.Module):
    """The original ViSTAR 457-D Transformer recognizer."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = 457,
        model_dim: int = 256,
        heads: int = 4,
        layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.2,
        sequence_length: int = 40,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            input_size, model_dim, kernel_size=3, padding=1
        )
        self.conv_residual = nn.Linear(input_size, model_dim)
        self.conv_norm = nn.LayerNorm(model_dim)
        self.input_linear = nn.Linear(model_dim, model_dim)
        self.positional_encoding = BaselinePositionalEncoding(
            model_dim, dropout, sequence_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )
        self.transformer_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        residual = self.conv_residual(source)
        source = self.conv1d(source.transpose(1, 2)).transpose(1, 2)
        source = self.conv_norm(source + residual)
        source = self.input_linear(source).transpose(0, 1)
        source = self.positional_encoding(source)
        memory = self.transformer_encoder(source)
        memory = self.transformer_norm(memory + source).mean(dim=0)
        return self.classifier(self.dropout(memory))


class CosineClassifier(nn.Module):
    def __init__(self, dimension: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=-1)
        weights = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * (features @ weights.t())


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_length: int = 40):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, dimension)
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dimension)
        )
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(
            positions * divisor[:dimension // 2]
        )
        self.register_buffer(
            "pe", encoding.unsqueeze(0).transpose(0, 1), persistent=True
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.dropout(source + self.pe[:source.size(0)])


class BranchStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def temporal_delta(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = torch.zeros_like(features)
    velocity[:, 1:] = features[:, 1:] - features[:, :-1]
    acceleration = torch.zeros_like(features)
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    return velocity, acceleration


class StaticMotionStem(nn.Module):
    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.static_stem = BranchStem(input_dim, branch_dim)
        self.motion_stem = BranchStem(input_dim * 2, branch_dim)
        hidden_dim = max(32, branch_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, branch_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        velocity, acceleration = temporal_delta(features)
        static = self.static_stem(features)
        motion = self.motion_stem(
            torch.cat([velocity, acceleration], dim=-1)
        )
        summary = torch.cat(
            [
                static.mean(dim=1),
                static.std(dim=1, unbiased=False),
                motion.mean(dim=1),
                motion.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(summary)).unsqueeze(1)
        return static + gate * motion


class ResidualReliabilityFusion(nn.Module):
    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        beta: float = 0.25,
    ):
        super().__init__()
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=False)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        global_features: torch.Tensor,
    ):
        def summarize(features: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    features.mean(dim=1),
                    features.std(dim=1, unbiased=False),
                ],
                dim=-1,
            )

        summary = torch.cat(
            [summarize(left), summarize(right), summarize(global_features)],
            dim=-1,
        )
        weights = torch.softmax(self.net(summary), dim=-1)
        return (
            left + self.beta * weights[:, 0, None, None] * left,
            right + self.beta * weights[:, 1, None, None] * right,
            global_features
            + self.beta * weights[:, 2, None, None] * global_features,
        )


class LocalGlobalExpert(nn.Module):
    def __init__(
        self,
        num_classes: int,
        branch_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.2,
        expert_kind: str = "baseline",
    ):
        super().__init__()
        self.left_stem = BranchStem(165, branch_dim)
        self.right_stem = BranchStem(165, branch_dim)
        self.global_stem = BranchStem(23, branch_dim)
        self.expert_kind = expert_kind
        if expert_kind == "lrg":
            self.lrg_left_stem = StaticMotionStem(165, branch_dim)
            self.lrg_right_stem = StaticMotionStem(165, branch_dim)
            self.lrg_global_stem = StaticMotionStem(23, branch_dim)
            self.lrg_project = nn.Linear(branch_dim * 3, branch_dim * 3)
            self.lrg_gamma = nn.Parameter(
                torch.tensor(0.25), requires_grad=False
            )
        if expert_kind == "rf":
            self.residual_reliability_fusion = ResidualReliabilityFusion(
                branch_dim, branch_dim, branch_dim
            )
        model_dim = branch_dim * 3
        if model_dim % heads != 0:
            raise ValueError(
                f"FELF model dimension {model_dim} must divide {heads} heads."
            )
        self.conv = nn.Conv1d(model_dim, model_dim, 3, padding=1)
        self.res = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.proj = nn.Linear(model_dim, model_dim)
        self.pe = PositionalEncoding(model_dim, dropout, 40)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = CosineClassifier(model_dim, num_classes)

    def forward_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        raw_left, raw_right, raw_global = left, right, global_features
        left = self.left_stem(left)
        right = self.right_stem(right)
        global_features = self.global_stem(global_features)
        if self.expert_kind == "lrg":
            motion = torch.cat(
                [
                    self.lrg_left_stem(raw_left),
                    self.lrg_right_stem(raw_right),
                    self.lrg_global_stem(raw_global),
                ],
                dim=-1,
            )
            base = torch.cat([left, right, global_features], dim=-1)
            mixed = base + self.lrg_gamma * self.lrg_project(motion)
            left, right, global_features = torch.split(
                mixed, [left.shape[-1], right.shape[-1], global_features.shape[-1]],
                dim=-1,
            )
        if self.expert_kind == "rf":
            left, right, global_features = self.residual_reliability_fusion(
                left, right, global_features
            )
        source = torch.cat([left, right, global_features], dim=-1)
        residual = self.res(source)
        source = self.conv(source.transpose(1, 2)).transpose(1, 2)
        source = self.norm(source + residual)
        source = self.proj(source).transpose(0, 1)
        source = self.pe(source)
        memory = self.encoder(source)
        memory = self.out_norm(memory + source)
        return self.drop(memory.mean(dim=0))

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(
            self.forward_features(left, right, global_features)
        )


class FELFStageOne(nn.Module):
    def __init__(
        self,
        num_classes: int,
        branch_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        lrg_weight: float,
        rf_weight: float,
    ):
        super().__init__()
        kwargs = dict(
            num_classes=num_classes,
            branch_dim=branch_dim,
            layers=layers,
            heads=heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.main = LocalGlobalExpert(**kwargs, expert_kind="baseline")
        self.lrg = LocalGlobalExpert(**kwargs, expert_kind="lrg")
        self.rf = LocalGlobalExpert(**kwargs, expert_kind="rf")
        self.register_buffer(
            "lrg_logit_weight_param",
            torch.tensor(float(lrg_weight)),
            persistent=False,
        )
        self.register_buffer(
            "rf_logit_weight_param",
            torch.tensor(float(rf_weight)),
            persistent=False,
        )

    def forward_all(self, left, right, global_features):
        main = self.main(left, right, global_features)
        lrg = self.lrg(left, right, global_features)
        rf = self.rf(left, right, global_features)
        normalizer = (
            1.0
            + torch.abs(self.lrg_logit_weight_param)
            + torch.abs(self.rf_logit_weight_param)
        )
        fused = (
            main
            + self.lrg_logit_weight_param * lrg
            + self.rf_logit_weight_param * rf
        ) / normalizer
        return {"main": main, "lrg": lrg, "rf": rf, "fused": fused}

    def forward(self, left, right, global_features):
        return self.forward_all(left, right, global_features)["fused"]


class FactorEncoder(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, dropout: float, heads: int = 4
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(
            output_dim, output_dim, kernel_size=3, padding=1
        )
        self.norm = nn.LayerNorm(output_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(features)
        temporal = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        return self.encoder(self.norm(hidden + temporal)).mean(dim=1)


class MorphTrajExpert(nn.Module):
    def __init__(
        self,
        morph_dim: int,
        trajectory_dim: int,
        orientation_dim: int,
        num_classes: int,
        branch_dim: int,
        dropout: float,
        conditioning: str = "none",
    ):
        super().__init__()
        if conditioning not in {
            "none",
            "film",
            "morph_gate_residual",
            "morph_gate_scale",
        }:
            raise ValueError(
                f"Unsupported FELF_MT_CONDITIONING={conditioning!r}."
            )
        self.conditioning = conditioning
        self.morph = FactorEncoder(morph_dim, branch_dim, dropout)
        self.traj = FactorEncoder(trajectory_dim, branch_dim, dropout)
        self.orient = FactorEncoder(orientation_dim, branch_dim, dropout)
        if conditioning == "film":
            self.traj_film = nn.Sequential(
                nn.LayerNorm(branch_dim), nn.Linear(branch_dim, branch_dim * 2)
            )
            self.orient_film = nn.Sequential(
                nn.LayerNorm(branch_dim), nn.Linear(branch_dim, branch_dim * 2)
            )
        elif conditioning in {"morph_gate_residual", "morph_gate_scale"}:
            self.traj_gate = nn.Sequential(
                nn.LayerNorm(branch_dim), nn.Linear(branch_dim, branch_dim)
            )
        self.fuse = nn.Sequential(
            nn.LayerNorm(branch_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(branch_dim * 3, branch_dim * 3),
            nn.GELU(),
            nn.LayerNorm(branch_dim * 3),
        )
        self.morph_classifier = CosineClassifier(branch_dim, num_classes)
        self.traj_classifier = CosineClassifier(branch_dim, num_classes)
        self.orient_classifier = CosineClassifier(branch_dim, num_classes)
        self.classifier = CosineClassifier(branch_dim * 3, num_classes)

    def forward(self, morphology, trajectory, orientation):
        morph_embedding = self.morph(morphology)
        trajectory_embedding = self.traj(trajectory)
        orientation_embedding = self.orient(orientation)
        if self.conditioning == "film":
            trajectory_scale, trajectory_bias = self.traj_film(
                morph_embedding
            ).chunk(2, dim=-1)
            orientation_scale, orientation_bias = self.orient_film(
                morph_embedding
            ).chunk(2, dim=-1)
            trajectory_embedding = (
                (1.0 + trajectory_scale) * trajectory_embedding
                + trajectory_bias
            )
            orientation_embedding = (
                (1.0 + orientation_scale) * orientation_embedding
                + orientation_bias
            )
        elif self.conditioning == "morph_gate_residual":
            gate = torch.sigmoid(self.traj_gate(morph_embedding))
            trajectory_embedding = trajectory_embedding + gate * trajectory_embedding
        elif self.conditioning == "morph_gate_scale":
            gate = torch.sigmoid(self.traj_gate(morph_embedding))
            trajectory_embedding = gate * trajectory_embedding
        fused_embedding = self.fuse(
            torch.cat(
                [morph_embedding, trajectory_embedding, orientation_embedding],
                dim=-1,
            )
        )
        return {
            "morph": self.morph_classifier(morph_embedding),
            "traj": self.traj_classifier(trajectory_embedding),
            "orient": self.orient_classifier(orientation_embedding),
            "fused": self.classifier(fused_embedding),
        }
