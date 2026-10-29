"""Narrowband and non-narrowband decisions with their physical and neural branches."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dtfe import dtfe_backbone


class TemporalProfileEncoder(nn.Module):

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.SiLU(),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.network(profile)


def _logit_for_target_reliability(target: float) -> float:
    probability = (target - 0.25) / 0.95
    return math.log(probability / (1.0 - probability))


class ProfileEncoder(nn.Module):

    def __init__(self, input_channels: int, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels, 16, kernel_size=9, stride=2, padding=4, bias=False
            ),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.SiLU(),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.network(profile)


def _inverse_sigmoid(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def node2_physics_score(raw_concepts: torch.Tensor) -> torch.Tensor:
    if raw_concepts.ndim != 2 or raw_concepts.shape[1] != 5:
        raise ValueError(
            f"Node2 expects five concepts, got {tuple(raw_concepts.shape)}"
        )
    strong = torch.minimum(
        (raw_concepts[:, 0] - 10.0) / 2.0, (raw_concepts[:, 1] - 3.0) / 0.5
    )
    persistent = torch.minimum(
        raw_concepts[:, 2] - 4.0, (raw_concepts[:, 3] - 0.02) / 0.01
    )
    return torch.maximum(strong, persistent)


def node3_physics_score(raw_concepts: torch.Tensor) -> torch.Tensor:
    if raw_concepts.ndim != 2 or raw_concepts.shape[1] != 3:
        raise ValueError(
            f"Node3 expects three concepts, got {tuple(raw_concepts.shape)}"
        )
    pc_lag = raw_concepts[:, 0]
    concentration = raw_concepts[:, 1]
    iq_lag = raw_concepts[:, 2]
    primary = (0.05 - pc_lag) / 0.02
    fallback = torch.minimum((0.05 - iq_lag) / 0.02, (concentration - 0.8) / 0.1)
    return torch.maximum(primary, fallback)


def base_consistency_descriptor(
    score: torch.Tensor, pulse_ac: torch.Tensor
) -> torch.Tensor:
    positive_score = torch.relu(score)
    log_pulse = torch.log(torch.clamp(pulse_ac, min=1e-06))
    signed_score = torch.sign(score) * torch.log1p(torch.abs(score))
    coexistence = log_pulse - torch.log1p(positive_score)
    pulse_per_vertical = pulse_ac / (1.0 + positive_score)
    return torch.stack(
        [log_pulse, signed_score, coexistence, pulse_per_vertical], dim=1
    )


class DirectLinearConsistencyExpert(nn.Module):

    def __init__(self, center, scale, weight, bias):
        super().__init__()
        center_tensor = torch.as_tensor(center, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        if center_tensor.shape != (4,) or scale_tensor.shape != (4,):
            raise ValueError("Direct consistency scaler must have four features")
        if torch.any(scale_tensor <= 0):
            raise ValueError("Consistency scales must be positive")
        self.register_buffer("center", center_tensor)
        self.register_buffer("scale", scale_tensor)
        self.linear = nn.Linear(4, 1)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.as_tensor(weight, dtype=torch.float32).reshape(1, 4)
            )
            self.linear.bias.copy_(
                torch.as_tensor(bias, dtype=torch.float32).reshape(1)
            )

    def forward(self, score: torch.Tensor, pulse_ac: torch.Tensor) -> torch.Tensor:
        descriptor = base_consistency_descriptor(score, pulse_ac)
        standardized = (descriptor - self.center) / self.scale
        return self.linear(standardized).squeeze(1)


class NarrowbandNode(nn.Module):

    def __init__(
        self,
        profile_dim: int = 64,
        pulse_profile_dim: int = 64,
        concept_dim: int = 32,
        hidden_dim: int = 128,
        concept_hidden: int = 128,
        dropout: float = 0.2,
        neural_logit_bound: float = 8.0,
        consistency=None,
    ):
        super().__init__()
        self.profile_dim = int(profile_dim)
        self.pulse_profile_dim = int(pulse_profile_dim)
        self.concept_dim = int(concept_dim)
        self.hidden_dim = int(hidden_dim)
        self.concept_hidden = int(concept_hidden)
        self.dropout = float(dropout)
        self.neural_logit_bound = float(neural_logit_bound)
        self.stft_encoder, stft_dim = dtfe_backbone(1)
        self.profile_encoder = TemporalProfileEncoder(profile_dim)
        self.pulse_profile_encoder = TemporalProfileEncoder(pulse_profile_dim)
        self.direct_concept_encoder = nn.Sequential(
            nn.Linear(5, concept_dim), nn.SiLU()
        )
        fused_dim = stft_dim + profile_dim + pulse_profile_dim + concept_dim
        self.neural_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.concept_head = nn.Sequential(
            nn.Linear(stft_dim + profile_dim + pulse_profile_dim, concept_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(concept_hidden, 5),
        )
        self.raw_neural_reliability = nn.Parameter(
            torch.tensor(_logit_for_target_reliability(1.0), dtype=torch.float32)
        )
        consistency = consistency or dict(
            center=[0.0] * 4, scale=[1.0] * 4, weight=[0.0] * 4, bias=[0.0]
        )
        self.consistency_expert = DirectLinearConsistencyExpert(
            consistency["center"],
            consistency["scale"],
            consistency["weight"],
            consistency["bias"],
        )

    @property
    def neural_reliability(self) -> torch.Tensor:
        return 0.25 + 0.95 * torch.sigmoid(self.raw_neural_reliability)

    def forward(self, stft, profile, pulse_profile, raw_concepts, scaled_concepts):
        if scaled_concepts.shape[1] != 5:
            raise ValueError(
                f"Node2 expects five scaled concepts, got {tuple(scaled_concepts.shape)}"
            )
        stft_feature = self.stft_encoder(stft)
        profile_feature = self.profile_encoder(profile)
        pulse_feature = self.pulse_profile_encoder(pulse_profile)
        concept_feature = self.direct_concept_encoder(scaled_concepts)
        fused = torch.cat(
            [stft_feature, profile_feature, pulse_feature, concept_feature], dim=1
        )
        neural_logit_raw = self.neural_head(fused).squeeze(1)
        neural_logit = self.neural_logit_bound * torch.tanh(
            neural_logit_raw / self.neural_logit_bound
        )
        predicted_concepts = self.concept_head(
            torch.cat([stft_feature, profile_feature, pulse_feature], dim=1)
        )
        score = node2_physics_score(raw_concepts)
        physics_logit = torch.clamp(score / 0.5, min=-8.0, max=8.0)
        neural_probability = torch.sigmoid(neural_logit_raw)
        consistency_logit = self.consistency_expert(score, raw_concepts[:, 4])
        consistency_probability = torch.sigmoid(consistency_logit)
        probability = 1.0 - (1.0 - neural_probability) * (1.0 - consistency_probability)
        probability = probability.clamp(1e-06, 1.0 - 1e-06)
        return {
            "logit": torch.logit(probability),
            "probability": probability,
            "physics_score": score,
            "physics_logit": physics_logit,
            "neural_logit_raw": neural_logit_raw,
            "neural_logit": neural_logit,
            "neural_probability": neural_probability,
            "predicted_concepts": predicted_concepts,
            "neural_reliability": self.neural_reliability.expand_as(neural_logit_raw),
            "consistency_logit": consistency_logit,
            "consistency_probability": consistency_probability,
            "vertical_physics_score": score,
        }

    def model_config(self):
        return dict(
            profile_dim=self.profile_dim,
            pulse_profile_dim=self.pulse_profile_dim,
            concept_dim=self.concept_dim,
            hidden_dim=self.hidden_dim,
            concept_hidden=self.concept_hidden,
            dropout=self.dropout,
            neural_logit_bound=self.neural_logit_bound,
        )


class NonNarrowbandNode(nn.Module):

    def __init__(
        self,
        pc_profile_dim: int = 64,
        iq_profile_dim: int = 64,
        concept_dim: int = 32,
        hidden_dim: int = 128,
        concept_hidden: int = 128,
        dropout: float = 0.2,
        neural_logit_bound: float = 6.0,
    ):
        super().__init__()
        self.pc_profile_dim = int(pc_profile_dim)
        self.iq_profile_dim = int(iq_profile_dim)
        self.concept_dim = int(concept_dim)
        self.hidden_dim = int(hidden_dim)
        self.concept_hidden = int(concept_hidden)
        self.dropout = float(dropout)
        self.neural_logit_bound = float(neural_logit_bound)
        self.stft_encoder, stft_dim = dtfe_backbone(1)
        self.pc_encoder = ProfileEncoder(2, pc_profile_dim)
        self.iq_encoder = ProfileEncoder(3, iq_profile_dim)
        self.direct_concept_encoder = nn.Sequential(
            nn.Linear(3, concept_dim), nn.SiLU()
        )
        fused_dim = stft_dim + pc_profile_dim + iq_profile_dim + concept_dim
        self.neural_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.concept_head = nn.Sequential(
            nn.Linear(stft_dim + pc_profile_dim + iq_profile_dim, concept_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(concept_hidden, 3),
        )
        initial_probability = (0.25 - 0.1) / 0.4
        self.raw_neural_reliability = nn.Parameter(
            torch.tensor(_inverse_sigmoid(initial_probability), dtype=torch.float32)
        )

    @property
    def neural_reliability(self) -> torch.Tensor:
        return 0.1 + 0.4 * torch.sigmoid(self.raw_neural_reliability)

    def model_config(self) -> dict:
        return {
            "pc_profile_dim": self.pc_profile_dim,
            "iq_profile_dim": self.iq_profile_dim,
            "concept_dim": self.concept_dim,
            "hidden_dim": self.hidden_dim,
            "concept_hidden": self.concept_hidden,
            "dropout": self.dropout,
            "neural_logit_bound": self.neural_logit_bound,
        }

    def forward(self, stft, pc_profile, iq_profile, raw_concepts, scaled_concepts):
        if scaled_concepts.shape[1] != 3:
            raise ValueError(
                f"Node3 expects three scaled concepts, got {tuple(scaled_concepts.shape)}"
            )
        stft_feature = self.stft_encoder(stft)
        pc_feature = self.pc_encoder(pc_profile)
        iq_feature = self.iq_encoder(iq_profile)
        concept_feature = self.direct_concept_encoder(scaled_concepts)
        fused = torch.cat(
            [stft_feature, pc_feature, iq_feature, concept_feature], dim=1
        )
        neural_logit_raw = self.neural_head(fused).squeeze(1)
        neural_logit = self.neural_logit_bound * torch.tanh(
            neural_logit_raw / self.neural_logit_bound
        )
        predicted_concepts = self.concept_head(
            torch.cat([stft_feature, pc_feature, iq_feature], dim=1)
        )
        score = node3_physics_score(raw_concepts)
        physics_logit = torch.clamp(score / 0.5, min=-12.0, max=12.0)
        logit = physics_logit + self.neural_reliability * neural_logit
        return {
            "logit": logit,
            "physics_score": score,
            "physics_logit": physics_logit,
            "neural_logit_raw": neural_logit_raw,
            "neural_logit": neural_logit,
            "predicted_concepts": predicted_concepts,
            "neural_reliability": self.neural_reliability.expand_as(logit),
        }
