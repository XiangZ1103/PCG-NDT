"""Root node: base decision, horizontal weak-ridge evidence, and a confidence-aware one-sided correction."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dtfe import dtfe_backbone

RIDGE_LENGTHS = (9, 17, 33, 65)
RIDGE_SLOPES = (0,)


def deterministic_order_statistic_values(
    x: torch.Tensor, k: int, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    normalized_dim = dim if dim >= 0 else x.ndim + dim
    if not 0 <= normalized_dim < x.ndim:
        raise IndexError(f"dim={dim} is invalid for shape={tuple(x.shape)}")
    size = int(x.shape[normalized_dim])
    if not 1 <= int(k) <= size:
        raise ValueError(f"k must be in [1, {size}], got {k}")
    sorted_values = torch.sort(x, dim=normalized_dim).values
    selected = sorted_values.narrow(normalized_dim, int(k) - 1, 1)
    return selected if keepdim else selected.squeeze(normalized_dim)


def deterministic_median_values(
    x: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    normalized_dim = dim if dim >= 0 else x.ndim + dim
    size = int(x.shape[normalized_dim])
    return deterministic_order_statistic_values(
        x, k=(size + 1) // 2, dim=normalized_dim, keepdim=keepdim
    )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ResidualFeatureAdapter(nn.Module):

    def __init__(self, feature_dim: int = 512, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.up(self.activation(self.down(feature)))


def _ridge_templates(length: int, slopes=(0,)) -> torch.Tensor:
    height = 5
    templates = []
    window = torch.hann_window(length, periodic=False).clamp_min(0.05)
    window = window / window.sum()
    half = max((length - 1) / 2.0, 1.0)
    for slope in slopes:
        kernel = torch.zeros(height, length)
        for t in range(length):
            offset = int(round(float(slope) * (t - half) / half))
            center = 2 + offset
            kernel[center, t] += window[t]
            for side in (center - 2, center + 2):
                if 0 <= side < height:
                    kernel[side, t] -= 0.5 * window[t]
        kernel -= kernel.mean()
        kernel /= kernel.abs().sum().clamp_min(1e-06)
        templates.append(kernel)
    return torch.stack(templates, dim=0).unsqueeze(1)


class HorizontalRidgeConv(nn.Module):

    def __init__(self, length: int, slopes=RIDGE_SLOPES, residual_scale: float = 0.1):
        super().__init__()
        self.slopes = tuple((int(value) for value in slopes))
        template = _ridge_templates(int(length), slopes=self.slopes)
        self.register_buffer("template", template)
        self.delta = nn.Parameter(torch.zeros_like(template))
        self.raw_gain = nn.Parameter(
            torch.full((template.shape[0],), _inverse_softplus(1.0))
        )
        self.length = int(length)
        self.residual_scale = float(residual_scale)

    def effective_kernel(self) -> torch.Tensor:
        kernel = self.template + self.residual_scale * torch.tanh(self.delta)
        kernel = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        gain = F.softplus(self.raw_gain).view(-1, 1, 1, 1)
        return gain * kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.effective_kernel(), padding=(2, self.length // 2))


class PersistentRidgeBranch(nn.Module):
    """Detect a frequency row that remains elevated across observation time."""

    def __init__(self, lengths=RIDGE_LENGTHS, embedding_dim=128):
        super().__init__()
        if tuple(lengths) != RIDGE_LENGTHS:
            raise ValueError(f"ridge lengths must be {RIDGE_LENGTHS}")
        self.freq_start, self.freq_end = (30, 226)
        self.banks = nn.ModuleList(
            [HorizontalRidgeConv(length, RIDGE_SLOPES) for length in lengths]
        )
        self.raw_positive_threshold = nn.Parameter(torch.tensor(-1.1))
        self.raw_burst_threshold = nn.Parameter(torch.tensor(0.0))
        self.raw_burst_temperature = nn.Parameter(torch.tensor(-0.5))
        self.raw_profile_mix = nn.Parameter(torch.tensor(1.0))
        self.raw_response_gain = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_attention_temperature = nn.Parameter(torch.tensor(-0.5))
        self.embedding = nn.Sequential(
            nn.Linear(8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.ridge_head = nn.Linear(embedding_dim, 1)
        self.embedding_dim = int(embedding_dim)
        self.embedding_input_dim = 8
        self.response_channels = 4

    @property
    def positive_threshold(self):
        return 0.1 + 1.4 * torch.sigmoid(self.raw_positive_threshold)

    @property
    def burst_threshold(self):
        return 1.0 + 4.0 * torch.sigmoid(self.raw_burst_threshold)

    @property
    def burst_temperature(self):
        return 0.2 + 1.3 * torch.sigmoid(self.raw_burst_temperature)

    @property
    def attention_temperature(self):
        return 0.2 + 1.8 * torch.sigmoid(self.raw_attention_temperature)

    @staticmethod
    def _frequency_normalize(profile: torch.Tensor) -> torch.Tensor:
        center = deterministic_median_values(profile, dim=-1, keepdim=True)
        scale = (profile - center).abs().mean(dim=-1, keepdim=True).clamp_min(0.0001)
        return torch.clamp((profile - center) / scale, min=-6.0, max=12.0)

    def forward(self, stft: torch.Tensor) -> dict[str, torch.Tensor]:
        x = stft[:, :, self.freq_start : self.freq_end, :]
        flat = x.flatten(2)
        center = deterministic_median_values(flat, dim=-1, keepdim=True).unsqueeze(-1)
        scale = (x - center).abs().mean(dim=(-2, -1), keepdim=True).clamp_min(0.0001)
        z = torch.clamp((x - center) / scale, min=-6.0, max=10.0)
        positive = F.relu(z - self.positive_threshold)
        column_energy = positive.mean(dim=2, keepdim=True)
        column_center = deterministic_median_values(column_energy, dim=-1, keepdim=True)
        column_scale = (
            (column_energy - column_center)
            .abs()
            .mean(dim=-1, keepdim=True)
            .clamp_min(0.0001)
        )
        vertical_z = torch.clamp(
            (column_energy - column_center) / column_scale, min=-6.0, max=20.0
        )
        keep = torch.sigmoid(
            (self.burst_threshold - vertical_z) / self.burst_temperature
        )
        masked = positive * keep
        responses = F.relu(torch.cat([bank(masked) for bank in self.banks], dim=1))
        row_response = responses.mean(dim=-1)
        response_profile = self._frequency_normalize(row_response.amax(dim=1))
        time_count = z.shape[-1]
        q25 = deterministic_order_statistic_values(
            z.squeeze(1), k=max(1, int(round(0.25 * time_count))), dim=-1
        )
        median = deterministic_median_values(z.squeeze(1), dim=-1)
        mix = torch.sigmoid(self.raw_profile_mix)
        persistent_profile = mix * median + (1.0 - mix) * q25
        profile_z = self._frequency_normalize(persistent_profile)
        row_score = profile_z + F.softplus(self.raw_response_gain) * response_profile
        attention = torch.softmax(row_score / self.attention_temperature, dim=-1)
        attention_pool = torch.sum(row_response * attention.unsqueeze(1), dim=-1)
        max_pool = row_response.amax(dim=-1)
        entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-08)), dim=-1)
        entropy = entropy / torch.log(
            torch.tensor(float(attention.shape[-1]), device=attention.device)
        )
        top_profile = torch.topk(profile_z, k=10, dim=-1).values
        top_response = torch.topk(response_profile, k=10, dim=-1).values
        stats = torch.stack(
            [
                profile_z.amax(dim=-1),
                top_profile[:, :3].mean(dim=-1),
                top_profile.mean(dim=-1),
                entropy,
                response_profile.amax(dim=-1),
                top_response.mean(dim=-1),
                vertical_z.amax(dim=(-2, -1, -3)),
                keep.mean(dim=(-2, -1, -3)),
                scale.flatten(1).squeeze(1),
            ],
            dim=1,
        )
        embedding_parts = [attention_pool, max_pool]
        embedding = self.embedding(torch.cat(embedding_parts, dim=1))
        ridge_head_input = embedding
        return {
            "embedding": embedding,
            "ridge_logit": self.ridge_head(ridge_head_input).squeeze(1),
            "ridge_stats": stats,
            "row_attention": attention,
            "profile_z": profile_z,
        }

    def deformation_penalty(self) -> torch.Tensor:
        return torch.stack([bank.delta.square().mean() for bank in self.banks]).mean()


class BaseDecisionBranch(nn.Module):

    def __init__(self, adapter_bottleneck: int = 32):
        super().__init__()
        self.encoder, feature_dim = dtfe_backbone()
        self.feature_adapter = ResidualFeatureAdapter(feature_dim, adapter_bottleneck)
        self.neural_head = nn.Linear(feature_dim, 1)
        self.raw_physics_reliability = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )
        self.raw_neural_reliability = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.fusion_bias = nn.Parameter(torch.zeros(()))
        self.raw_temperature = nn.Parameter(torch.tensor(-0.6931472))

    @property
    def physics_reliability(self) -> torch.Tensor:
        return F.softplus(self.raw_physics_reliability) + 0.0001

    @property
    def neural_reliability(self) -> torch.Tensor:
        return F.softplus(self.raw_neural_reliability) + 0.0001

    @property
    def temperature(self) -> torch.Tensor:
        return 0.5 + 1.5 * torch.sigmoid(self.raw_temperature)

    def forward(self, stft: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        latent = self.feature_adapter(self.encoder(stft))
        p_neural = torch.sigmoid(self.neural_head(latent).squeeze(1)).clamp(
            1e-05, 1.0 - 1e-05
        )
        p_physics = physics.float().clamp(1e-05, 1.0 - 1e-05)
        survival = torch.pow(1.0 - p_physics, self.physics_reliability) * torch.pow(
            1.0 - p_neural, self.neural_reliability
        )
        probability = (1.0 - survival).clamp(1e-05, 1.0 - 1e-05)
        logit = (torch.logit(probability) + self.fusion_bias) / self.temperature
        return torch.sigmoid(logit).clamp(1e-05, 1.0 - 1e-05)


class WeakRidgeBranch(nn.Module):

    def __init__(self, ridge_dim=128, dropout=0.2, modality_dropout=0.15):
        super().__init__()
        self.encoder, feature_dim = dtfe_backbone()
        self.feature_adapter = ResidualFeatureAdapter(feature_dim, bottleneck=32)
        self.ridge = PersistentRidgeBranch(embedding_dim=ridge_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(ridge_dim + 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.feature_gate[-1].bias, -0.5)
        self.ridge_projection = nn.Linear(ridge_dim, feature_dim)
        nn.init.zeros_(self.ridge_projection.weight)
        nn.init.zeros_(self.ridge_projection.bias)
        self.neural_head = nn.Linear(feature_dim, 1)
        self.raw_neural_reliability = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.ridge_dim = int(ridge_dim)
        self.ridge_context_dim = int(ridge_dim)
        self.dropout = float(dropout)
        self.modality_dropout = float(modality_dropout)

    def initialize_from_base(self, base):
        self.encoder.load_state_dict(base.encoder.state_dict(), strict=True)
        self.feature_adapter.load_state_dict(
            base.feature_adapter.state_dict(), strict=True
        )
        self.neural_head.load_state_dict(base.neural_head.state_dict(), strict=True)

    @property
    def neural_reliability(self):
        return F.softplus(self.raw_neural_reliability) + 0.0001

    def forward(
        self, stft: torch.Tensor, physics_probability: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        base_feature = self.feature_adapter(self.encoder(stft))
        ridge = self.ridge(stft)
        p_phys = physics_probability.float().clamp(1e-05, 1.0 - 1e-05)
        physics_context = torch.stack([p_phys, 4.0 * p_phys * (1.0 - p_phys)], dim=1)
        ridge_context = ridge["embedding"]
        gate = 0.1 + 0.9 * torch.sigmoid(
            self.feature_gate(torch.cat([ridge_context, physics_context], dim=1))
        )
        if self.training and self.modality_dropout > 0:
            keep = (
                torch.rand(base_feature.shape[0], 1, device=base_feature.device)
                >= self.modality_dropout
            )
            base_feature = base_feature * keep / (1.0 - self.modality_dropout)
        fused_feature = base_feature + gate * self.ridge_projection(ridge_context)
        neural_logit = self.neural_head(fused_feature).squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            p_phys = p_phys.float().clamp(1e-05, 1.0 - 1e-05)
            p_neural = torch.sigmoid(neural_logit.float()).clamp(1e-05, 1.0 - 1e-05)
            survival = (1.0 - p_phys) * torch.pow(
                1.0 - p_neural, self.neural_reliability.float()
            )
            p_final = (1.0 - survival).clamp(1e-05, 1.0 - 1e-05)
            final_logit = torch.logit(p_final).clamp(-12.0, 12.0)
        return {
            "logit": final_logit,
            "probability": p_final,
            "neural_logit": neural_logit,
            "neural_probability": p_neural,
            "ridge_logit": ridge["ridge_logit"],
            "ridge_probability": torch.sigmoid(ridge["ridge_logit"].float()),
            "physics_probability": p_phys,
            "feature_gate": gate.squeeze(1),
            "neural_reliability": self.neural_reliability.expand_as(p_final),
            "ridge_stats": ridge["ridge_stats"],
            "row_attention": ridge["row_attention"],
        }

    def deformation_penalty(self) -> torch.Tensor:
        return self.ridge.deformation_penalty()


class RootNode(nn.Module):
    CONTEXT_DIM = 10

    def __init__(self, adapter_bottleneck=32, ridge_dim=128):
        super().__init__()
        self.base_expert = BaseDecisionBranch(adapter_bottleneck)
        self.dp_expert = WeakRidgeBranch(ridge_dim=ridge_dim)
        self.threshold_net = nn.Sequential(
            nn.LayerNorm(10), nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.threshold_net[-1].weight)
        nn.init.constant_(self.threshold_net[-1].bias, 1.5)
        self.raw_gate_temperature = nn.Parameter(torch.tensor(-0.5))
        self.raw_ridge_exponent = nn.Parameter(torch.tensor(-0.6931472))
        self.adapter_bottleneck, self.ridge_dim = (
            int(adapter_bottleneck),
            int(ridge_dim),
        )

    def model_config(self):
        return dict(
            adapter_bottleneck=self.adapter_bottleneck, ridge_dim=self.ridge_dim
        )

    def forward(
        self, stft: torch.Tensor, physics: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base_probability = (
                self.base_expert(stft, physics).float().clamp(1e-05, 1.0 - 1e-05)
            )
            dp = self.dp_expert(stft, physics)
            dp_probability = dp["probability"].float().clamp(1e-05, 1.0 - 1e-05)
            base_logit = torch.logit(base_probability).clamp(-10.0, 10.0)
            dp_logit = torch.logit(dp_probability).clamp(-10.0, 10.0)
            physics_probability = physics.float().clamp(1e-05, 1.0 - 1e-05)
            physics_logit = torch.logit(physics_probability).clamp(-10.0, 10.0)
            rescue_delta = F.relu(dp_logit - base_logit).clamp(max=12.0)
            ridge_probability = torch.sigmoid(dp["ridge_logit"].float()).clamp(
                1e-05, 1.0
            )
            rescue_evidence = rescue_delta * torch.pow(
                ridge_probability, self.ridge_exponent.detach()
            )
            context = torch.cat(
                [
                    base_logit[:, None],
                    (4.0 * base_probability * (1.0 - base_probability))[:, None],
                    dp_logit[:, None],
                    (4.0 * dp_probability * (1.0 - dp_probability))[:, None],
                    rescue_delta[:, None],
                    rescue_evidence[:, None],
                    physics_logit[:, None],
                    (4.0 * physics_probability * (1.0 - physics_probability))[:, None],
                    dp["ridge_logit"].float().clamp(-10.0, 10.0)[:, None],
                    dp["feature_gate"].float()[:, None],
                ],
                dim=1,
            )
        if context.shape[1] != self.CONTEXT_DIM:
            raise RuntimeError(
                f"confidence context must be {self.CONTEXT_DIM}-D, got {context.shape[1]}"
            )
        raw_threshold = self.threshold_net(context).squeeze(1)
        threshold = 0.05 + F.softplus(raw_threshold)
        rescue_evidence = rescue_delta * torch.pow(
            ridge_probability, self.ridge_exponent
        )
        gate_logit = (rescue_evidence - threshold) / self.gate_temperature
        gate = torch.sigmoid(gate_logit)
        final_logit = (base_logit + gate * rescue_evidence).clamp(-12.0, 12.0)
        return {
            "logit": final_logit,
            "probability": torch.sigmoid(final_logit),
            "base_probability": base_probability,
            "dp_probability": dp_probability,
            "physics_probability": physics_probability,
            "ridge_probability": ridge_probability,
            "rescue_delta": rescue_delta,
            "rescue_evidence": rescue_evidence,
            "gate": gate,
            "gate_logit": gate_logit,
            "threshold": threshold,
            "gate_temperature": self.gate_temperature.expand_as(gate),
            "ridge_exponent": self.ridge_exponent.expand_as(gate),
        }

    @property
    def gate_temperature(self) -> torch.Tensor:
        return 0.15 + 0.85 * torch.sigmoid(self.raw_gate_temperature)

    @property
    def ridge_exponent(self) -> torch.Tensor:
        return 0.5 + 1.5 * torch.sigmoid(self.raw_ridge_exponent)
