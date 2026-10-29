"""Decoupled Time-Frequency Encoder with a 512-dimensional output."""

from __future__ import annotations
import torch
import torch.nn as nn

DEFAULT_STEM_WIDTH = 32
DEFAULT_STAGE_WIDTHS = (48, 96, 192, 384)
DEFAULT_BLOCKS_PER_STAGE = (2, 2, 2, 2)
DEFAULT_TIME_DILATIONS = (1, 2)
DEFAULT_OUTPUT_DIM = 512
DTFE_PRESETS = {"base": {"stem_width": 32, "stage_widths": (48, 96, 192, 384)}}


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = int(max_groups)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _scaled_width(width: int, multiplier: float, divisor: int = 8) -> int:
    value = int(round(width * float(multiplier) / divisor)) * divisor
    return max(divisor, value)


class AnisotropicStem(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = DEFAULT_STEM_WIDTH,
        kernel: int = 7,
        anisotropic: bool = True,
    ):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError(f"stem width must be even, got {out_channels}")
        half = out_channels // 2
        padding = kernel // 2
        self.anisotropic = bool(anisotropic)
        if not self.anisotropic:
            self.square_path = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                bias=False,
            )
        else:
            self.frequency_path = nn.Conv2d(
                in_channels,
                half,
                kernel_size=(kernel, 1),
                stride=(2, 2),
                padding=(padding, 0),
                bias=False,
            )
            self.time_path = nn.Conv2d(
                in_channels,
                half,
                kernel_size=(1, kernel),
                stride=(2, 2),
                padding=(0, padding),
                bias=False,
            )
        self.norm = _group_norm(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.anisotropic:
            merged = torch.cat(
                [self.frequency_path(value), self.time_path(value)], dim=1
            )
        else:
            merged = self.square_path(value)
        return self.pool(self.activation(self.norm(merged)))


class AxisSeparableBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        time_dilation: int = 1,
        frequency_kernel: int = 5,
        time_kernel: int = 5,
    ):
        super().__init__()
        self.frequency_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(frequency_kernel, 1),
            stride=(stride, 1),
            padding=(frequency_kernel // 2, 0),
            groups=in_channels,
            bias=False,
        )
        self.frequency_norm = _group_norm(in_channels)
        self.time_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, time_kernel),
            stride=(1, stride),
            padding=(0, time_dilation * (time_kernel // 2)),
            dilation=(1, time_dilation),
            groups=in_channels,
            bias=False,
        )
        self.time_norm = _group_norm(in_channels)
        self.mixer = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.mixer_norm = _group_norm(out_channels)
        self.activation = nn.SiLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(value)
        out = self.activation(self.frequency_norm(self.frequency_conv(value)))
        out = self.activation(self.time_norm(self.time_conv(out)))
        out = self.mixer_norm(self.mixer(out))
        return self.activation(out + identity)


class DualAxisStatisticsPool(nn.Module):
    STATISTICS = 4

    def __init__(self, dual_axis: bool = True):
        super().__init__()
        self.dual_axis = bool(dual_axis)
        self.statistics = self.STATISTICS if self.dual_axis else 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"expected (N, C, F, T), got {tuple(value.shape)}")
        statistics_value = value.float()
        mean = statistics_value.mean(dim=(2, 3))
        if not self.dual_axis:
            return mean
        frequency_marginal = statistics_value.mean(dim=3)
        time_marginal = statistics_value.mean(dim=2)
        frequency_peak = frequency_marginal.amax(dim=2)
        time_peak = time_marginal.amax(dim=2)
        time_spread = time_marginal.std(dim=2, unbiased=False)
        return torch.cat([mean, frequency_peak, time_peak, time_spread], dim=1)


class DTFEEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        stem_width: int = DEFAULT_STEM_WIDTH,
        stage_widths=DEFAULT_STAGE_WIDTHS,
        blocks_per_stage=DEFAULT_BLOCKS_PER_STAGE,
        time_dilations=DEFAULT_TIME_DILATIONS,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        width_multiplier: float = 1.0,
        dropout: float = 0.0,
        anisotropic_stem: bool = True,
        dual_axis_pool: bool = True,
    ):
        super().__init__()
        stage_widths = tuple((int(width) for width in stage_widths))
        blocks_per_stage = tuple((int(count) for count in blocks_per_stage))
        time_dilations = tuple((int(value) for value in time_dilations))
        if len(stage_widths) != len(blocks_per_stage):
            raise ValueError("stage_widths and blocks_per_stage disagree")
        if not time_dilations or any((value < 1 for value in time_dilations)):
            raise ValueError(f"invalid time dilations: {time_dilations}")
        if width_multiplier != 1.0:
            stem_width = _scaled_width(stem_width, width_multiplier, 2)
            stage_widths = tuple(
                (_scaled_width(width, width_multiplier) for width in stage_widths)
            )
        self.in_channels = int(in_channels)
        self.stem_width = int(stem_width)
        self.stage_widths = stage_widths
        self.blocks_per_stage = blocks_per_stage
        self.time_dilations = time_dilations
        self.output_dim = int(output_dim)
        self.width_multiplier = float(width_multiplier)
        self.dropout = float(dropout)
        self.anisotropic_stem = bool(anisotropic_stem)
        self.dual_axis_pool = bool(dual_axis_pool)
        self.stem = AnisotropicStem(
            self.in_channels, self.stem_width, anisotropic=self.anisotropic_stem
        )
        stages = []
        current = self.stem_width
        for stage_index, (width, blocks) in enumerate(
            zip(stage_widths, blocks_per_stage)
        ):
            for block_index in range(blocks):
                stages.append(
                    AxisSeparableBlock(
                        current,
                        width,
                        stride=2 if stage_index > 0 and block_index == 0 else 1,
                        time_dilation=time_dilations[block_index % len(time_dilations)],
                    )
                )
                current = width
        self.stages = nn.Sequential(*stages)
        self.pool = DualAxisStatisticsPool(dual_axis=self.dual_axis_pool)
        readout_dim = current * self.pool.statistics
        self.projection = nn.Sequential(
            nn.Linear(readout_dim, self.output_dim, bias=False),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )
        self.feature_dim = self.output_dim
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encoder_config(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "stem_width": self.stem_width,
            "stage_widths": list(self.stage_widths),
            "blocks_per_stage": list(self.blocks_per_stage),
            "time_dilations": list(self.time_dilations),
            "output_dim": self.output_dim,
            "width_multiplier": self.width_multiplier,
            "dropout": self.dropout,
            "anisotropic_stem": self.anisotropic_stem,
            "dual_axis_pool": self.dual_axis_pool,
        }

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        if stft.ndim != 4 or stft.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (N, {self.in_channels}, F, T), got {tuple(stft.shape)}"
            )
        value = self.stem(stft)
        value = self.stages(value)
        return self.projection(self.pool(value))


def dtfe_backbone(
    in_channels: int = 1, preset: str = "base", width_multiplier: float = 1.0, **kwargs
) -> tuple[nn.Module, int]:
    if preset not in DTFE_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}, expected one of {sorted(DTFE_PRESETS)}"
        )
    settings = {**DTFE_PRESETS[preset], **kwargs}
    encoder = DTFEEncoder(
        in_channels=in_channels, width_multiplier=width_multiplier, **settings
    )
    return (encoder, encoder.feature_dim)


def parameter_count(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters()))
