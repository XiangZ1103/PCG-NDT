"""Physics-Concept-Guided Neural Decision Tree for radar jamming recognition."""

from __future__ import annotations
from pathlib import Path
from typing import Mapping, Any
import numpy as np
import torch
import torch.nn as nn
from .node1 import RootNode
from .nodes import NarrowbandNode, NonNarrowbandNode
from .prepared import PreparedInputs
from .features import (
    extract_node1,
    extract_node2,
    extract_node3,
    normalize_stft,
    quantize_profiles,
    NODE1_FEATURE_NAMES,
    NODE2_FEATURE_NAMES,
    NODE3_FEATURE_NAMES,
)

CLASS_NAMES = ("NNJ", "ISRJ", "DFTJ", "CJ")


class PCGNDT(nn.Module):

    def __init__(self, node1, node2, node3, scalers, metadata=None):
        super().__init__()
        self.node1, self.node2, self.node3 = (node1, node2, node3)
        self.metadata = dict(metadata or {})
        self._register_scaler("node1", scalers["node1"], 6)
        self._register_scaler("node2", scalers["node2"], 5)
        self._register_scaler("node3", scalers["node3"], 3)

    def _register_scaler(
        self, name: str, payload: Mapping[str, Any], size: int
    ) -> None:
        center = torch.as_tensor(payload["center"], dtype=torch.float32)
        scale = torch.as_tensor(payload["scale"], dtype=torch.float32)
        if center.shape != (size,) or scale.shape != (size,):
            raise ValueError(
                f"{name} scaler expected {(size,)}, got {tuple(center.shape)}/{tuple(scale.shape)}"
            )
        if torch.any(scale <= 0):
            raise ValueError(f"{name} scaler contains a non-positive scale")
        self.register_buffer(f"{name}_scaler_center", center)
        self.register_buffer(f"{name}_scaler_scale", scale)
        self.register_buffer(
            f"{name}_scaler_clip",
            torch.tensor(float(payload.get("clip", 8.0)), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _scale(self, name: str, values: torch.Tensor) -> torch.Tensor:
        center = getattr(self, f"{name}_scaler_center")
        scale = getattr(self, f"{name}_scaler_scale")
        clip = getattr(self, f"{name}_scaler_clip")
        standardized = (values.float() - center) / scale
        return torch.clamp(standardized, min=-clip, max=clip)

    def forward(self, batch):
        node2_raw = batch["node2_concepts"]
        node2_scaled = self._scale("node2", node2_raw)
        node3_raw = batch["node3_concepts"]
        node3_scaled = self._scale("node3", node3_raw)
        output1 = self.node1(batch["stft"], batch["node1_physics_probability"])
        output2 = self.node2(
            batch["stft"],
            batch["node2_temporal_profile"],
            batch["node2_pulse_profile"],
            node2_raw,
            node2_scaled,
        )
        output3 = self.node3(
            batch["stft"],
            batch["node3_pc_profile"],
            batch["node3_iq_profile"],
            node3_raw,
            node3_scaled,
        )
        p1 = output1["probability"].float()
        p2 = output2["probability"].float()
        p3 = torch.sigmoid(output3["logit"]).float()
        p1d, p2d, p3d = (p1.double(), p2.double(), p3.double())
        class_probability = torch.stack(
            [
                p1d * (1.0 - p2d),
                (1.0 - p1d) * p3d,
                (1.0 - p1d) * (1.0 - p3d),
                p1d * p2d,
            ],
            dim=1,
        ).float()
        prediction = torch.argmax(class_probability, dim=1)
        narrow = p1 >= 0.5
        hard_narrow = torch.where(p2 >= 0.5, 3, 0)
        hard_non_narrow = torch.where(p3 >= 0.5, 1, 2)
        hard_prediction = torch.where(narrow, hard_narrow, hard_non_narrow).long()
        if not torch.all(torch.isfinite(class_probability)):
            raise RuntimeError("non-finite four-leaf probability")
        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "class_probability": class_probability,
            "prediction": prediction,
            "hard_prediction": hard_prediction,
            "node1_physics_probability": output1["physics_probability"].float(),
            "node1_base_probability": output1["base_probability"].float(),
            "node1_dp_probability": output1["dp_probability"].float(),
            "node1_ridge_probability": output1["ridge_probability"].float(),
            "node1_rescue_evidence": output1["rescue_evidence"].float(),
            "node1_confidence_gate": output1["gate"].float(),
            "node1_adaptive_threshold": output1["threshold"].float(),
            "node2_physics_probability": batch["node2_physics_probability"].float(),
            "node2_neural_probability": output2["neural_probability"].float(),
            "node2_consistency_probability": output2["consistency_probability"].float(),
            "node3_physics_probability": batch["node3_physics_probability"].float(),
            "node3_neural_probability": torch.sigmoid(
                output3["neural_logit_raw"]
            ).float(),
        }

    def prepare_inputs(
        self,
        stft_db: np.ndarray,
        b_pulses: np.ndarray,
        c_pulses: np.ndarray,
        *,
        quantize_inputs: bool = True,
        progress_every: int = 100,
    ) -> PreparedInputs:
        stft_db = np.asarray(stft_db)
        b_pulses = np.asarray(b_pulses)
        c_pulses = np.asarray(c_pulses)
        if stft_db.ndim != 3 or stft_db.shape[1:] != (256, 368):
            raise ValueError(f"expected stft (N,256,368), got {stft_db.shape}")
        count = len(stft_db)
        if len(b_pulses) != count or len(c_pulses) != count:
            raise ValueError("stft, B_pulses and C_pulses are not aligned")
        arrays = {
            "stft": np.empty((count, 1, 256, 368), dtype=np.float32),
            "node1_concepts": np.empty((count, 6), dtype=np.float32),
            "node1_physics_probability": np.empty(count, dtype=np.float32),
            "node2_temporal_profile": np.empty((count, 2, 368), dtype=np.float32),
            "node2_pulse_profile": np.empty((count, 2, 512), dtype=np.float32),
            "node2_concepts": np.empty((count, 5), dtype=np.float32),
            "node2_physics_probability": np.empty(count, dtype=np.float32),
            "node3_pc_profile": np.empty((count, 2, 512), dtype=np.float32),
            "node3_iq_profile": np.empty((count, 3, 512), dtype=np.float32),
            "node3_concepts": np.empty((count, 3), dtype=np.float32),
            "node3_physics_probability": np.empty(count, dtype=np.float32),
        }
        quantize = (
            quantize_profiles
            if quantize_inputs
            else lambda value: np.asarray(value, dtype=np.float32)
        )
        for index in range(count):
            image = stft_db[index]
            b_pulse = b_pulses[index]
            c_pulse = c_pulses[index]
            node1, p1 = extract_node1(image)
            node2, temporal, pulse_profile, p2 = extract_node2(image, b_pulse)
            node3, pc_profile, iq_profile, p3 = extract_node3(c_pulse, b_pulse, image)
            arrays["stft"][index, 0] = quantize(normalize_stft(image))
            arrays["node1_concepts"][index] = node1
            arrays["node1_physics_probability"][index] = p1
            arrays["node2_temporal_profile"][index] = quantize(temporal)
            arrays["node2_pulse_profile"][index] = quantize(pulse_profile)
            arrays["node2_concepts"][index] = node2
            arrays["node2_physics_probability"][index] = p2
            arrays["node3_pc_profile"][index] = quantize(pc_profile)
            arrays["node3_iq_profile"][index] = quantize(iq_profile)
            arrays["node3_concepts"][index] = node3
            arrays["node3_physics_probability"][index] = p3
            if progress_every and (index + 1) % progress_every == 0:
                print(f"[physical concepts] {index + 1}/{count}", flush=True)
        return PreparedInputs(**arrays)


def identity_scalers():
    """Neutral scalers for untrained architecture inspection."""
    return {
        name: dict(center=[0.0] * count, scale=[1.0] * count, clip=8.0)
        for name, count in [("node1", 6), ("node2", 5), ("node3", 3)]
    }


def build_model(
    *,
    scalers=None,
    node1_config=None,
    node2_config=None,
    node2_consistency=None,
    node3_config=None,
):
    node1 = RootNode(**dict(node1_config or {}))
    node2 = NarrowbandNode(consistency=node2_consistency, **dict(node2_config or {}))
    node3 = NonNarrowbandNode(**dict(node3_config or {}))
    return PCGNDT(node1, node2, node3, scalers or identity_scalers())


def parameter_breakdown(model):
    counts = {
        name: sum(p.numel() for p in getattr(model, name).parameters())
        for name in ("node1", "node2", "node3")
    }
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


def architecture_summary(model):
    return dict(
        class_names=CLASS_NAMES,
        concept_names=dict(
            node1=NODE1_FEATURE_NAMES,
            node2=NODE2_FEATURE_NAMES,
            node3=NODE3_FEATURE_NAMES,
        ),
        node1=model.node1.model_config(),
        node2=model.node2.model_config(),
        node3=model.node3.model_config(),
        node2_fused_dim=model.node2.neural_head[0].in_features,
        node3_fused_dim=model.node3.neural_head[0].in_features,
        parameters=parameter_breakdown(model),
    )


def save_bundle(model, path):
    """Save parameters, node settings, and training-fitted concept scalers."""
    scalers = {
        name: dict(
            center=getattr(model, name + "_scaler_center").detach().cpu().tolist(),
            scale=getattr(model, name + "_scaler_scale").detach().cpu().tolist(),
            clip=float(getattr(model, name + "_scaler_clip").cpu()),
        )
        for name in ("node1", "node2", "node3")
    }
    expert = model.node2.consistency_expert
    consistency = dict(
        center=expert.center.cpu().tolist(),
        scale=expert.scale.cpu().tolist(),
        weight=expert.linear.weight.detach().cpu().reshape(-1).tolist(),
        bias=expert.linear.bias.detach().cpu().tolist(),
    )
    payload = dict(
        format="pcg-ndt",
        version=1,
        scalers=scalers,
        node1=model.node1.model_config(),
        node2=model.node2.model_config(),
        node2_consistency=consistency,
        node3=model.node3.model_config(),
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        metadata=model.metadata,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_bundle(path, *, device="cpu"):
    """Load a complete PCG-NDT bundle, checking every state tensor strictly."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "pcg-ndt" or payload.get("version") != 1:
        raise ValueError("Expected a PCG-NDT bundle with format pcg-ndt and version 1")
    model = build_model(
        scalers=payload["scalers"],
        node1_config=payload["node1"],
        node2_config=payload["node2"],
        node2_consistency=payload["node2_consistency"],
        node3_config=payload["node3"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.metadata = dict(payload.get("metadata", {}))
    return model.to(device).eval()
