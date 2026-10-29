"""Validated arrays for the PCG-NDT model inputs."""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
import torch


@dataclass
class PreparedInputs:
    stft: np.ndarray
    node1_concepts: np.ndarray
    node1_physics_probability: np.ndarray
    node2_temporal_profile: np.ndarray
    node2_pulse_profile: np.ndarray
    node2_concepts: np.ndarray
    node2_physics_probability: np.ndarray
    node3_pc_profile: np.ndarray
    node3_iq_profile: np.ndarray
    node3_concepts: np.ndarray
    node3_physics_probability: np.ndarray
    SCHEMA = "pcg-ndt-prepared"
    VERSION = 1
    _SHAPES = {
        "stft": (1, 256, 368),
        "node1_concepts": (6,),
        "node1_physics_probability": (),
        "node2_temporal_profile": (2, 368),
        "node2_pulse_profile": (2, 512),
        "node2_concepts": (5,),
        "node2_physics_probability": (),
        "node3_pc_profile": (2, 512),
        "node3_iq_profile": (3, 512),
        "node3_concepts": (3,),
        "node3_physics_probability": (),
    }

    def __post_init__(self) -> None:
        counts = set()
        for item in fields(self):
            value = np.asarray(getattr(self, item.name))
            expected = self._SHAPES[item.name]
            if value.ndim < 1 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"{item.name}: expected (N,{','.join(map(str, expected))}), got {value.shape}"
                )
            if value.dtype != np.float32:
                raise TypeError(f"{item.name}: expected float32, got {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{item.name}: contains NaN or Inf")
            counts.add(len(value))
        if len(counts) != 1:
            raise ValueError(f"inconsistent prepared lengths: {counts}")

    def __len__(self) -> int:
        return len(self.stft)

    def tensor_batch(
        self, start: int, end: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        result = {}
        for item in fields(self):
            value = np.ascontiguousarray(
                np.asarray(getattr(self, item.name)[start:end])
            )
            result[item.name] = torch.from_numpy(value).to(device, non_blocking=True)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            __schema__=np.asarray(self.SCHEMA),
            __version__=np.asarray(self.VERSION, dtype=np.int64),
            **{item.name: getattr(self, item.name) for item in fields(self)},
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreparedInputs":
        with np.load(path, allow_pickle=False) as source:
            schema = str(np.asarray(source["__schema__"]).item())
            version = int(np.asarray(source["__version__"]).item())
            if schema != cls.SCHEMA or version != cls.VERSION:
                raise ValueError(f"incompatible prepared archive: {schema} v{version}")
            missing = [
                item.name for item in fields(cls) if item.name not in source.files
            ]
            if missing:
                raise KeyError(f"prepared archive is missing {missing}")
            values = {
                item.name: np.asarray(source[item.name], dtype=np.float32)
                for item in fields(cls)
            }
        return cls(**values)
