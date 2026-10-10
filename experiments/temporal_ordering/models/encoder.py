"""Shared per-timestep encoder: projects the raw (large, e.g. 3200-D CrossTask) feature
into a smaller latent space. The same encoder instance is applied independently to every
timestep of a video (section 7 of the spec) — it has no notion of temporal order itself.
"""
from pathlib import Path

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(input_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))

    def save(self, path: str) -> None:
        torch.save({"input_dim": self.input_dim, "latent_dim": self.latent_dim,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FeatureEncoder":
        ckpt = torch.load(Path(path), map_location=device)
        model = cls(ckpt["input_dim"], ckpt["latent_dim"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)
