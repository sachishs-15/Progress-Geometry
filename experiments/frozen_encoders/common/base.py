"""Common interface every frozen-encoder plugin (models/<name>/model.py) implements."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class EncodingResult:
    embeddings: np.ndarray
    # Index into the *input* `frames` list each embedding row corresponds to.
    # 1:1 (0..T-1) for per-frame image encoders; may be shorter/grouped for
    # encoders that consume clips (e.g. one embedding per temporal tubelet).
    frame_indices: List[int]
    # (len(embeddings), grid_h, grid_w) or None if the model has no exposed
    # token-level attention.
    attention_maps: Optional[np.ndarray] = None


class FrozenEncoder(ABC):
    name: str = "base"
    supports_attention: bool = False
    attention_label: str = "attention"
    pooling_choices: List[str] = ["mean_patch"]
    # Whether get_patch_tokens() below is implemented for this model.
    supports_patch_tokens: bool = False

    def __init__(self, device: Optional[str] = None, pooling: str = "mean_patch", **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if pooling not in self.pooling_choices:
            raise ValueError(f"{self.name} supports pooling in {self.pooling_choices}, got {pooling!r}")
        self.pooling = pooling
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Instantiate self.model (+ processor, if any) on self.device, eval mode."""

    @abstractmethod
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        """Run the model over `frames` and return one embedding per output step."""

    def get_patch_tokens(self, frames: List[Image.Image]):
        """Optional: return raw (unpooled) per-patch tokens, for visualizations
        that need patch-level detail rather than one pooled vector per frame.

        Returns:
            patch_tokens: (T, P, D) tensor, T == len(frames), CLS/register
                tokens excluded.
            grid_hw: (grid_h, grid_w) spatial layout of the P patches.
        """
        raise NotImplementedError(f"{self.name} does not expose raw patch tokens")

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> FrozenEncoder:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
