"""The one probe architecture used to evaluate every representation (raw features, or a
frozen encoder's output) — identical for all three experimental conditions, per the
critical-evaluation-protocol requirement that only the representation-learning objective
changes across conditions. The trainable model is exactly `nn.Linear(D, 1)` as specified;
the fixed (non-trainable) input standardization is only there so raw ~3200-D unnormalized
features (unlike the encoder outputs, which already pass through a LayerNorm) don't blow up
plain SGD/Adam — standard practice for any frozen-feature linear probe, not a change to the
probe's capacity.
"""
import torch
import torch.nn as nn


class ProgressProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1)

    def set_normalization(self, mean, std, std_floor: float = 1e-2) -> None:
        """`std_floor` matters: a train-set dimension that's ~constant (e.g. a dead ReLU
        unit, always 0 across every training example) has std ~0; dividing by that on a
        held-out video where the same dimension isn't exactly 0 amplifies float noise by
        up to 1e6x and blows up the whole prediction. Flooring std (rather than the usual
        `std + eps`) avoids that the way scikit-learn's StandardScaler does for
        zero-variance columns — effectively just stops scaling a dimension with no train
        signal instead of turning it into a noise amplifier.
        """
        std = torch.as_tensor(std, dtype=self.std.dtype).clamp_min(std_floor)
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        self.std.copy_(std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = (z - self.mean) / self.std
        return self.linear(z).squeeze(-1)
