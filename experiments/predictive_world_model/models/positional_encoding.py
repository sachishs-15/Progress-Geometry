import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, interpolated if a sequence longer than
    max_len is ever passed in (same convention as LiFT's lift/model.py)."""

    def __init__(self, d_model, max_len=16):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        t = x.size(1)
        if t == self.max_len:
            return x + self.pe
        if t > self.max_len:
            pe = F.interpolate(self.pe.transpose(1, 2), size=(t,), mode="linear", align_corners=False)
            return x + pe.transpose(1, 2)
        return x + self.pe[:, :t]
