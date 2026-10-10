"""token_mode='patch' encoder: wraps GlobalProjector/ChannelProjector (projectors.py)
around raw DINOv2 patch tokens. Unlike ClsMlpEncoder, both projectors already end in
their own normalization (see projectors.py docstrings), so nothing extra is needed here
for the K400-instability-style ||z|| drift problem cls-mode had.
"""
import torch.nn as nn

from .projectors import ChannelProjector, GlobalProjector


class PatchProjectorEncoder(nn.Module):
    def __init__(self, feature_dim=384, projector_type="global", latent_dim=384,
                 channel_out_dim=8, channel_hidden=192):
        super().__init__()
        self.projector_type = projector_type
        if projector_type == "global":
            self.projector = GlobalProjector(in_dim=feature_dim, out_dim=latent_dim, hidden=latent_dim, pool_hw=1)
            self.tokens_per_frame = 1
            self.out_dim = latent_dim
        elif projector_type == "channel":
            self.projector = ChannelProjector(in_dim=feature_dim, out_dim=channel_out_dim, hidden=channel_hidden)
            self.tokens_per_frame = None  # set on first forward, once the patch grid size is known
            self.out_dim = channel_out_dim
        else:
            raise ValueError(f"projector_type must be 'global' or 'channel', got {projector_type!r}")

    def forward(self, x):
        """x: (B, T, N, feature_dim) raw patch tokens, N a perfect square.
        Returns (B, T, P, out_dim): P=1 for global, P=N for channel."""
        b, t, n, c = x.shape
        h = w = int(round(n ** 0.5))
        if h * w != n:
            raise ValueError(f"Expected a square patch grid, got N={n}")
        x = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)
        out = self.projector(x)  # global: (B*T, out_dim); channel: (B*T, N, out_dim)
        if out.dim() == 2:
            out = out.unsqueeze(1)  # (B*T, 1, out_dim)
        p = out.shape[1]
        self.tokens_per_frame = p
        return out.reshape(b, t, p, out.shape[-1])
