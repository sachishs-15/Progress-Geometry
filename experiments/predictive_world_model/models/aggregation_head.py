"""Learnable aggregation head that maps a per-frame patch-token grid down to ONE
global vector, used ONLY for the curvature/VICReg regularizers on spatial (P>1)
latents -- ported from external/temporal-straightening/models/dino.py's
DinoV2Encoder.agg(agg_type="mlp"): "an MLP with an output dimension of 128" (paper
Section B.6), used in every main experiment. Exact reference shape (conf/encoder/
dino_channel.yaml: agg_mlp_hidden_dim=512, agg_out_dim=128):
    Linear(P*in_dim, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 128)
    -> LayerNorm(128)

Mean-pooling (this project's own behavior before this file existed) is the reference's
separately-ablated "mean" agg_type, shown to underperform "mlp" in every environment
tested (paper Figure 14).

The prediction loss itself is untouched by this: it still operates on the predictor's
native per-patch output (see world_model_loss.py's prediction_loss), matching the
reference's own split between prediction loss (on spatial features) and curvature loss
(on aggregated features).
"""
import torch.nn as nn


class AggregationHead(nn.Module):
    def __init__(self, num_patches, in_dim, hidden=None, out_dim=128):
        super().__init__()
        # Same fallback as the reference's DinoV2Encoder (agg_mlp_hidden_dim unset ->
        # 4 * agg_out_dim), which equals 512 at the default out_dim=128.
        hidden = hidden if hidden is not None else 4 * out_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(num_patches * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.post_norm = nn.LayerNorm(out_dim)

    def forward(self, z):
        """z: (B, T, P, in_dim) -> (B, T, out_dim)."""
        b, t, p, d = z.shape
        x = self.net(z.reshape(b, t, p * d))
        return self.post_norm(x)
