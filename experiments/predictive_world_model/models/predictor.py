"""Causal transformer predicting each frame's next-step latent from the latents up to
and including that frame. Same architecture as LiFT's LatentPredictor (self-contained
copy, not an import, since this project generalizes beyond LiFT).

Used for token_mode='cls' only (this project's original, still-in-use K400 configs and
checkpoints, e.g. configs/cls_mlp_k400.yaml / checkpoints/pwm_cls_mlp_k400* — kept
unchanged rather than swapped for models/block_causal_predictor.py's reference-exact
predictor, so those stay loadable/reproducible). token_mode='patch' (both
projector_type='global' and 'channel') uses BlockCausalPredictor instead — see its
module docstring for why patch mode specifically needed the reference's exact predictor
internals (independent dim_head, learned positional embedding).

No actions/proprioception, unlike external/temporal-straightening's VisualWorldModel —
procedural video has no agent whose interventions we'd condition on; prediction is
driven by observed dynamics alone.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_encoding import PositionalEncoding


class CausalPredictor(nn.Module):
    def __init__(self, latent_dim, nhead=8, num_layers=2, dropout=0.1, max_len=16):
        super().__init__()
        self.pos_encoder = PositionalEncoding(latent_dim, max_len=max_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # norm_first=True (pre-LN) requires a final LayerNorm after the last block —
        # without it the residual stream leaves the stack unnormalized and can grow
        # across layers. The reference has this (temporal-straightening models/vit.py's
        # Transformer: self.norm = nn.LayerNorm(dim), applied on the way out); LiFT's
        # LatentPredictor omits it, and we inherited the omission until now.
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(latent_dim),
        )

    def forward(self, z):
        """
        Args:
            z: [B, T, latent_dim] — encoder's per-frame latents
        Returns:
            z_pred: [B, T, latent_dim] — z_pred[:, t] predicts z[:, t+1] (causally
                masked: position t only attends to positions <= t)
        """
        t = z.size(1)
        # The explicit mask is what actually enforces causality. is_causal= is only a
        # hint: passing is_causal=True WITHOUT this mask silently yields fully dense
        # (non-causal) attention, with no error — verified by dependency-matrix test in
        # tests/test_model.py, which exists to catch exactly that regression.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(z.device)
        x = self.pos_encoder(z)
        return self.transformer(x, mask=causal_mask, is_causal=True)
