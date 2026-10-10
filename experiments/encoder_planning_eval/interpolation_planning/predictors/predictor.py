"""Predictors for filling in intermediate latents from (z1, zT) alone -- registry-driven
(same NAME_REGISTRY = {name: constructor} convention as
predictive_world_model/models/norms.py's NORM_REGISTRY), so a new variant just needs to
be added to PREDICTOR_REGISTRY to become selectable via train_predictor.py's
--predictor-type, with no other code changes.

Common interface every variant implements, regardless of internal design: `forward(z1,
zT, t_len) -> z_pred (B, T, D)`. Only positions [1, T-2] are ever used downstream (loss,
eval) -- whatever a variant puts at positions 0/T-1 is unused filler, kept only so every
variant returns the same shape.

Two variants:
  - "mlp" (PlanningMLP): a small feedforward net applied independently to each
    intermediate position, conditioned on (z1, zT, sinusoidal(t/T)). Cheap, no attention
    between the positions being filled in.
  - "transformer" (PlanningTransformer, default): a bidirectional TransformerEncoder over
    the whole length-T sequence with masked positions replaced by a shared learned
    mask_token -- mirrors external/chirality_in_action/LiFT/lift/model.py's
    MaskedLatentPredictor + MaskedJEPAWorldModel's masking convention (P-JEPA,
    arXiv:2606.23256, Eq. 2), specialized to this task's FIXED mask (always exactly
    positions [1, T-2], never LiFT's random ratio). Unlike the MLP, every filled-in
    position can attend to every other (including other intermediate positions), which is
    the point of calling it a "fills in the intermediate latents" model rather than T-2
    independent regressions.

Every variant is fixed-architecture across encoders under test: only d_model / the
boundary layer widths track the encoder's own feature_dim, never the depth/width/
attention-head hyperparameters (see ../README.md for why that fixedness is the point).
"""
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding -- self-contained copy of
    predictive_world_model/models/positional_encoding.py's class (same formula), kept
    here rather than imported since this analysis is meant to stand on its own (same
    convention losses/world_model_loss.py's own docstring describes for this repo)."""

    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SinusoidalStepEmbedding(nn.Module):
    """alpha in [0,1] -> a fixed-size sinusoidal embedding -- same construction as
    PositionalEncoding above, but for a continuous scalar rather than an integer sequence
    index, so the embedding width (and thus PlanningMLP's input width) is decoupled from
    clip length T."""

    def __init__(self, dim: int = 32):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        freqs = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.register_buffer("freqs", freqs)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: (...,) in [0,1] -> (..., dim)"""
        args = alpha.unsqueeze(-1) * self.freqs  # (..., dim/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PlanningMLP(nn.Module):
    """ẑ_t = MLP([z1 ; zT ; step_embed(t/T)]), applied independently (shared weights) to
    every intermediate position -- no attention between the T-2 positions being filled
    in, unlike PlanningTransformer below."""

    def __init__(self, dim: int, hidden: int = 512, depth: int = 2, embed_dim: int = 32,
                 dropout: float = 0.1, max_len: int = 16):
        super().__init__()
        self.dim = dim
        self.step_embed = SinusoidalStepEmbedding(embed_dim)
        layers = []
        in_dim = 2 * dim + embed_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor, zT: torch.Tensor, t_len: int) -> torch.Tensor:
        """z1, zT: (B, D). Returns z_pred: (B, T, D) -- positions 0/T-1 are filled with
        z1/zT verbatim (never used downstream, kept only for interface parity)."""
        b, d = z1.shape
        n_mid = t_len - 2
        alpha = torch.arange(1, t_len - 1, device=z1.device, dtype=z1.dtype) / (t_len - 1)  # (T-2,)
        embed = self.step_embed(alpha).unsqueeze(0).expand(b, n_mid, -1)  # (B, T-2, embed_dim)
        z1_rep = z1.unsqueeze(1).expand(b, n_mid, d)
        zT_rep = zT.unsqueeze(1).expand(b, n_mid, d)
        x = torch.cat([z1_rep, zT_rep, embed], dim=-1)
        mid_pred = self.net(x)  # (B, T-2, D)

        z_pred = z1.new_zeros(b, t_len, d)
        z_pred[:, 0] = z1
        z_pred[:, -1] = zT
        z_pred[:, 1:-1] = mid_pred
        return z_pred


class PlanningTransformer(nn.Module):
    """z_pred = Transformer([z1, mask_token, ..., mask_token, zT]): a length-T sequence
    with the start/goal latents in their true positions and every intermediate position
    replaced by one shared learned mask token, refined by a bidirectional (non-causal)
    TransformerEncoder into a filled-in prediction at every position -- see this module's
    docstring for the LiFT precedent this mirrors.

    Operates directly in the encoder's own latent_dim (d_model = feature_dim, no
    up/down-projection) -- same convention as predictive_world_model/models/predictor.py's
    CausalPredictor. norm_first=True needs an explicit final LayerNorm after the last
    block (pre-LN leaves the residual stream unnormalized otherwise) -- CausalPredictor's
    own docstring flags this exact omission as a real bug LiFT's LatentPredictor has and
    this repo fixed; included here from the start rather than repeating it.
    """

    def __init__(self, dim: int, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1,
                 max_len: int = 16, pos_encoding: str = "fixed"):
        """pos_encoding -- diagnostic knob (see ../README.md's investigation of why every
        trained predictor underperformed plain interpolation): "fixed" (default, original
        behavior) adds the sinusoidal PositionalEncoding at its own fixed ~0.71-per-dim
        RMS, unscaled -- fine when the encoder's own per-dim feature scale is comparable
        or larger, but drowns out the (std=0.02) mask_token's content signal at masked
        positions for encoders with smaller natural scale (DINOv3, LevlJEPA). "scaled"
        multiplies the same fixed sinusoidal table by a learned scalar, so gradient
        descent can shrink it to match whatever scale this encoder actually has.
        "learned" replaces the sinusoidal table entirely with a freely learned per-position
        embedding (initialized small, std=0.02, matching mask_token's own init) -- lets
        both its scale AND shape be learned rather than assuming sinusoids are even the
        right structure."""
        super().__init__()
        if pos_encoding not in ("fixed", "scaled", "learned"):
            raise ValueError(f"pos_encoding must be 'fixed', 'scaled' or 'learned', got {pos_encoding!r}")
        self.dim = dim
        self.max_len = max_len
        self.pos_encoding = pos_encoding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        if pos_encoding in ("fixed", "scaled"):
            self.pos_encoder = PositionalEncoding(dim, max_len=max_len)
            if pos_encoding == "scaled":
                self.pos_scale = nn.Parameter(torch.tensor(1.0))
        else:  # learned
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(dim),
        )

    def forward(self, z1: torch.Tensor, zT: torch.Tensor, t_len: int) -> torch.Tensor:
        """z1, zT: (B, D) -- start/goal latents. t_len: clip length T (python int, same
        for the whole batch). Returns z_pred: (B, T, D), filled in at every position
        (only positions [1, T-2] are meaningful -- position 0/T-1 are just z1/zT run
        through the transformer, never used downstream)."""
        b, d = z1.shape
        x = self.mask_token.expand(b, t_len, d).clone()
        x[:, 0] = z1
        x[:, -1] = zT
        if self.pos_encoding == "fixed":
            x = self.pos_encoder(x)
        elif self.pos_encoding == "scaled":
            x = x + self.pos_scale * self.pos_encoder.pe[:, :t_len]
        else:  # learned
            x = x + self.pos_embed[:, :t_len]
        return self.transformer(x)  # no mask: bidirectional, every position sees z1 AND zT


# name -> (constructor, {arg: default}) -- the defaults double as the CLI flags
# train_predictor.py exposes for that predictor_type (see its ARCH_ARGS).
#
# transformer_scaled_pe / transformer_learned_pe: diagnostic variants for the
# positional-encoding scale-mismatch investigation (PlanningTransformer's own docstring/
# pos_encoding kwarg). Fixed via functools.partial rather than an extra dict key, since
# arch_kwargs is built from train_predictor.py's CLI flags by name (getattr(args, k) for
# k in this entry's own default-dict keys) -- pos_encoding isn't a CLI flag, so it must be
# baked into the constructor itself, not passed through arch_kwargs, for the
# checkpoint's saved arch_kwargs / eval-time reconstruction round-trip to keep working
# unchanged for these too.
PREDICTOR_REGISTRY = {
    "mlp": (PlanningMLP, dict(hidden=512, depth=2, embed_dim=32, dropout=0.1)),
    "transformer": (PlanningTransformer, dict(nhead=8, num_layers=2, dropout=0.1)),
    "transformer_scaled_pe": (partial(PlanningTransformer, pos_encoding="scaled"),
                              dict(nhead=8, num_layers=2, dropout=0.1)),
    "transformer_learned_pe": (partial(PlanningTransformer, pos_encoding="learned"),
                               dict(nhead=8, num_layers=2, dropout=0.1)),
}


def build_predictor(predictor_type: str, dim: int, max_len: int = 16, **kwargs) -> nn.Module:
    if predictor_type not in PREDICTOR_REGISTRY:
        raise ValueError(f"predictor_type must be one of {sorted(PREDICTOR_REGISTRY)}, "
                          f"got {predictor_type!r}")
    cls, defaults = PREDICTOR_REGISTRY[predictor_type]
    args = {**defaults, **kwargs}
    return cls(dim=dim, max_len=max_len, **args)
