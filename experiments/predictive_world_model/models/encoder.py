"""token_mode='cls' encoder: DINOv2 CLS features are already 1 token/frame, so no
projector is structurally required.

Note on the reference: external/temporal-straightening does NOT have a projector in
cls-token mode — models/dino.py only creates one under `feature_key ==
"x_norm_patchtokens"`, since its GlobalProjector/ChannelProjector are Conv2d stacks that
need the patch grid's [B, C, H, W] layout to convolve over. A CLS token has no spatial
extent, so that path is not expressible for it. This MLP is therefore our own addition,
with no counterpart in the reference, kept so the encoder has something trainable (with
use_projector=False the encoder is Identity, z is frozen DINOv2 output, and the
curvature/VICReg terms have zero gradient) and so latent_dim can differ from feature_dim.

Because it is our own addition, it also has to supply on its own the property the
reference gets for free everywhere: a normalized output. In cls mode the reference's z is
DINOv2's `x_norm_clstoken` (post-final-LayerNorm) passed through untouched, so its scale
is frozen; in patch mode both projectors end in a LayerNorm. Ours is the only encoder in
the family that can let ||z|| drift during training, and that drift is what made
vicreg_cov (which grows as ||z||^4) explode on the first K400 run. Hence norm_type — see
models/norms.py for what each option does and why plain LayerNorm's per-frame-independent
statistics can distort trajectory straightness in a way the alternatives don't.

patch_mode (channel/global CNN projectors) is intentionally NOT here yet — this is the
cls-only first slice; a patch encoder module will be added alongside a block-causal
predictor once this slice is validated end-to-end.
"""
import torch.nn as nn

from .norms import NORM_REGISTRY


class ClsMlpEncoder(nn.Module):
    def __init__(self, feature_dim=384, latent_dim=384, hidden_dim=None, use_projector=True,
                 norm_type="layernorm", residual=False):
        super().__init__()
        # residual=True: z = norm(x + projector(x)) instead of norm(projector(x)) -- the
        # projector then only has to learn a CORRECTION on top of the frozen DINOv2
        # feature rather than reconstruct useful structure from scratch, which changes
        # what the straightening/VICReg terms are actually shaping (raw DINOv2 + a
        # learned delta, vs. a fully re-learned representation). Requires
        # feature_dim == latent_dim, same constraint as use_projector=False, for the
        # same reason (x is added directly, so the shapes must already match).
        self.residual = residual
        if residual:
            if not use_projector:
                raise ValueError("residual=True has no effect when use_projector=False "
                                  "(there is no projector output to add x to)")
            if feature_dim != latent_dim:
                raise ValueError(f"residual=True requires feature_dim == latent_dim, "
                                  f"got {feature_dim} != {latent_dim}")
        if use_projector:
            hidden_dim = hidden_dim or latent_dim
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            if feature_dim != latent_dim:
                raise ValueError(f"use_projector=False requires feature_dim == latent_dim, "
                                  f"got {feature_dim} != {latent_dim}")
            self.projector = nn.Identity()
        if norm_type not in NORM_REGISTRY:
            raise ValueError(f"norm_type must be one of {sorted(NORM_REGISTRY)}, got {norm_type!r}")
        self.norm = NORM_REGISTRY[norm_type](latent_dim)

    def forward(self, x):
        """x: (B, T, feature_dim) -> (B, T, latent_dim)"""
        z = self.projector(x)
        if self.residual:
            z = z + x
        return self.norm(z)


class ResidualMlpEncoder(nn.Module):
    """Pre-norm residual MLP blocks: z = z + MLP(LayerNorm(z)), stacked `depth` times,
    each MLP expanding to `expansion` * latent_dim internally.

    Why this exists alongside ClsMlpEncoder: that one is a bare
    Linear -> GELU -> Linear with NO residual path and a hardcoded depth of 2, so
    anything it learns has to be re-derived from scratch rather than accumulated as
    corrections, and it cannot be deepened at all. This is the standard transformer-FFN
    block arrangement (pre-norm + residual), which is better-conditioned at depth and is
    the one part of "make the encoder proper" that does NOT require giving the encoder
    cross-frame context. Every block is still applied per frame independently, so
    z_t = f(x_t) exactly as before -- the encoder/predictor separation that makes z a
    rollable world-model state is preserved (see ProceduralWorldModel's docstring), and
    the metric-gaming that a temporal encoder invites is avoided.

    ClsMlpEncoder's `residual` flag is a DIFFERENT thing and the two compose: that one
    adds the raw frozen DINOv2 feature back once at the very end (z = norm(x + proj(x))),
    a single skip around the whole projector. `residual_blocks` here is a skip around
    every block, which is what makes depth trainable. Set `input_residual=True` to also
    get the outer skip, which requires feature_dim == latent_dim for the same
    shape reason ClsMlpEncoder does.

    depth=0 reduces to in_proj + norm (no blocks), which is a linear probe with a norm --
    a useful floor to compare the deeper variants against.
    """

    def __init__(self, feature_dim=384, latent_dim=384, depth=4, expansion=4,
                 norm_type="layernorm", dropout=0.0, input_residual=False):
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        self.input_residual = input_residual
        if input_residual and feature_dim != latent_dim:
            raise ValueError(f"input_residual=True requires feature_dim == latent_dim, "
                              f"got {feature_dim} != {latent_dim}")

        self.in_proj = (nn.Identity() if feature_dim == latent_dim
                        else nn.Linear(feature_dim, latent_dim))
        hidden = expansion * latent_dim
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, latent_dim),
            )
            for _ in range(depth)
        )
        if norm_type not in NORM_REGISTRY:
            raise ValueError(f"norm_type must be one of {sorted(NORM_REGISTRY)}, got {norm_type!r}")
        self.norm = NORM_REGISTRY[norm_type](latent_dim)

    def forward(self, x):
        """x: (B, T, feature_dim) -> (B, T, latent_dim). Applied per frame; no mixing
        across T anywhere in here."""
        z = self.in_proj(x)
        for block in self.blocks:
            z = z + block(z)
        if self.input_residual:
            z = z + x
        return self.norm(z)


# Same {name: constructor} convention as norms.py's NORM_REGISTRY, so a new cls-mode
# encoder becomes selectable via --encoder-type with no other code change. "mlp" is the
# default everywhere and is ClsMlpEncoder unchanged, so every existing config and
# checkpoint keeps loading and training exactly as before.
#
# Constructors take keyword args only and ignore what they don't use (world_model.py
# passes the union), which is the same convention train.py already uses for its
# cls-only vs. patch-only flags.
ENCODER_REGISTRY = {
    "mlp": lambda **kw: ClsMlpEncoder(
        feature_dim=kw["feature_dim"], latent_dim=kw["latent_dim"],
        hidden_dim=kw.get("hidden_dim"), use_projector=kw.get("use_projector", True),
        norm_type=kw.get("norm_type", "layernorm"), residual=kw.get("residual", False),
    ),
    "residual_mlp": lambda **kw: ResidualMlpEncoder(
        feature_dim=kw["feature_dim"], latent_dim=kw["latent_dim"],
        depth=kw.get("encoder_depth", 4), expansion=kw.get("encoder_expansion", 4),
        norm_type=kw.get("norm_type", "layernorm"), dropout=kw.get("encoder_dropout", 0.0),
        input_residual=kw.get("residual", False),
    ),
}
