import torch.nn as nn

from .aggregation_head import AggregationHead
from .block_causal_predictor import BlockCausalPredictor
from .encoder import ENCODER_REGISTRY
from .patch_encoder import PatchProjectorEncoder
from .pixel_encoder import PIXEL_ENCODER_REGISTRY
from .predictor import CausalPredictor


class ProceduralWorldModel(nn.Module):
    """Encode -> predict next-step latents causally from history. Straightness is not
    architecturally forced (unlike LiFT's interpolation-based variants) — it's an
    emergent property (if it occurs) of latents being one-step predictable, same
    mechanism as external/temporal-straightening and LiFT's own PredictiveWorldModel,
    minus actions.

    token_mode='cls' (default, UNCHANGED from the original slice — this project's
    original K400 configs/checkpoints, e.g. configs/cls_mlp_k400.yaml,
    checkpoints/pwm_cls_mlp_k400*/ablation_*, depend on this exact architecture staying
    put): DINOv2 CLS features, already 1 token/frame, through ClsMlpEncoder ->
    predictor.py's CausalPredictor (nn.TransformerEncoderLayer + sinusoidal positional
    encoding). Controlled by nhead/pred_num_layers/dropout/max_len. z: (B, T, latent_dim).
    residual_projector=True: ClsMlpEncoder adds the raw feature back after its own
    projector (see encoder.py) -- the projector then only has to learn a correction on
    top of frozen DINOv2, not reconstruct structure from scratch. Requires
    feature_dim == latent_dim, cls mode + use_projector=True only.

    token_mode='raw': same as 'cls' downstream of the feature vector -- identical
    ENCODER_REGISTRY (encoder_type=mlp/residual_mlp) and CausalPredictor, no new knobs
    there -- except the feature_dim input now comes from `self.pixel_encoder`
    (pixel_encoder.py's PIXEL_ENCODER_REGISTRY, default "small_resnet": a CNN trained
    FROM SCRATCH, jointly with everything downstream of it) run over raw RGB frames,
    instead of a frozen pretrained DINOv2 forward pass. This is the only token_mode
    where the backbone itself has gradients; every other mode's "encoder" is strictly a
    small head on top of an already-frozen feature. x: (B, T, 3, H, W) raw pixels, uint8
    [0, 255] or already-float [0, 1] (self.pixel_encoder normalizes either way; see its
    own docstring), rather than (B, T, feature_dim) precomputed features.

    token_mode='patch' (new in this project — no prior configs/checkpoints depend on its
    internals, so it adopts the reference's exact predictor architecture directly rather
    than reusing cls mode's): raw DINOv2 patch tokens through PatchProjectorEncoder
    (projectors.py's GlobalProjector or ChannelProjector, ported from
    external/temporal-straightening) -> block_causal_predictor.py's BlockCausalPredictor
    for BOTH projector_type='global' (num_patches=1; block-causal over a 1-patch grid IS
    plain causal) and 'channel' (num_patches>1, 196 by default -- 14x14, matching the
    reference's Table 1 "14x14x..." dims exactly: it resizes DINOv2's input so the grid
    stays 14x14 regardless of source resolution, NOT the 16x16=256 grid a naive
    224px/patch14 resize would give. Fixed by image resolution/patch size, so it must be
    passed in up front rather than inferred from data: BlockCausalPredictor has to exist
    at __init__ time, before train.py builds the optimizer over model.parameters(), or
    its weights would silently never be optimized). z: (B, T, P, out_dim), P=1 for
    global, P=num_patches for channel.

    Patch-mode predictor hyperparameters (depth/heads/dim_head/mlp_dim/p_dropout/
    emb_dropout) match external/temporal-straightening/conf/predictor/vit.yaml exactly
    (depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, emb_dropout=0) -- the
    SAME predictor config the reference uses for every encoder mode, no per-mode
    override.

    num_frames is the patch-mode predictor's own sequence-length capacity (its
    positional embedding table size) -- pass window_size+1 to match the reference's
    actual training regime (a fixed short history window, num_hist=window_size /
    num_pred=1; conf/train.yaml: num_hist=3), or the dataset's full clip length for this
    project's default full-clip causal training (every next-step in one pass, a
    superset of what windowed training supervises). See data/dataset.py's window_size
    for the paired dataset-side change.

    Regularizer aggregation (reference Section B.6 / models/dino.py's
    DinoV2Encoder.agg(agg_type="mlp")): the curvature and VICReg terms need ONE
    representative vector per frame, but the reference's best-performing way to get one
    from spatial (P>1) latents is a LEARNABLE aggregation head (a 3-layer MLP with a
    final LayerNorm, see aggregation_head.py), not a naive mean-pool over patches --
    mean-pooling is the reference's separately-ablated, worse-performing "mean" variant
    (Figure 14). `self.agg_head` (AggregationHead) is built here for
    projector_type='channel', and for token_mode='cls'/'raw' only when cls_agg_head=True —
    otherwise z is already 1 token/frame ("global features (nv=1): compute cosine
    similarity directly between vectors", no head needed). Call `self.aggregate(z)` to get the
    regularizer-ready tensor regardless of mode; train.py passes its output to
    WorldModelLoss as `z_agg`.
    """

    def __init__(self, feature_dim=384, latent_dim=384, hidden_dim=None, use_projector=True,
                 norm_type="layernorm", token_mode="cls", projector_type="global",
                 channel_out_dim=8, channel_hidden=192,
                 # cls mode only (predictor.py's CausalPredictor -- unchanged, see docstring)
                 nhead=8, pred_num_layers=2, max_len=16, residual_projector=False,
                 encoder_type="mlp", encoder_depth=4, encoder_expansion=4, encoder_dropout=0.0,
                 # raw mode only (pixel_encoder.py's PIXEL_ENCODER_REGISTRY)
                 pixel_encoder_type="small_resnet", raw_img_size=224,
                 # patch mode only (block_causal_predictor.py's BlockCausalPredictor)
                 num_patches=196, num_frames=16, depth=6, heads=16, dim_head=64, mlp_dim=2048,
                 emb_dropout=0.0, agg_mlp_hidden_dim=None, agg_out_dim=128,
                 # cls/raw mode only: opt-in aggregation head (default off, so existing
                 # checkpoints/configs are unchanged) -- see aggregate()
                 cls_agg_head=False,
                 # shared
                 dropout=0.1):
        super().__init__()
        self.token_mode = token_mode
        self.projector_type = projector_type
        self.agg_head = None
        self.pixel_encoder = None

        if token_mode in ("cls", "raw"):
            if token_mode == "raw":
                if pixel_encoder_type not in PIXEL_ENCODER_REGISTRY:
                    raise ValueError(f"pixel_encoder_type must be one of "
                                      f"{sorted(PIXEL_ENCODER_REGISTRY)}, got {pixel_encoder_type!r}")
                self.pixel_encoder = PIXEL_ENCODER_REGISTRY[pixel_encoder_type](
                    feature_dim=feature_dim, img_size=raw_img_size,
                )
            if encoder_type not in ENCODER_REGISTRY:
                raise ValueError(f"encoder_type must be one of {sorted(ENCODER_REGISTRY)}, "
                                  f"got {encoder_type!r}")
            self.encoder = ENCODER_REGISTRY[encoder_type](
                feature_dim=feature_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
                use_projector=use_projector, norm_type=norm_type, residual=residual_projector,
                encoder_depth=encoder_depth, encoder_expansion=encoder_expansion,
                encoder_dropout=encoder_dropout,
            )
            self.predictor = CausalPredictor(
                latent_dim=latent_dim, nhead=nhead, num_layers=pred_num_layers,
                dropout=dropout, max_len=max_len,
            )
            if cls_agg_head:
                # Same AggregationHead as patch/channel mode, with one token per frame:
                # the curvature/VICReg/SIGReg terms can then read this head's output
                # instead of z itself (z stays what the predictor consumes). Which terms
                # use it is picked by --vicreg-granularity / --sigreg-granularity
                # (agg = head output, token = raw z).
                self.agg_head = AggregationHead(
                    num_patches=1, in_dim=latent_dim, hidden=agg_mlp_hidden_dim,
                    out_dim=agg_out_dim,
                )
        elif token_mode == "patch":
            if cls_agg_head:
                raise ValueError("cls_agg_head only applies to token_mode 'cls'/'raw'; patch "
                                 "mode builds its own aggregation head for projector_type='channel'")
            self.encoder = PatchProjectorEncoder(
                feature_dim=feature_dim, projector_type=projector_type, latent_dim=latent_dim,
                channel_out_dim=channel_out_dim, channel_hidden=channel_hidden,
            )
            pred_dim = self.encoder.out_dim
            p = 1 if projector_type == "global" else num_patches
            self.predictor = BlockCausalPredictor(
                # A clip with T frames supplies T-1 next-step source positions.
                latent_dim=pred_dim, num_patches=p, num_frames=max(num_frames - 1, 1), depth=depth,
                heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
                emb_dropout=emb_dropout,
            )
            if projector_type == "channel":  # built now, not lazily (see class docstring)
                self.agg_head = AggregationHead(
                    num_patches=num_patches, in_dim=pred_dim, hidden=agg_mlp_hidden_dim,
                    out_dim=agg_out_dim,
                )
        else:
            raise ValueError(f"token_mode must be 'cls', 'raw' or 'patch', got {token_mode!r}")

    def aggregate(self, z):
        """Regularizer-ready per-frame vector: z unchanged (squeezed) when already 1
        token/frame, else the learnable aggregation head's output. NOT a mean-pool."""
        if self.agg_head is not None:
            return self.agg_head(z.unsqueeze(2) if z.dim() == 3 else z)  # cls: (B,T,D) -> P=1
        return z.squeeze(2) if z.dim() == 4 else z

    def forward(self, x):
        """cls: x (B,T,feature_dim) -> z, z_pred each (B,T,latent_dim).
        raw: x (B,T,3,H,W) raw pixels -> z, z_pred each (B,T,latent_dim) (run through
        self.pixel_encoder first to get a (B,T,feature_dim) vector, then identical to cls).
        patch: x (B,T,N,feature_dim) -> z (B,T,P,D), z_pred (B,T-1,P,D)."""
        if self.pixel_encoder is not None:
            x = self.pixel_encoder(x)
        z = self.encoder(x)

        if self.token_mode in ("cls", "raw"):
            z_pred = self.predictor(z)
            return z, z_pred

        # A next-step predictor has only T-1 source positions for a T-frame clip.  Passing
        # all T frames used to create an extra final output that the loss discarded.  In
        # the block-causal model that also left the final learned positional embedding
        # permanently gradient-free, because a causal final token cannot influence any
        # earlier (supervised) output.  Feed exactly the supervised source prefix instead.
        z_pred = self.predictor(z[:, :-1])  # patch: (B,T-1,P,D), predicts z[:,1:]
        return z, z_pred
