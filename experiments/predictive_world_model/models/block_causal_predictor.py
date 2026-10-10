"""Block-causal ViT predictor, ported near-verbatim from
external/temporal-straightening/models/vit.py's Attention/FeedForward/Transformer/
ViTPredictor. The reference uses exactly ONE predictor class for every encoder mode
(global, cls, channel/spatial) — block-causal attention with num_patches=1 collapses to
plain causal attention.

Used here for token_mode='patch' (both projector_type='global', num_patches=1, and
'channel', num_patches>1) — NOT for token_mode='cls', which keeps this project's
original CausalPredictor (predictor.py, nn.TransformerEncoderLayer + sinusoidal
positional encoding) unchanged, so existing cls-mode configs/checkpoints (e.g.
configs/cls_mlp_k400.yaml, checkpoints/pwm_cls_mlp_k400*) stay loadable/reproducible.
Patch mode is new in this project (no prior configs/checkpoints depend on its
predictor's internals), so it adopts the reference's exact predictor architecture
directly instead.

Deviates from PyTorch's nn.TransformerEncoderLayer (this project's earlier
implementation) in two ways that matter for matching the reference exactly:

1. Attention has a separately configurable `dim_head` (reference default: 64),
   independent of the token's own channel dim `latent_dim` — it projects UP to
   heads*dim_head before attending and back down to latent_dim after. PyTorch's
   built-in MultiheadAttention can't do this: it forces head_dim = embed_dim/nhead,
   which for the channel projector's 8-dim tokens with nhead=8 would give a
   nearly-useless 1-dim-per-head attention. The reference keeps full attention
   capacity (heads=16, dim_head=64 -> 1024-dim attention space) regardless of how
   small the projector has crushed the token's own channel depth.

2. Positional information is a single LEARNED parameter over the whole flattened
   (num_frames * num_patches) sequence (nn.Parameter, trained by backprop) — not a
   fixed sinusoidal encoding broadcast per frame (this project's earlier choice).

`num_frames` here is the predictor's own sequence-length capacity — the reference sizes
it to num_hist (a short fixed history window, e.g. 3), not necessarily a whole clip's
frame count; train.py may pass a shorter length than the dataset's own clip length when
--window-size is set (see data/dataset.py), or the dataset's clip length itself for
full-clip causal training (--window-size 0, this project's default).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_block_causal_mask(num_frames, num_patches):
    """(1, 1, T*P, T*P) boolean mask, True where a (frame,patch) token may attend.
    Block-lower-triangular of (P,P) all-True/all-False blocks — ported from
    generate_mask_matrix(npatch, nwindow), without its global mutable
    NUM_FRAMES/NUM_PATCHES state or its hardcoded .to('cuda')."""
    frame_allowed = torch.tril(torch.ones(num_frames, num_frames, dtype=torch.bool))
    token_allowed = frame_allowed.repeat_interleave(num_patches, dim=0).repeat_interleave(num_patches, dim=1)
    return token_allowed.unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout_p = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        b, n, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in qkv)  # (B,H,N,dim_head)

        # Same math as the manual matmul -> masked_fill -> softmax -> dropout -> matmul
        # this replaced, but routed through PyTorch's fused kernel: when a flash/memory-
        # efficient backend is selected, the full (B,heads,N,N) score matrix is never
        # materialized in HBM (computed/reduced in on-chip tiles instead), which is what
        # made N=3136 (full 16-frame window) need ~9.6 GiB/sample under the old code.
        # mask is already True-where-allowed (see build_block_causal_mask), matching
        # SDPA's own boolean-mask convention directly -- no ~mask inversion needed here.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, :, :n, :n],
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockCausalPredictor(nn.Module):
    def __init__(self, latent_dim, num_patches, num_frames, depth=6, heads=16, dim_head=64,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        # Single learned position embedding over the whole flattened sequence (reference:
        # ViTPredictor.pos_embedding) -- NOT a per-frame-broadcast sinusoidal term.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, latent_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(latent_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(latent_dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        # Built once at max capacity (num_frames), sliced per forward for shorter inputs
        # -- registered as a non-persistent buffer so .to(device) moves it automatically
        # without it polluting state_dict checkpoints.
        self.register_buffer("_mask", build_block_causal_mask(num_frames, num_patches), persistent=False)

    def forward(self, z):
        """
        Args:
            z: [B, T, P, latent_dim] -- encoder's per-frame, per-patch latents, T <=
                self.num_frames (the predictor's own capacity)
        Returns:
            z_pred: [B, T, P, latent_dim] -- z_pred[:, t] predicts the frame after input
                frame t (block-causal:
                dense within frame t and to all earlier frames, never to frame > t)
        """
        b, t, p, d = z.shape
        assert p == self.num_patches, f"expected {self.num_patches} patches/frame, got {p}"
        assert t <= self.num_frames, (
            f"predictor built for at most {self.num_frames} frames, got {t} -- pass a "
            f"matching --window-size/--n-frames, or rebuild with a larger num_frames"
        )
        x = z.reshape(b, t * p, d)
        x = x + self.pos_embedding[:, : t * p]
        x = self.emb_dropout(x)
        for attn, ff in self.layers:
            x = attn(x, self._mask) + x
            x = ff(x) + x
        x = self.norm(x)
        return x.reshape(b, t, p, d)
