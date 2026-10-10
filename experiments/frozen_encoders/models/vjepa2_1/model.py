"""V-JEPA 2.1 (per-frame image mode) plugin.

Loaded via a community HF-`transformers`-format port of Meta's official weights
(`apiantonio/vjepa2.1-vit-base-384`), not `torch.hub` — `facebookresearch/vjepa2`'s own
`torch.hub` loader is confirmed broken as of now: `src/hub/backbones.py` has its real
checkpoint URL commented out and replaced with a `http://localhost:8300` placeholder,
which fails with `Connection refused`. This HF repo's own README documents that its
weights are Meta's, copied bit-exact (only the fused QKV projection is split into
separate q/k/v matrices, a re-parameterization not a weight change) from the SAME
checkpoint (`vjepa2_1_vitb_dist_vitG_384.pt`) that's still genuinely publicly hosted by
Meta at https://dl.fbaipublicfiles.com/vjepa2/ (verified: HTTP 200, correct size) — the
official repo's *hub loader* is broken, not the weights themselves.

Uses `trust_remote_code=True` to load this repo's custom modeling code, since V-JEPA2.1
isn't in mainline `transformers` yet. That executes third-party (not Meta's own account)
Python on load — a real trust boundary, accepted deliberately for this specific checkpoint
after checking its provenance, not something to extend to arbitrary `trust_remote_code`
repos without the same scrutiny.

Per-frame, not per-clip: V-JEPA2's usual mode (see models/vjepa2/model.py) consumes a
whole multi-frame clip and pools each 2-frame tubelet into one embedding. This checkpoint's
config instead has `img_temporal_dim_size=1`: feeding it a "clip" with exactly T=1 frame
routes through a separate single-image patch embedding (tubelet_size=1) rather than the
video path, so batching each input frame as its own T=1 clip yields one embedding per
INPUT FRAME — directly comparable to per-frame image encoders like DINOv2/DINOv3, unlike
the tubelet-pooled output you'd get from feeding it a real multi-frame clip.
"""
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

from common.base import EncodingResult, FrozenEncoder, register
from common.pooling import pool_patch_tokens


@register
class VJEPA21Encoder(FrozenEncoder):
    name = "vjepa2.1"
    supports_attention = False
    supports_patch_tokens = True
    pooling_choices = ["mean_patch", "max_patch"]

    def __init__(self, hf_id: str = "apiantonio/vjepa2.1-vit-base-384", **kwargs):
        kwargs.setdefault("pooling", "mean_patch")
        super().__init__(**kwargs)
        self.hf_id = hf_id

    def load(self) -> None:
        self.processor = AutoVideoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.hf_id, trust_remote_code=True, dtype=torch.float32)
        self.model.to(self.device).eval()

    def _patch_tokens(self, frames: List[Image.Image]) -> torch.Tensor:
        clips = [[f] for f in frames]  # each frame its own T=1 "clip" -> image code path
        inputs = self.processor(clips, return_tensors="pt").to(self.device)
        out = self.model(**inputs, skip_predictor=True)
        return out.last_hidden_state  # (T, P, D) — one row per input frame, no tubelet pooling

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        tokens = self._patch_tokens(frames)
        pooled = pool_patch_tokens(tokens, self.pooling)
        return EncodingResult(embeddings=pooled.cpu().numpy(), frame_indices=list(range(len(frames))))

    @torch.no_grad()
    def get_patch_tokens(self, frames: List[Image.Image]):
        tokens = self._patch_tokens(frames)  # (T, P, D)
        grid = int(round(tokens.shape[1] ** 0.5))
        return tokens, (grid, grid)
