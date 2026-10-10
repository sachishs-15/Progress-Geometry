"""SigLIP (plain ViT, no CLS token) plugin.

Unlike DINOv2/DINOv3/CLIP, SigLIP has no CLS token in its sequence at all —
`last_hidden_state` is patch tokens only (verified: 196, not 197, for a
14x14 patch16/224 grid). Instead of a prepended summary token, SigLIP computes
its "whole image" embedding via a separate multi-head attention-pooling head,
exposed as `pooler_output` — added here as an extra `pooler` pooling choice
alongside the usual mean/max patch pooling.

Attention: with no CLS row to read "what does the summary token look at" from,
this uses the same "average attention received by each patch, last layer" idea
used for V-JEPA2. Cheap here (only ~196 patch tokens vs V-JEPA2's 8192), so a
plain `output_attentions=True` is fine — no hook/OOM workaround needed.
"""
from typing import List

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

from common.base import EncodingResult, FrozenEncoder, register
from common.pooling import pool_patch_tokens


@register
class SigLIPEncoder(FrozenEncoder):
    name = "siglip"
    supports_attention = True
    attention_label = "patch attention received (avg. over heads/queries, last layer)"
    supports_patch_tokens = True
    pooling_choices = ["pooler", "mean_patch", "max_patch"]

    def __init__(self, hf_id: str = "google/siglip-base-patch16-224", **kwargs):
        kwargs.setdefault("pooling", "mean_patch")
        super().__init__(**kwargs)
        self.hf_id = hf_id

    def load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = SiglipVisionModel.from_pretrained(self.hf_id, attn_implementation="eager")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_attentions=True)

        patch_toks = out.last_hidden_state  # (T, P, D) — no CLS/register prefix here

        if self.pooling == "pooler":
            pooled = out.pooler_output  # (T, D) — SigLIP's own attention-pooling head
        else:
            pooled = pool_patch_tokens(patch_toks, self.pooling)

        last_attn = out.attentions[-1]  # (T, heads, P, P)
        patch_attn = last_attn.mean(dim=1).mean(dim=1)  # avg over heads, then over queries -> (T, P)
        grid = int(round(patch_attn.shape[-1] ** 0.5))
        attn_maps = patch_attn.reshape(-1, grid, grid).cpu().numpy()

        return EncodingResult(
            embeddings=pooled.cpu().numpy(),
            frame_indices=list(range(len(frames))),
            attention_maps=attn_maps,
        )

    @torch.no_grad()
    def get_patch_tokens(self, frames: List[Image.Image]):
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        patch_toks = out.last_hidden_state  # (T, P, D)
        grid = int(round(patch_toks.shape[1] ** 0.5))
        return patch_toks, (grid, grid)
