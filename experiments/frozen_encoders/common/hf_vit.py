"""Generic plain-ViT-via-HF-`transformers` encoder.

DINOv2 and DINOv3 (and any other timm/DINO-style ViT with register tokens that
HF exposes through AutoModel/AutoImageProcessor) share the exact same forward
API: `last_hidden_state` = [CLS, registers..., patches...], plus optional
`attentions`. So the per-model folder only needs to set `default_hf_id`/`name`
— no bespoke inference code.
"""
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from common.base import EncodingResult, FrozenEncoder
from common.pooling import pool_patch_tokens


class GenericViTEncoder(FrozenEncoder):
    supports_attention = True
    attention_label = "attention (CLS -> patches)"
    pooling_choices = ["cls", "mean_patch", "max_patch"]
    supports_patch_tokens = True
    default_hf_id: str = None
    # None: use the HF processor's own default resize/crop. Set (by a subclass's
    # class attribute, or the image_size= constructor kwarg) to force a specific
    # square resize instead -- e.g. a checkpoint's default crop_size might not give
    # the patch-grid size a downstream consumer expects (patch_size=14 at the
    # default 224 crop gives a 16x16=256 grid; some pipelines need 14x14=196,
    # which needs a 196 crop instead).
    default_image_size: int = None

    def __init__(self, hf_id: str = None, image_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.hf_id = hf_id or self.default_hf_id
        self.image_size = image_size if image_size is not None else self.default_image_size

    def load(self) -> None:
        processor_kwargs = {}
        if self.image_size is not None:
            processor_kwargs = {
                "size": {"shortest_edge": self.image_size},
                "crop_size": {"height": self.image_size, "width": self.image_size},
            }
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id, **processor_kwargs)
        # eager attention so `output_attentions=True` actually returns weights
        # (sdpa/flash paths don't materialize them).
        self.model = AutoModel.from_pretrained(self.hf_id, attn_implementation="eager")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_attentions=True)

        # getattr(..., 0) only falls back when the attribute is ABSENT -- a
        # no-register checkpoint (e.g. facebook/dinov2-small) has
        # num_register_tokens present but set to None, so getattr returns None
        # (not 0) and `1 + None` would crash. `or 0` catches both cases.
        n_prefix = 1 + (getattr(self.model.config, "num_register_tokens", 0) or 0)
        cls_tok = out.last_hidden_state[:, 0]
        patch_toks = out.last_hidden_state[:, n_prefix:]
        pooled = pool_patch_tokens(patch_toks, self.pooling, cls_token=cls_tok)

        last_attn = out.attentions[-1]  # (T, heads, seq, seq)
        cls_attn = last_attn[:, :, 0, n_prefix:].mean(dim=1)  # (T, num_patches)
        grid = int(round(cls_attn.shape[-1] ** 0.5))
        attn_maps = cls_attn.reshape(-1, grid, grid).cpu().numpy()

        return EncodingResult(
            embeddings=pooled.cpu().numpy(),
            frame_indices=list(range(len(frames))),
            attention_maps=attn_maps,
        )

    @torch.no_grad()
    def get_patch_tokens(self, frames: List[Image.Image]):
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        out = self.model(**inputs)

        n_prefix = 1 + (getattr(self.model.config, "num_register_tokens", 0) or 0)
        patch_toks = out.last_hidden_state[:, n_prefix:]  # (T, P, D), CLS/registers dropped
        grid = int(round(patch_toks.shape[1] ** 0.5))
        return patch_toks.cpu(), (grid, grid)
