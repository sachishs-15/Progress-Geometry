"""LeVLJEPA (Kuhn et al. 2026, https://huggingface.co/lukaskuhn/LeVLJEPA-ViT-B-DataComp-200k)
vision encoder plugin.

Non-contrastive vision-language pretraining method; this plugin only wants its frozen
vision tower (a plain `vit_base_patch16_224` from timm, trained on DataComp-large) as a
per-frame image encoder -- the paired GPT-2 text encoder and the cross-modal
`projector.*` weights in the checkpoint are for image-text alignment and irrelevant here,
so only the `encoder.*` (backbone) weights are loaded. The `pre_proj.*` head (768->256,
the space the contrastive/prediction loss was actually computed in) is also skipped: like
every other plain-ViT plugin here, this exposes the backbone's own 768-dim CLS/patch
tokens, not a downstream projection head's space, so pooling choices are comparable
across plugins.

Not an AutoModel/AutoImageProcessor checkpoint (unlike common/hf_vit.py's plugins) -- the
weights are plain timm state dict shards on the Hub, loaded by hand per the model card's
own usage example (hf_hub_download + safetensors, matching the key prefixes the card
documents).

No attention_maps support: HF's `attn_implementation="eager"` (used by every other plugin
here) materializes readable per-layer attention weights for free, but timm's `Attention`
module keeps its softmax(QK^T) tensor purely local to `forward()` even with
`fused_attn=False` -- reading it out would mean monkeypatching the block's forward method,
not just hooking it. Not worth the fragility for a first pass.
"""
from typing import List

import torch
import torch.nn as nn
import timm
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

from common.base import EncodingResult, FrozenEncoder, register
from common.pooling import pool_patch_tokens

HIDDEN = 768
DEFAULT_REPO = "lukaskuhn/LeVLJEPA-ViT-B-DataComp-200k"


@register
class LeVLJEPAEncoder(FrozenEncoder):
    name = "levljepa"
    supports_attention = False
    supports_patch_tokens = True
    pooling_choices = ["cls", "mean_patch", "max_patch"]

    def __init__(self, hf_id: str = DEFAULT_REPO, timm_model: str = "vit_base_patch16_224",
                 **kwargs):
        kwargs.setdefault("pooling", "mean_patch")
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.timm_model = timm_model

    def load(self) -> None:
        self.model = timm.create_model(self.timm_model, pretrained=False, num_classes=0,
                                        dynamic_img_size=True)
        weights_path = hf_hub_download(self.hf_id, "vision_encoder.safetensors")
        state_dict = load_file(weights_path)
        encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items()
                          if k.startswith("encoder.")}
        self.model.load_state_dict(encoder_state)
        self.model.to(self.device).eval()
        # Matches the model card's own preprocessing exactly (plain ImageNet-normalized
        # resize+crop, not an HF AutoImageProcessor since there isn't one for this checkpoint).
        self.processor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _tokens(self, frames: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.processor(f.convert("RGB")) for f in frames]).to(self.device)
        return self.model.forward_features(x)  # (T, num_prefix_tokens + P, D)

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        tokens = self._tokens(frames)
        n_prefix = self.model.num_prefix_tokens  # 1 (CLS only -- no registers in this checkpoint)
        cls_tok = tokens[:, 0]
        patch_toks = tokens[:, n_prefix:]
        pooled = pool_patch_tokens(patch_toks, self.pooling, cls_token=cls_tok)
        return EncodingResult(embeddings=pooled.cpu().numpy(), frame_indices=list(range(len(frames))))

    @torch.no_grad()
    def get_patch_tokens(self, frames: List[Image.Image]):
        tokens = self._tokens(frames)
        n_prefix = self.model.num_prefix_tokens
        patch_toks = tokens[:, n_prefix:]  # (T, P, D)
        grid = int(round(patch_toks.shape[1] ** 0.5))
        return patch_toks.cpu(), (grid, grid)
