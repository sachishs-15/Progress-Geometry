"""CLIP (ViT image tower, has a CLS token) plugin.

Same token layout as DINOv2/DINOv3 — CLS token at index 0 followed by patch
tokens, no registers — so it slots straight into `GenericViTEncoder` with just
the model/processor classes overridden: `AutoModel` on a CLIP checkpoint loads
the full text+vision `CLIPModel` (whose forward wants text inputs too), so we
need the vision-only `CLIPVisionModel` explicitly.
"""
from transformers import CLIPImageProcessor, CLIPVisionModel

from common.base import register
from common.hf_vit import GenericViTEncoder


@register
class CLIPEncoder(GenericViTEncoder):
    name = "clip"
    default_hf_id = "openai/clip-vit-base-patch16"

    def load(self) -> None:
        self.processor = CLIPImageProcessor.from_pretrained(self.hf_id)
        self.model = CLIPVisionModel.from_pretrained(self.hf_id, attn_implementation="eager")
        self.model.to(self.device).eval()
