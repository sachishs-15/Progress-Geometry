from common.base import register
from common.hf_vit import GenericViTEncoder


@register
class DINOv3Encoder(GenericViTEncoder):
    name = "dinov3"
    # Gated on HF: request access at
    # https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and set
    # HF_TOKEN (`huggingface-cli login`) before this will load.
    default_hf_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
