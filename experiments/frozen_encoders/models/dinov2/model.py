from common.base import register
from common.hf_vit import GenericViTEncoder


@register
class DINOv2Encoder(GenericViTEncoder):
    name = "dinov2"
    default_hf_id = "facebook/dinov2-with-registers-small"


@register
class DINOv2BaseEncoder(GenericViTEncoder):
    name = "dinov2_base"
    default_hf_id = "facebook/dinov2-with-registers-base"


@register
class DINOv2LargeEncoder(GenericViTEncoder):
    name = "dinov2_large"
    default_hf_id = "facebook/dinov2-with-registers-large"


@register
class DINOv2GiantEncoder(GenericViTEncoder):
    name = "dinov2_giant"
    default_hf_id = "facebook/dinov2-with-registers-giant"


@register
class DINOv2NoRegEncoder(GenericViTEncoder):
    """facebook/dinov2-small: the NO-register-token checkpoint (num_register_tokens=
    None, unlike every other class in this file, which use the *-with-registers-*
    variants) -- matches external/temporal-straightening's own DINOv2 checkpoint
    (torch.hub's "dinov2_vits14", confirmed architecturally identical: patch_size=14,
    hidden_size=384, no registers).

    default_image_size=196 overrides the HF processor's own default (a 224 crop,
    which at patch_size=14 gives a 16x16=256 patch grid) to instead give a 14x14=196
    grid -- matching experiments/predictive_world_model's patch_channel/patch_global
    configs and the reference paper's own Table 1 "14x14x..." dims exactly.
    """
    name = "dinov2_noreg"
    default_hf_id = "facebook/dinov2-small"
    default_image_size = 196
