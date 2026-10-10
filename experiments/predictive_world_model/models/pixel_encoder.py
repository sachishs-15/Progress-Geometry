"""Trainable raw-pixel encoders -- the token_mode="raw" counterpart to feature_provider.py's
DINOv2 tiers. Every other token_mode (cls, patch) starts from a FROZEN pretrained backbone;
this is the one place in the project where the backbone itself has gradients, so the
straightening/VICReg/prediction losses shape pixel-level features from scratch rather than
only the small projector on top of them.

SmallResnetPixelEncoder is a self-contained copy of
external/temporal-straightening/models/encoder/resnet.py's SmallResNet (same 5
downsampling resblocks + Linear-GELU-Linear projection head, same conf/encoder/
scratch_resnet.yaml's dim=384 default) -- copied rather than imported so this project
keeps standing on its own (same convention predictor.py's PositionalEncoding docstring
describes), and because the reference's resblock/SmallResNet classes are private module
globals, not something meant to be imported cross-project.

Output feature_dim matches whatever ENCODER_REGISTRY (encoder.py: "mlp" / "residual_mlp")
already expects as its `feature_dim` input, so token_mode="raw" reuses that registry,
ClsMlpEncoder/ResidualMlpEncoder unchanged, and CausalPredictor unchanged -- see
ProceduralWorldModel's docstring for how this wires together. Normalization (ImageNet
mean/std) lives HERE, not in the data pipeline, mirroring the reference's own resnet18
class (which also self-normalizes) -- FeatureProvider's token_mode="raw" tier hands back
plain uint8-range-or-0..1 pixels, unnormalized, by design.
"""
import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Pre-activation-ish residual conv block with a strided downsample on the skip path
    -- verbatim port of temporal-straightening's `resblock` (down-only, this project has
    no use for its `resample=None` variant here)."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.skip = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding),
        )
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding),
            nn.MaxPool2d(2, stride=2),
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idnty = self.skip(x)
        residual = self.act1(self.bn1(self.conv1(x)))
        residual = self.act2(self.bn2(self.conv2(residual)))
        return idnty + residual


class SmallResnetPixelEncoder(nn.Module):
    """Raw RGB frame -> single feature_dim vector, trained end-to-end (no pretrained
    weights, no frozen backbone anywhere in this module). 5 stride-2 resblocks take a
    224x224 input down to 256x7x7, flattened and projected to feature_dim.

    img_size other than 224 works (every resblock just halves whatever spatial size it's
    given) but is untested here -- 224 is what conf/encoder/scratch_resnet.yaml and this
    project's other token_modes (patch_img_size=196, DINOv2's own 224 default) both are
    already in the same ballpark of, so it's the default rather than a free parameter.
    """

    def __init__(self, feature_dim: int = 384, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False,
        )

        self.rb1 = _ResBlock(3, 16)
        self.rb2 = _ResBlock(16, 32)
        self.rb3 = _ResBlock(32, 64)
        self.rb4 = _ResBlock(64, 128)
        self.rb5 = _ResBlock(128, 256)

        spatial = img_size // 32  # 5 stride-2 blocks
        self.flat = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(256 * spatial * spatial, 512),
            nn.GELU(),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) -> (B, T, feature_dim). Accepts FeatureProvider's
        token_mode="raw" tier's uint8 [0,255] frames directly (cast + scaled to [0,1]
        here, not on disk -- see feature_provider.py's _compute_live_raw for why it stays
        uint8 through the live-cache) as well as an already-float [0,1] tensor, so a unit
        test can hand this module float input with no separate uint8 fixture. T is
        folded into the batch dim for the conv stack (every frame encoded independently,
        matching every other encoder in this family -- temporal modelling lives entirely
        in the predictor, never here)."""
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self._mean) / self._std
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.flat(x)
        x = self.projection(x)
        return x.reshape(b, t, -1)


# {name: constructor} -- same convention as encoder.py's ENCODER_REGISTRY / norms.py's
# NORM_REGISTRY, so a new token_mode="raw" backbone becomes selectable via
# --pixel-encoder-type with no other code changes.
PIXEL_ENCODER_REGISTRY = {
    "small_resnet": lambda **kw: SmallResnetPixelEncoder(
        feature_dim=kw["feature_dim"], img_size=kw.get("img_size", 224),
    ),
}
