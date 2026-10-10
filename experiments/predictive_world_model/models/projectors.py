"""CNN projectors for DINOv2 patch tokens, ported from
external/temporal-straightening/models/dino.py's GlobalProjector/ChannelProjector.

Both already end in normalization by construction (GlobalProjector: final LayerNorm;
ChannelProjector: LayerNorm after every conv including the last) — unlike ClsMlpEncoder
(cls-mode), which had no such guarantee and needed norm_type added explicitly after the
K400 instability (see encoder.py, norms.py). Ported near-verbatim; the only changes are
replacing the reference's Hydra-object conv_layers configs with plain dicts, and default
constructor args that reproduce conf/encoder/dino_global.yaml / dino_channel.yaml exactly,
so calling either with no args matches the paper's own setup.
"""
import torch.nn as nn
import torch.nn.functional as F


class GlobalProjector(nn.Module):
    """Reference conf/encoder/dino_global.yaml: in_dim=384, out_dim=384, hidden=384,
    pool_hw=1 (full pooling -> exactly 1 token/frame). 3 stride-2 conv blocks each halve
    H,W (14x14 -> 2x2 for a 196-patch grid), then AdaptiveAvgPool2d((1,1)) -> flatten ->
    LayerNorm."""

    def __init__(self, in_dim=384, out_dim=384, hidden=384, pool_hw=1, gn_groups=8,
                 conv_layers=None):
        super().__init__()
        self.pool_hw = int(pool_hw)
        conv_layers = conv_layers or [
            {"in_dim": hidden, "out_dim": hidden, "kernel_size": 3, "stride": 2, "padding": 1},
            {"in_dim": hidden, "out_dim": hidden, "kernel_size": 3, "stride": 2, "padding": 1},
            {"in_dim": hidden, "out_dim": hidden, "kernel_size": 3, "stride": 2, "padding": 1},
        ]

        self.mix = nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(num_groups=max(1, min(gn_groups, hidden)), num_channels=hidden)

        blocks = []
        for cfg in conv_layers:
            blocks.append(nn.Conv2d(cfg["in_dim"], cfg["out_dim"], kernel_size=cfg["kernel_size"],
                                     stride=cfg["stride"], padding=cfg["padding"], bias=False))
            blocks.append(nn.GroupNorm(num_groups=max(1, min(gn_groups, cfg["out_dim"])),
                                        num_channels=cfg["out_dim"]))
            blocks.append(nn.GELU())
        last_dim = conv_layers[-1]["out_dim"]
        self.down_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.head = nn.Conv2d(last_dim, out_dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        """x: (B, in_dim, H, W) -> (B, out_dim) if pool_hw==1, else (B, pool_hw*pool_hw, out_dim)"""
        x = F.gelu(self.gn0(self.mix(x)))
        x = self.down_blocks(x)
        x = self.pool(x)
        x = self.head(x)
        if self.pool_hw == 1:
            x = x.flatten(1)
            return self.ln(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.ln(x)


class ChannelProjector(nn.Module):
    """Reference conf/encoder/dino_channel.yaml: in_dim=384, out_dim=8, norm_type="layer".
    Both convs stride=1 -> NO spatial downsampling, only channel depth is crushed
    (384 -> 192 -> 8). Keeps the full patch grid, unlike GlobalProjector."""

    def __init__(self, in_dim=384, out_dim=8, hidden=192, norm_type="layer"):
        super().__init__()
        self.norm_type = norm_type
        conv_layers = [
            {"in_dim": in_dim, "out_dim": hidden, "kernel_size": 3, "stride": 1, "padding": 1},
            {"in_dim": hidden, "out_dim": out_dim, "kernel_size": 3, "stride": 1, "padding": 1},
        ]
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.layer_norm_layers = nn.ModuleList()
        for cfg in conv_layers:
            self.conv_layers.append(nn.Conv2d(cfg["in_dim"], cfg["out_dim"], kernel_size=cfg["kernel_size"],
                                               stride=cfg["stride"], padding=cfg["padding"]))
            self.batch_norm_layers.append(nn.BatchNorm2d(cfg["out_dim"]))
            self.layer_norm_layers.append(nn.LayerNorm(cfg["out_dim"]))
        self.activation = nn.ReLU()
        self.out_dim = out_dim

    def forward(self, x):
        """x: (B, in_dim, H, W) -> (B, H*W, out_dim) — same H, W as input."""
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.norm_type == "batch":
                x = self.batch_norm_layers[i](x)
            elif self.norm_type == "layer":
                x = x.permute(0, 2, 3, 1)
                x = self.layer_norm_layers[i](x)
                x = x.permute(0, 3, 1, 2)
            if len(self.conv_layers) > 1 and i != len(self.conv_layers) - 1:
                x = self.activation(x)
        return x.flatten(2).transpose(1, 2)
