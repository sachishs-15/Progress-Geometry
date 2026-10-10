"""LiFT: Linearized Feature Trajectories (Bagad & Zisserman, NeurIPS 2025) plugin.

Wraps the pretrained LiFT model from `external/chirality_in_action/LiFT` — a DINOv2
ViT-S/14 (4 registers) backbone whose 16 linearly-sampled per-frame CLS features are fed
through a small dual-CLS transformer autoencoder trained to "straighten" the DINO
trajectory (inspired by perceptual straightening, Hénaff et al. 2019) into a compact
768-dim time-aware embedding.

Directly relevant to this project's own hypothesis: LiFT's whole point is to make the
trajectory MORE linear/ordinal than raw DINOv2's. So rather than exposing its pooled
768-dim video-level embedding (`lift_output["concat"]`), this plugin exposes its per-frame
*reconstruction* (`lift_output["reconstructed"]`, the linearized trajectory back in
384-dim DINO-feature space) as `embeddings` — that's what lets LiFT go through the exact
same per-frame ordinality/PCA-trajectory analysis (common/viz.py) as every other plugin,
which is the actually interesting comparison here (does LiFT's straightening show up as a
higher ordinality score than plain dinov2's?).

Requires the checkpoint already present at
`external/chirality_in_action/models/LiFT/<ckpt_name>` (not downloaded by this plugin —
see that repo's README for the gdown command) and the sibling LiFT repo checked out on
disk for its `lift`/`shared` packages (added to sys.path below, not pip-installed).

LiFT's transformer autoencoder was trained on exactly T=16 linearly-sampled frames per
video (see LiFT/precompute_features.py) — this plugin resamples whatever frames it's given
down to 16 via linspace, the same pattern the vjepa2/vjepa2.1 plugins use for their own
fixed clip lengths.
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from common.base import EncodingResult, FrozenEncoder, register

LIFT_REPO = Path(__file__).resolve().parents[4] / "external" / "chirality_in_action" / "LiFT"
LIFT_CKPT_ROOT = Path(__file__).resolve().parents[4] / "external" / "chirality_in_action" / "models" / "LiFT"
if str(LIFT_REPO) not in sys.path:
    sys.path.insert(0, str(LIFT_REPO))


@register
class LiFTEncoder(FrozenEncoder):
    name = "lift"
    supports_attention = True
    attention_label = "DINOv2 CLS -> patch attention (backbone, last block)"
    supports_patch_tokens = False
    # LiFT has no patch-pooling step of its own (its output is already one fixed-dim
    # vector per frame) — this choice list exists only so run.py's shared --pooling CLI
    # flag doesn't error out when this plugin is run alongside the others; it's a no-op.
    pooling_choices = ["mean_patch"]

    def __init__(self, ckpt_name: str = "ggwirp95-epoch=458-step=834003.ckpt",
                 dinov2_model_id: str = "vit_small_patch14_reg4_dinov2.lvd142m",
                 n_frames: int = 16, **kwargs):
        kwargs.setdefault("pooling", "mean_patch")
        super().__init__(**kwargs)
        self.ckpt_name = ckpt_name
        self.dinov2_model_id = dinov2_model_id
        self.n_frames = n_frames

    def load(self) -> None:
        from lift import DINOv2ForVideo, load_lift_module, make_classification_eval_transform

        self.backbone = DINOv2ForVideo(model_id=self.dinov2_model_id).to(self.device).eval()
        self.preprocess = make_classification_eval_transform()
        self.model = load_lift_module(ckpt_root=str(LIFT_CKPT_ROOT), ckpt_name=self.ckpt_name).to(self.device)

    def _resample(self, frames: List[Image.Image]):
        idx = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(int)
        return [frames[i] for i in idx], idx

    def _preprocess_clip(self, clip: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(f) for f in clip])  # (T, C, H, W)
        return x.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, C, T, H, W)

    @torch.no_grad()
    def encode(self, frames: List[Image.Image]) -> EncodingResult:
        from lift.demo import compute_lift_embeddings

        clip, clip_idx = self._resample(frames)
        x = self._preprocess_clip(clip)
        dino_feats = self.backbone(x)[0]  # (T, D=384)

        lift_output = compute_lift_embeddings(dino_feats.unsqueeze(0), self.model, device=self.device)
        reconstructed = lift_output["reconstructed"].squeeze(0).numpy()  # (T, D) linearized trajectory

        from lift.dinov2 import get_cls_attention_maps
        frames_thw = x[0].permute(1, 0, 2, 3)  # (T, C, H, W)
        attn_maps = get_cls_attention_maps(self.backbone, frames_thw, device=self.device)

        return EncodingResult(embeddings=reconstructed, frame_indices=clip_idx.tolist(), attention_maps=attn_maps)
