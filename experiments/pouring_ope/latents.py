"""Cached tokens -> per-frame latents: pooling specs for frozen encoders, or a trained checkpoint."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROC_ROOT / "experiments" / "encoder_planning_eval"))

PARTS = "cls | mean | max | gridG (patch grid pooled to GxG then flattened) | flatten (every patch)"


def parse_pool(spec):
    parts = spec.split("+")
    for p in parts:
        if p not in ("cls", "mean", "max", "flatten") and not (p.startswith("grid") and p[4:].isdigit()):
            raise SystemExit(f"unknown pooling part {p!r}; parts are {PARTS}, joined with '+'")
    return parts


def needed_tokens(spec):
    parts = parse_pool(spec)
    kinds = set()
    if "cls" in parts:
        kinds.add("cls")
    if any(p != "cls" for p in parts):
        kinds.add("patch")
    return kinds


def reduce_tokens(cls, patch, grid_hw, spec, part_norm="none"):
    """cls (T, D) / patch (T, P, D) -> (T, d). Parts are concatenated in the order given."""
    out = []
    for p in parse_pool(spec):
        if p == "cls":
            x = cls
        elif p == "mean":
            x = patch.mean(dim=1)
        elif p == "max":
            x = patch.max(dim=1).values
        elif p == "flatten":
            x = patch.flatten(1)
        else:
            g = int(p[4:])
            h, w = grid_hw
            grid = patch.transpose(1, 2).reshape(patch.shape[0], patch.shape[2], h, w)
            x = F.adaptive_avg_pool2d(grid, g).flatten(1)
        if part_norm == "l2":
            x = F.normalize(x, dim=-1)
        out.append(x)
    return torch.cat(out, dim=-1)


def load_checkpoint(checkpoint, device):
    from eval_common.loading import load_world_model  # puts predictive_world_model on sys.path
    ckpt = Path(checkpoint)
    model, cfg, _ = load_world_model(ckpt, ckpt.parent / "config.yaml", device)
    mode = cfg.get("token_mode", "cls")
    if mode == "raw":
        raise SystemExit("token_mode='raw' checkpoints consume pixels, not cached tokens")
    return model, mode


CKPT_POOLS = ("mean", "flatten", "agg")


@torch.no_grad()
def encode_checkpoint(model, cls, patch, device, pool="mean"):
    """Per-frame latent from a trained checkpoint, taken at the encoder output.

    mean / flatten: encoder(x) is (T, P, C) per-patch tokens (patch checkpoints); pool them
        over patches by mean (-> C) or concatenation (-> P*C). Before the aggregation head.
    agg: aggregate(encoder(x)), the vector the training losses see (after the head).
    cls checkpoints have one token per frame, so every option returns encoder(x)."""
    x = (cls if patch is None else patch).float().unsqueeze(0).to(device)
    z = model.encoder(x)
    if pool == "agg":
        return model.aggregate(z).squeeze(0).cpu()
    z = z.squeeze(0)
    if z.dim() == 3:
        z = z.mean(dim=1) if pool == "mean" else z.flatten(1)
    return z.cpu()
