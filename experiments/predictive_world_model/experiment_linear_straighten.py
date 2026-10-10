"""Standalone experiment -- NOT part of the tracked pipeline, touches no existing file.

Question: why does raw (untrained) DINOv2 beat this project's trained ClsMlpEncoder on
time_progress_spearman, and can we fix it?

Diagnostic done before writing this script (see conversation): the existing
straighten_weight loss (1 - cos(adjacent step vectors), losses/world_model_loss.py) barely
moves measured curvature at all across the WHOLE sw=0.05..0.9 sweep (1.24-1.26 vs raw's
1.26 -- a >=3% reduction at best), yet high sw still measurably hurts spearman. Likely
cause: a soft cosine penalty on a handful of consecutive step vectors is fighting
concentration-of-measure in 384-dim space (two generic directions are already ~orthogonal),
and the prediction loss has zero incentive to counteract that.

LiFT's own answer to this (external/chirality_in_action/LiFT/lift/model.py:300-324) is not
a soft loss at all: z_t = z_s + (t/T)*z_d, a HARD architectural constraint -- every
trajectory is exactly linear by construction, so there is nothing for a curvature loss to
fight. This script tests a lightweight version of that idea against this project's own
raw dinov2 CLS features on SSv2 (train encoder+static/dynamic heads on ssv2_train,
evaluate ordinality on ssv2_validation), self-contained: no import from any sibling
project, the ordinality metric formulas are reimplemented inline below (same formulas
LiFT's visualization/progress_metrics.py uses, credited in each docstring).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda:1"
TRAIN_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt"
VAL_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt"
TRAIN_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv"
VAL_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv"
OUT_DIR = Path("/home/sachish/proc-planning/experiments/predictive_world_model/scratch_linear_straighten")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- ordinality metrics, reimplemented inline (same formulas as LiFT's
# visualization/progress_metrics.py's clip_progress_metrics, kept self-contained so this
# script has zero dependency on any sibling project) ----
def clip_progress_metrics(z, eps=1e-8):
    """z: (T, D) numpy, one clip's raw per-frame latents."""
    t = z.shape[0]
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)
    steps = z[1:] - z[:-1]
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = float(step_norms.sum())
    start_end_disp = float(np.linalg.norm(z[-1] - z[0]))
    straightness = start_end_disp / path_length if path_length > eps else 0.0
    cos_adj = np.sum(z[:-1] * z[1:], axis=1)
    adj_cos_dist = float(np.mean(1.0 - cos_adj))

    u = z[-1] - z[0]
    u_sq = float(np.dot(u, u))
    if u_sq < 1e-10:
        return None
    p = (z - z[0]) @ u / u_sq
    time_idx = np.arange(t)
    spearman = float(spearmanr(time_idx, p).statistic)
    p_steps = p[1:] - p[:-1]
    tv = float(np.sum(np.abs(p_steps)))
    efficiency = 1.0 / tv if tv > eps else 0.0
    return {"trajectory_straightness": straightness, "time_progress_spearman": spearman,
            "progress_efficiency": efficiency, "adjacent_cosine_distance": adj_cos_dist,
            "start_end_displacement": start_end_disp}


def evaluate_ordinality(z_all: np.ndarray) -> dict:
    """z_all: (N, T, D) -> median of every clip_progress_metrics field."""
    rows = [m for m in (clip_progress_metrics(z_all[i]) for i in range(len(z_all))) if m is not None]
    df = pd.DataFrame(rows)
    return df.median().to_dict()


# ---- data ----
class ClipDataset(Dataset):
    def __init__(self, features_path, manifest_path, path_col="video_path"):
        data = torch.load(features_path, map_location="cpu")
        feats, paths = data["features"].float(), data["video_paths"]
        idx = {p: i for i, p in enumerate(paths)}
        df = pd.read_csv(manifest_path, dtype=str)
        keep = [idx[p] for p in df[path_col] if p in idx]
        self.z = feats[keep]  # (N, T, D)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        return self.z[i]


# ---- model: LiFT-style hard linear reparametrization ----
class StaticDynamicHead(nn.Module):
    """Encodes a whole (T, D) clip into one static vector z_s and one dynamic vector z_d
    via a small TransformerEncoder with two learnable readout tokens (mirrors LiFT's
    DualCLSTransformer, much smaller). Trajectory is then FORCED linear:
    z_t = z_s + (t/(T-1)) * z_d for every t -- curvature is exactly 0 by construction,
    nothing for a loss to fight."""

    def __init__(self, dim=384, d_model=256, nhead=8, num_layers=2, max_len=16):
        super().__init__()
        self.in_proj = nn.Linear(dim, d_model)
        self.static_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dynamic_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.static_out = nn.Linear(d_model, dim)
        self.dynamic_out = nn.Linear(d_model, dim)

    def forward(self, x):
        """x: (B, T, D) -> z_pred: (B, T, D), exactly linear in t."""
        b, t, _ = x.shape
        tok = self.in_proj(x)
        static = self.static_token.expand(b, -1, -1)
        dynamic = self.dynamic_token.expand(b, -1, -1)
        seq = torch.cat([static, dynamic, tok], dim=1) + self.pos_embed[:, :t + 2]
        out = self.transformer(seq)
        z_s = self.static_out(out[:, 0])   # (B, D)
        z_d = self.dynamic_out(out[:, 1])  # (B, D)
        alpha = torch.linspace(0, 1, t, device=x.device).view(1, t, 1)  # (1, T, 1)
        return z_s.unsqueeze(1) + alpha * z_d.unsqueeze(1)  # (B, T, D)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for z in loader:
            z = z.to(device)
            z_pred = model(z)
            loss = F.mse_loss(z_pred, z)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / n


def get_all_z(model, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for z in loader:
            chunks.append(model(z.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


if __name__ == "__main__":
    train_ds = ClipDataset(TRAIN_FEATURES, TRAIN_MANIFEST)
    val_ds = ClipDataset(VAL_FEATURES, VAL_MANIFEST)
    print(f"train={len(train_ds)} val={len(val_ds)} D={train_ds.z.shape[-1]} T={train_ds.z.shape[1]}")

    raw_val = val_ds.z.numpy()
    raw_metrics = evaluate_ordinality(raw_val)
    print(f"raw dinov2 (val):        {raw_metrics}")

    model = StaticDynamicHead(dim=train_ds.z.shape[-1], max_len=train_ds.z.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StaticDynamicHead params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader_simple = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    log_rows = []
    t0 = time.time()
    for epoch in range(100):
        train_loss = run_epoch(model, train_loader, DEVICE, opt)
        val_loss = run_epoch(model, val_loader_simple, DEVICE, optimizer=None)
        row = {"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss}
        if epoch % 10 == 0 or epoch == 99:
            z_val_pred = get_all_z(model, val_ds, DEVICE)
            m = evaluate_ordinality(z_val_pred)
            row.update({f"ord_{k}": v for k, v in m.items()})
            print(f"epoch {epoch}: train_mse={train_loss:.4f} val_mse={val_loss:.4f} "
                  f"spearman={m['time_progress_spearman']:.4f} "
                  f"straightness={m['trajectory_straightness']:.4f} "
                  f"([{time.time()-t0:.0f}s elapsed])")
        log_rows.append(row)

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "train_log.csv", index=False)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")

    z_val_pred = get_all_z(model, val_ds, DEVICE)
    final_metrics = evaluate_ordinality(z_val_pred)
    print("\n=== FINAL (val) ===")
    print(f"raw dinov2:            {raw_metrics}")
    print(f"linear static+dynamic: {final_metrics}")
