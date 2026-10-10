"""Standalone experiment -- NOT part of the tracked pipeline, touches no existing file.

Prior finding (this session): a soft curvature loss on ClsMlpEncoder's (LayerNorm'd, NOT
unit-norm) output barely reduces curvature regardless of weight, and neither
straighten_weight nor chord_weight beats raw dinov2 on SSv2. None of the 4 existing
norm_type options (models/norms.py: layernorm, clip_rmsnorm, batchnorm, none) actually
constrain ||z||=1 -- LayerNorm fixes per-feature variance, not vector norm; ClipRMSNorm/
BatchNorm rescale but don't project onto a sphere. But the EVAL metric
(clip_progress_metrics) L2-normalizes every frame before computing anything. This script
tests training directly on the unit hypersphere instead, three ways, all reusing
ClsMlpEncoder(norm_type="none") + CausalPredictor from models/ (imported, not copied) with
a hard L2-projection on top:

  sphere_basic     -- L2-normalize encoder output; keep L1 prediction loss + the existing
                       cosine straightening loss (both computed post-normalization, so
                       training finally matches what evaluation measures exactly).
  sphere_cosine    -- same sphere projection, but prediction loss becomes
                       1 - cos(z_pred, z_target) (angular) instead of L1 (Euclidean) --
                       matches the loss geometry to the constraint geometry.
  sphere_uniform   -- sphere_cosine + a hyperspherical uniformity regularizer (Wang &
                       Isola 2020, "Understanding Contrastive Representation Learning
                       through Alignment and Uniformity on the Hypersphere", ICML 2020:
                       L_uniform = log E[exp(-t * ||x_i - x_j||^2)] over random pairs)
                       IN PLACE OF VICReg std/cov, which are Euclidean-variance based and
                       don't match sphere geometry -- uniformity is the sphere-native
                       anti-collapse term the paper derives from first principles.

All three trained on SSv2 (train/val, domain-matched, same as the earlier scratch runs),
same architecture size and epoch budget as the existing cls_mlp.yaml recipe, so results
are directly comparable to raw dinov2 / the SSv2-trained baseline / chord_weight already
measured in this session.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoder import ClsMlpEncoder  # noqa: E402 (needs sys.path insert above)
from models.predictor import CausalPredictor  # noqa: E402

TRAIN_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt"
VAL_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt"
TRAIN_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv"
VAL_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv"
OUT_ROOT = Path("/home/sachish/proc-planning/experiments/predictive_world_model/scratch_hypersphere")


# ---- ordinality metrics, reimplemented inline (same formulas LiFT's own
# visualization/progress_metrics.py uses; self-contained, no sibling-project import) ----
def clip_progress_metrics(z, eps=1e-8):
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
    rows = [m for m in (clip_progress_metrics(z_all[i]) for i in range(len(z_all))) if m is not None]
    return pd.DataFrame(rows).median().to_dict()


class ClipDataset(Dataset):
    def __init__(self, features_path, manifest_path, path_col="video_path"):
        data = torch.load(features_path, map_location="cpu")
        feats, paths = data["features"].float(), data["video_paths"]
        idx = {p: i for i, p in enumerate(paths)}
        df = pd.read_csv(manifest_path, dtype=str)
        keep = [idx[p] for p in df[path_col] if p in idx]
        self.z = feats[keep]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        return self.z[i]


def sphere_project(z, eps=1e-8):
    return z / (z.norm(dim=-1, keepdim=True) + eps)


def straightening_loss(z, step_thresh=1e-6):
    steps = z[:, 1:] - z[:, :-1]
    v1, v2 = steps[:, :-1], steps[:, 1:]
    n1, n2 = v1.norm(dim=-1), v2.norm(dim=-1)
    valid = (n1 > step_thresh) & (n2 > step_thresh)
    if not valid.any():
        return z.new_zeros(())
    cos = F.cosine_similarity(v1, v2, dim=-1)
    return (1 - cos[valid]).mean()


def uniformity_loss(z, t=2.0, n_pairs=2048):
    """Wang & Isola 2020: log E[exp(-t*||xi-xj||^2)] over random pairs of (already unit-
    norm) vectors -- pushes the WHOLE population toward a uniform spread on the sphere,
    the sphere-native analogue of VICReg's Euclidean variance/covariance terms."""
    x = z.reshape(-1, z.shape[-1])
    n = x.shape[0]
    i = torch.randint(0, n, (n_pairs,), device=x.device)
    j = torch.randint(0, n, (n_pairs,), device=x.device)
    sq_dist = (x[i] - x[j]).pow(2).sum(dim=-1)
    return torch.logsumexp(-t * sq_dist, dim=0) - np.log(n_pairs)


def run_epoch(encoder, predictor, loader, device, variant, optimizer=None):
    train = optimizer is not None
    encoder.train(train)
    predictor.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x in loader:
            x = x.to(device)
            z = sphere_project(encoder(x))
            z_pred = predictor(z)
            z_src, z_tgt = z_pred[:, :-1], z[:, 1:].detach()

            if variant == "sphere_basic":
                pred_loss = F.l1_loss(z_src, z_tgt)
            else:  # sphere_cosine, sphere_uniform
                pred_loss = (1 - F.cosine_similarity(z_src, z_tgt, dim=-1)).mean()

            straighten = straightening_loss(z)
            loss = pred_loss + 0.1 * straighten
            if variant == "sphere_uniform":
                loss = loss + 1.0 * uniformity_loss(z)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), 1.0)
                optimizer.step()
            total += loss.item()
            n += 1
    return total / n


def get_all_z(encoder, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = []
    encoder.eval()
    with torch.no_grad():
        for x in loader:
            chunks.append(sphere_project(encoder(x.to(device))).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_variant(variant, device, epochs=100):
    torch.manual_seed(0)
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ClipDataset(TRAIN_FEATURES, TRAIN_MANIFEST)
    val_ds = ClipDataset(VAL_FEATURES, VAL_MANIFEST)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    encoder = ClsMlpEncoder(feature_dim=384, latent_dim=384, hidden_dim=384,
                             use_projector=True, norm_type="none").to(device)
    predictor = CausalPredictor(latent_dim=384, nhead=8, num_layers=2, max_len=16).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()),
                             lr=1e-4, weight_decay=1e-5)

    rows = []
    t0 = time.time()
    for epoch in range(epochs):
        train_loss = run_epoch(encoder, predictor, train_loader, device, variant, opt)
        val_loss = run_epoch(encoder, predictor, val_loader, device, variant, optimizer=None)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if epoch % 10 == 0 or epoch == epochs - 1:
            z_val = get_all_z(encoder, val_ds, device)
            m = evaluate_ordinality(z_val)
            row.update({f"ord_{k}": v for k, v in m.items()})
            print(f"[{variant}] epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"spearman={m['time_progress_spearman']:.4f} "
                  f"straightness={m['trajectory_straightness']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
               out_dir / "model.pt")
    z_val = get_all_z(encoder, val_ds, device)
    final = evaluate_ordinality(z_val)
    print(f"\n[{variant}] FINAL (val): {final}")
    return final


if __name__ == "__main__":
    variant = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    train_variant(variant, device)
