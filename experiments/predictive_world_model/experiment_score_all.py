"""Standalone scoring script -- NOT part of the tracked pipeline, touches no existing file.

Loads every checkpoint written by experiment_hypersphere.py / experiment_goal_conditioning.py
and reports two independent signals for each:
  1. Ordinality metrics (same inline formulas used throughout this session).
  2. Interpolation quality: cosine_sim between a plain straight-line guess
     z_hat_t = z_0 + (t/(T-1))*(z_T - z_0) and the encoder's own actual z_t, at the
     intermediate positions only. This is a DIFFERENT question from ordinality -- it
     asks whether the encoder's own latent space is already well-approximated by a
     straight line between its own endpoints, independent of whether time_progress_spearman
     (which only cares about projection onto one axis) looks good.

Caveat printed inline for every goal_* row: those encoders see x_T (the goal frame) as
part of their OWN input when producing z_t, so their interpolation score is not directly
comparable to baseline/raw's -- of course z_t is easier to reconstruct from z_0/z_T when
the encoder already used x_T to compute z_t in the first place. Reported anyway, with the
caveat, because that circularity is itself the point of testing goal conditioning: the
question isn't "is this a fair fight" but "does giving the encoder the goal produce a
MORE progress-aware, more interpolation-friendly latent space than not giving it one."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.predictor import CausalPredictor  # noqa: E402
from models.encoder import ClsMlpEncoder  # noqa: E402
from experiment_goal_conditioning import ENCODERS, sphere_project  # noqa: E402

VAL_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt"
VAL_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv"
DEVICE = "cuda:0"


def clip_progress_metrics(z, eps=1e-8):
    t = z.shape[0]
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)
    steps = z[1:] - z[:-1]
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = float(step_norms.sum())
    start_end_disp = float(np.linalg.norm(z[-1] - z[0]))
    straightness = start_end_disp / path_length if path_length > eps else 0.0
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
            "progress_efficiency": efficiency}


def evaluate_ordinality(z_all):
    rows = [m for m in (clip_progress_metrics(z_all[i]) for i in range(len(z_all))) if m is not None]
    return pd.DataFrame(rows).median().to_dict()


def interpolation_quality(z_all):
    """z_all: (N, T, D), NOT necessarily unit-norm. Returns median cosine_sim between
    linear-interpolated and actual z at intermediate positions [1, T-2]."""
    n, t, d = z_all.shape
    alpha = np.linspace(0, 1, t).reshape(1, t, 1)
    z0 = z_all[:, :1]
    zT = z_all[:, -1:]
    z_hat = z0 + alpha * (zT - z0)
    a, b = z_all[:, 1:-1], z_hat[:, 1:-1]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    return float(np.median(cos))


def get_features():
    data = torch.load(VAL_FEATURES, map_location="cpu")
    feats, paths = data["features"].float(), data["video_paths"]
    idx = {p: i for i, p in enumerate(paths)}
    df = pd.read_csv(VAL_MANIFEST, dtype=str)
    keep = [idx[p] for p in df["video_path"] if p in idx]
    return feats[keep]


def score_checkpoint(name, encoder_variant, model_path, x_val):
    ckpt = torch.load(model_path, map_location=DEVICE)
    if encoder_variant == "cls_mlp_none":
        encoder = ClsMlpEncoder(feature_dim=384, latent_dim=384, hidden_dim=384,
                                 use_projector=True, norm_type="none").to(DEVICE)
    else:
        encoder = ENCODERS[encoder_variant](dim=384, hidden=384).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    with torch.no_grad():
        z = sphere_project(encoder(x_val.to(DEVICE))).cpu().numpy()
    ord_m = evaluate_ordinality(z)
    interp = interpolation_quality(z)
    return {"name": name, **ord_m, "interp_cosine_sim_median": interp}


if __name__ == "__main__":
    x_val = get_features()
    print(f"raw dinov2:  {evaluate_ordinality(x_val.numpy())}  "
          f"interp_cosine_sim={interpolation_quality(x_val.numpy()):.4f}")

    root = Path("/home/sachish/proc-planning/experiments/predictive_world_model")
    jobs = [
        ("sphere_basic", "cls_mlp_none", root / "scratch_hypersphere/sphere_basic/model.pt"),
        ("sphere_cosine", "cls_mlp_none", root / "scratch_hypersphere/sphere_cosine/model.pt"),
        ("sphere_uniform", "cls_mlp_none", root / "scratch_hypersphere/sphere_uniform/model.pt"),
        ("goal_baseline", "baseline", root / "scratch_goal_conditioning/baseline/model.pt"),
        ("goal_baseline_shaping", "baseline", root / "scratch_goal_conditioning/baseline_shaping/model.pt"),
        ("goal_concat", "goal_concat", root / "scratch_goal_conditioning/goal_concat/model.pt"),
        ("goal_delta", "goal_delta", root / "scratch_goal_conditioning/goal_delta/model.pt"),
        ("goal_film", "goal_film", root / "scratch_goal_conditioning/goal_film/model.pt"),
        ("goal_delta_only", "goal_delta_only", root / "scratch_goal_conditioning/goal_delta_only/model.pt"),
        ("goal_delta_shaping", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping/model.pt"),
        ("goal_concat_shaping", "goal_concat", root / "scratch_goal_conditioning/goal_concat_shaping/model.pt"),
        ("baseline_interp", "baseline", root / "scratch_goal_conditioning/baseline_interp/model.pt"),
        ("goal_delta_interp", "goal_delta", root / "scratch_goal_conditioning/goal_delta_interp/model.pt"),
        ("goal_delta_shaping_interp", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp/model.pt"),
        ("goal_delta_only_interp", "goal_delta_only", root / "scratch_goal_conditioning/goal_delta_only_interp/model.pt"),
        ("gds_iw0.15", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw0.15/model.pt"),
        ("gds_iw1.5", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw1.5/model.pt"),
        ("gds_iw3.0", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw3.0/model.pt"),
        ("gds_250ep", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp/model.pt"),
        ("gds_iw0.05", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw0.05/model.pt"),
        ("gds_iw0.1", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw0.1/model.pt"),
        ("gds_iw0.2", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw0.2/model.pt"),
        ("gds_iw0.3", "goal_delta", root / "scratch_goal_conditioning/goal_delta_shaping_interp_iw0.3/model.pt"),
        ("xframe_plain", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe/model.pt"),
        ("xframe_shaping", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping/model.pt"),
        ("xframe_shaping_interp", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw0.1/model.pt"),
        ("xframe_nogoal", "baseline_xframe", root / "scratch_goal_conditioning/baseline_xframe_shaping_interp_iw0.1/model.pt"),
        ("xframe_iw0.05", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw0.05/model.pt"),
        ("xframe_iw0.3", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw0.3/model.pt"),
        ("xframe_iw0.5", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp/model.pt"),
        ("xframe_iw0.8", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw0.8/model.pt"),
        ("xframe_iw1.0", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw1.0/model.pt"),
        ("xframe_iw1.5", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw1.5/model.pt"),
        ("xframe_iw2.0", "goal_delta_xframe", root / "scratch_goal_conditioning/goal_delta_xframe_shaping_interp_iw2.0/model.pt"),
        ("causal_xframe_noniterp", "goal_delta_causal_xframe", root / "scratch_goal_conditioning/goal_delta_causal_xframe_shaping/model.pt"),
        ("causal_xframe_iw0.5", "goal_delta_causal_xframe", root / "scratch_goal_conditioning/goal_delta_causal_xframe_shaping_interp/model.pt"),
        ("causal_xframe_iw1.0", "goal_delta_causal_xframe", root / "scratch_goal_conditioning/goal_delta_causal_xframe_shaping_interp_iw1.0/model.pt"),
        ("causal_xframe_iw2.0", "goal_delta_causal_xframe", root / "scratch_goal_conditioning/goal_delta_causal_xframe_shaping_interp_iw2.0/model.pt"),
    ]
    rows = []
    for name, variant, path in jobs:
        if not path.exists():
            print(f"[skip] {name}: {path} not found yet")
            continue
        rows.append(score_checkpoint(name, variant, path, x_val))
        print(rows[-1])

    if rows:
        pd.DataFrame(rows).to_csv(root / "scratch_all_scores.csv", index=False)
