"""Standalone experiment -- NOT part of the tracked pipeline, touches no existing file.

Continues from experiment_hypersphere.py's finding: hard L2-normalization + angular
prediction loss + Wang&Isola hyperspherical uniformity (instead of VICReg) is the first
free-form, genuinely-trained encoder in this whole investigation to MATCH raw dinov2 on
time_progress_spearman (0.9412 = 0.9412) with a stable (non-declining) training curve.
This script keeps that winning recipe fixed and varies two new things:

  (A) Does conditioning the encoder on the GOAL frame help? "Progress" is inherently
      relative to a goal -- x_t alone can't distinguish "70% done with task A" from
      "70% done with task B" as easily as (x_t, x_goal) can. Three ways to inject the
      goal, increasing in how much goal information leaks into every frame's encoding:
        goal_concat  -- z_t = MLP([x_t ; x_T])           (sees raw goal features)
        goal_delta   -- z_t = MLP([x_t ; x_t - x_T])      (sees only the offset from goal)
        goal_film    -- z_t = MLP(x_t) FiLM-modulated by a small MLP(x_T) (gamma, beta)
                        (goal only rescales/shifts the per-frame computation, never
                        concatenated in directly -- the leakiest-to-least-leaky ordering
                        is goal_concat > goal_delta > goal_film > no-goal baseline)
      All three keep CausalPredictor and the sphere_uniform loss recipe unchanged
      (imported from models/predictor.py, not reimplemented), only the ENCODER differs.

  (B) Does adding a goal-distance SHAPING loss help without leaking goal info into the
      encoder's input at all (the "fair" comparison to raw dinov2, which also never sees
      the goal)? goal_shaping: same encoder as sphere_uniform (x_t alone), same losses,
      PLUS a VIP-style (arXiv:2210.00030) monotonic-distance-to-goal term:
      mean(relu(d_{t+1} - d_t)) where d_t = 1 - cos(z_t, z_goal) -- penalizes any step
      that moves further from the goal than the previous frame was.

Every variant trained on SSv2 (train/val, same domain-matched setup as the rest of this
session's experiments), same epoch budget, directly comparable to raw dinov2 (spearman
0.9412) and sphere_uniform (0.9412) already measured.
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
from models.predictor import CausalPredictor  # noqa: E402

TRAIN_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt"
VAL_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt"
TRAIN_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv"
VAL_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv"
OUT_ROOT = Path("/home/sachish/proc-planning/experiments/predictive_world_model/scratch_goal_conditioning")


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
    monotonic = float(np.mean(p_steps >= -eps))
    return {"trajectory_straightness": straightness, "time_progress_spearman": spearman,
            "progress_efficiency": efficiency, "adjacent_cosine_distance": adj_cos_dist,
            "start_end_displacement": start_end_disp, "monotonic_progress_fraction": monotonic}


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
    x = z.reshape(-1, z.shape[-1])
    n = x.shape[0]
    i = torch.randint(0, n, (n_pairs,), device=x.device)
    j = torch.randint(0, n, (n_pairs,), device=x.device)
    sq_dist = (x[i] - x[j]).pow(2).sum(dim=-1)
    return torch.logsumexp(-t * sq_dist, dim=0) - np.log(n_pairs)


# ---- encoder variants ----
class BaselineEncoder(nn.Module):
    """z_t = MLP(x_t) -- no goal info, same as experiment_hypersphere.py's ClsMlpEncoder
    stand-in (reimplemented here, not imported, so this file's --variant switch can share
    one training loop across every encoder shape without touching models/encoder.py)."""

    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D). Goal not used."""
        return self.net(x)


class GoalConcatEncoder(nn.Module):
    """z_t = MLP([x_t ; x_goal]) -- sees the goal frame's raw features directly,
    concatenated onto every timestep's input."""

    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        goal = x[:, -1:].expand_as(x)
        return self.net(torch.cat([x, goal], dim=-1))


class GoalDeltaEncoder(nn.Module):
    """z_t = MLP([x_t ; x_t - x_goal]) -- sees only the OFFSET from goal, not the raw
    goal features themselves -- less leaky than GoalConcatEncoder (can't reconstruct
    x_goal from x_t and the delta alone without also already knowing x_t precisely, but
    the offset directly encodes "how far is this frame from done")."""

    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        goal = x[:, -1:].expand_as(x)
        return self.net(torch.cat([x, x - goal], dim=-1))


class GoalFilmEncoder(nn.Module):
    """z_t = gamma(x_goal) * MLP_base(x_t) + beta(x_goal) -- goal only rescales/shifts
    the per-frame computation (FiLM, arXiv:1709.07871), never concatenated in directly.
    The least leaky of the three: at gamma=1, beta=0 this reduces exactly to
    BaselineEncoder, so goal conditioning is an ADDITIVE capability the model can choose
    to use or not, rather than a mandatory input change."""

    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.film = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden))
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        goal = x[:, -1]  # (B, D)
        gamma_beta = self.film(goal)  # (B, 2*hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B, hidden) each
        h = self.base(x)  # (B, T, hidden)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.out(h)


class CrossFrameGoalDeltaEncoder(nn.Module):
    """Every encoder variant so far computes z_t from x_t (and x_goal) in total
    isolation from its neighbours -- the only place frame t can ever "hear about" frame
    t+1 is indirectly, through the prediction loss's backward pass. This variant lets
    the encoder itself attend across the whole clip: input tokens are the GoalDelta
    representation [x_t - x_goal] for every t, fed through a small BIDIRECTIONAL (not
    causal -- this is the encoder, not the predictor) TransformerEncoder, so z_t can
    directly use what nearby frames look like, not just its own frame + the goal."""

    def __init__(self, dim=384, hidden=None, d_model=256, nhead=8, num_layers=2, max_len=16):
        super().__init__()
        self.in_proj = nn.Linear(dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, dim)

    def forward(self, x):
        t = x.shape[1]
        goal = x[:, -1:].expand_as(x)
        delta = x - goal
        h = self.in_proj(delta) + self.pos_embed[:, :t]
        h = self.transformer(h)
        return self.out_proj(h)


class CrossFrameBaselineEncoder(nn.Module):
    """Same bidirectional TransformerEncoder as CrossFrameGoalDeltaEncoder, but fed raw
    x_t directly -- NO goal conditioning at all. Isolates whether cross-frame attention
    alone (letting frame t's encoding see its neighbours) accounts for
    CrossFrameGoalDeltaEncoder's big win, or whether the goal-delta input is still doing
    real work on top of it."""

    def __init__(self, dim=384, hidden=None, d_model=256, nhead=8, num_layers=2, max_len=16):
        super().__init__()
        self.in_proj = nn.Linear(dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, dim)

    def forward(self, x):
        t = x.shape[1]
        h = self.in_proj(x) + self.pos_embed[:, :t]
        h = self.transformer(h)
        return self.out_proj(h)


class CausalGoalDeltaXframeEncoder(nn.Module):
    """FIX for CrossFrameGoalDeltaEncoder's flaw: that encoder used FULL bidirectional
    attention, so z_t could see raw content from FUTURE frames t+1..T-1, not just the
    goal -- not a realistic constraint (at inference you don't have future intermediate
    frames, only the current one and a known goal). This version applies the exact same
    causal mask CausalPredictor uses (models/predictor.py, imported not copied) so
    z_t only attends to positions <= t; the goal is still visible everywhere because
    it's baked into the INPUT itself (x_t - x_goal, computed per-position before any
    attention happens) -- the goal is a privileged, always-known input (like a
    goal-conditioned policy's goal), not a future frame being peeked at."""

    def __init__(self, dim=384, hidden=None, d_model=256, nhead=8, num_layers=2, max_len=16):
        super().__init__()
        self.in_proj = nn.Linear(dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                   norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, dim)

    def forward(self, x):
        t = x.shape[1]
        goal = x[:, -1:].expand_as(x)
        delta = x - goal
        h = self.in_proj(delta) + self.pos_embed[:, :t]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        return self.out_proj(h)


class GoalDeltaOnlyEncoder(nn.Module):
    """z_t = MLP(x_t - x_goal) -- drops raw x_t entirely, sees ONLY the offset from
    goal. Purest version of the delta idea: removes any raw-appearance information the
    MLP could use to satisfy the uniformity/straightening losses in ways uncorrelated
    with actual progress (e.g. "which video is this" identity leakage via x_t alone)."""

    def __init__(self, dim=384, hidden=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        goal = x[:, -1:].expand_as(x)
        return self.net(x - goal)


ENCODERS = {
    "baseline": BaselineEncoder,
    "goal_concat": GoalConcatEncoder,
    "goal_delta": GoalDeltaEncoder,
    "goal_delta_only": GoalDeltaOnlyEncoder,
    "goal_film": GoalFilmEncoder,
    "goal_delta_xframe": CrossFrameGoalDeltaEncoder,
    "baseline_xframe": CrossFrameBaselineEncoder,
    "goal_delta_causal_xframe": CausalGoalDeltaXframeEncoder,
}


def interp_fit_loss(z):
    """Directly optimizes what experiment_score_all.py's interp_cosine_sim MEASURES but
    training never targeted until now: pulls each intermediate z_t toward the straight
    line between z_0 and z_T (the model's OWN endpoints, differentiable end to end),
    instead of only shaping local adjacent-pair curvature (straightening_loss) or a
    single global direction (goal_shaping_loss's cos-to-goal). This is the training-side
    analogue of the interpolation-quality gap found in this session: every variant so
    far tied or beat raw dinov2 on time_progress_spearman while scoring WORSE than raw
    on interp_cosine_sim (chord-projection rank can stay monotonic even while the path
    wanders off-axis; this loss directly penalizes that wander)."""
    t = z.shape[1]
    alpha = torch.linspace(0, 1, t, device=z.device).view(1, t, 1)
    z_hat = z[:, :1] + alpha * (z[:, -1:] - z[:, :1])
    cos = F.cosine_similarity(z[:, 1:-1], z_hat[:, 1:-1], dim=-1)
    return (1 - cos).mean()


def slerp_fit_loss(z, eps=1e-7):
    """Spherical counterpart of interp_fit_loss. That one builds its target as a
    EUCLIDEAN chord z_0 + alpha*(z_T - z_0), which passes through the sphere's interior;
    because the comparison is cosine (scale-invariant) the target is effectively
    normalize(lerp) = "nlerp", which lands on the correct great circle but traverses it
    with distorted timing -- lagging true constant-angular-velocity motion by up to ~1.1
    degrees at alpha=0.25 for 60-degree-separated endpoints, leading by the same at 0.75.
    SLERP is the constant-angular-velocity geodesic between z_0 and z_T:
        slerp(alpha) = [sin((1-alpha)*theta)*z_0 + sin(alpha*theta)*z_T] / sin(theta)
    Everything else in this recipe is already angular (pred_loss, goal_shaping_loss) or
    monotone in angle (uniformity_loss, since ||x_i-x_j||^2 = 2-2cos for unit vectors),
    so this makes the interpolation term match the rest of the geometry."""
    t = z.shape[1]
    z0, zT = z[:, :1], z[:, -1:]                              # (B,1,D), unit norm
    cos_theta = (z0 * zT).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)                             # (B,1,1)
    sin_theta = torch.sin(theta)
    alpha = torch.linspace(0, 1, t, device=z.device).view(1, t, 1)
    z_hat = (torch.sin((1 - alpha) * theta) * z0 + torch.sin(alpha * theta) * zT) / sin_theta
    cos = F.cosine_similarity(z[:, 1:-1], z_hat[:, 1:-1], dim=-1)
    return (1 - cos).mean()


def goal_shaping_loss(z, step_thresh=1e-6):
    """VIP-style (arXiv:2210.00030): distance-to-goal must not increase step to step.
    d_t = 1 - cos(z_t, z_goal); penalize any (d_{t+1} - d_t) > 0."""
    goal = z[:, -1:].expand_as(z)
    d = 1 - F.cosine_similarity(z, goal, dim=-1)  # (B, T)
    increase = F.relu(d[:, 1:] - d[:, :-1])
    return increase.mean()


def run_epoch(encoder, predictor, loader, device, use_shaping, use_interp=False,
               interp_weight=0.5, use_slerp=False, optimizer=None):
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

            pred_loss = (1 - F.cosine_similarity(z_src, z_tgt, dim=-1)).mean()
            straighten = straightening_loss(z)
            loss = pred_loss + 0.1 * straighten + 1.0 * uniformity_loss(z)
            if use_shaping:
                loss = loss + 0.5 * goal_shaping_loss(z)
            if use_interp:
                fit = slerp_fit_loss(z) if use_slerp else interp_fit_loss(z)
                loss = loss + interp_weight * fit

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


def train_variant(variant, device, epochs=100, use_shaping=False, use_interp=False,
                   interp_weight=0.5, use_slerp=False, tag=None):
    torch.manual_seed(0)
    name = tag or variant
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ClipDataset(TRAIN_FEATURES, TRAIN_MANIFEST)
    val_ds = ClipDataset(VAL_FEATURES, VAL_MANIFEST)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    encoder = ENCODERS[variant](dim=384, hidden=384).to(device)
    predictor = CausalPredictor(latent_dim=384, nhead=8, num_layers=2, max_len=16).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()),
                             lr=1e-4, weight_decay=1e-5)

    rows = []
    t0 = time.time()
    for epoch in range(epochs):
        train_loss = run_epoch(encoder, predictor, train_loader, device, use_shaping, use_interp,
                                interp_weight, use_slerp, opt)
        val_loss = run_epoch(encoder, predictor, val_loader, device, use_shaping, use_interp,
                              interp_weight, use_slerp, optimizer=None)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if epoch % 10 == 0 or epoch == epochs - 1:
            z_val = get_all_z(encoder, val_ds, device)
            m = evaluate_ordinality(z_val)
            row.update({f"ord_{k}": v for k, v in m.items()})
            print(f"[{name}] epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"spearman={m['time_progress_spearman']:.4f} "
                  f"straightness={m['trajectory_straightness']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
               out_dir / "model.pt")
    z_val = get_all_z(encoder, val_ds, device)
    final = evaluate_ordinality(z_val)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n[{name}] FINAL (val, encoder_params={n_params}): {final}")
    return final


if __name__ == "__main__":
    variant = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    use_shaping = "--shaping" in sys.argv
    use_interp = "--interp" in sys.argv
    use_slerp = "--slerp" in sys.argv
    epochs = 100
    interp_weight = 0.5
    for a in sys.argv:
        if a.startswith("--epochs="):
            epochs = int(a.split("=")[1])
        if a.startswith("--interp-weight="):
            interp_weight = float(a.split("=")[1])
    tag = (variant + ("_shaping" if use_shaping else "") + ("_interp" if use_interp else "")
           + ("_slerp" if use_slerp else "")
           + (f"_iw{interp_weight}" if use_interp and interp_weight != 0.5 else ""))
    train_variant(variant, device, epochs=epochs, use_shaping=use_shaping, use_interp=use_interp,
                  interp_weight=interp_weight, use_slerp=use_slerp, tag=tag)
