"""Shared loading for the encoder_planning_eval scripts: ProceduralWorldModel and LiFT
checkpoints, precomputed feature files (single file or `<name>.shard*.pt` shards), and
encoding them to per-frame latents.
"""
import inspect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

PROC_ROOT = Path(__file__).resolve().parents[3]
PWM_ROOT = PROC_ROOT / "experiments" / "predictive_world_model"
LIFT_ROOT = PROC_ROOT / "external" / "chirality_in_action" / "LiFT"

sys.path.insert(0, str(PWM_ROOT))
from models.world_model import ProceduralWorldModel  # noqa: E402

PREDICTOR_PREFIX = "predictor."
RAW_MODES = ("cls", "mean", "flatten")


def load_world_model(checkpoint, config, device, with_predictor=False):
    """Rebuild a ProceduralWorldModel from its train.py config.yaml and load its weights.
    Constructor kwargs come from the config where present and the constructor's own
    defaults otherwise; num_frames is derived exactly as train.py's predictor_num_frames.
    with_predictor=False drops the predictor's weights (encoder/agg_head only)."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    if "agg_hidden" in cfg:  # checkpoints from before the agg_hidden -> agg_mlp_hidden_dim rename
        cfg.setdefault("agg_mlp_hidden_dim", cfg["agg_hidden"])
    params = inspect.signature(ProceduralWorldModel.__init__).parameters
    kwargs = {k: v for k, v in cfg.items() if k in params}
    window = cfg.get("window_size") or 0
    kwargs["num_frames"] = window + 1 if window > 0 else cfg.get("n_frames", 16)

    model = ProceduralWorldModel(**kwargs).to(device).eval()
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"]
    if not with_predictor:
        state = {k: v for k, v in state.items() if not k.startswith(PREDICTOR_PREFIX)}
    else:
        # Older patch checkpoints allocated one positional block for every clip frame,
        # although the last predictor output had no next-frame target and its exclusive
        # positional row never received a gradient. New models allocate T-1 source
        # blocks. Drop that known-dead trailing block while preserving old checkpoints.
        pos_key = "predictor.pos_embedding"
        target = model.state_dict().get(pos_key)
        saved = state.get(pos_key)
        if target is not None and saved is not None and saved.shape[1] > target.shape[1]:
            expected_legacy_len = target.shape[1] + model.predictor.num_patches
            if (saved.shape[0] != target.shape[0] or saved.shape[2:] != target.shape[2:]
                    or saved.shape[1] != expected_legacy_len):
                raise SystemExit(
                    f"checkpoint/config architecture mismatch for {pos_key}: "
                    f"saved={tuple(saved.shape)}, expected={tuple(target.shape)}"
                )
            state = dict(state)
            state[pos_key] = saved[:, :target.shape[1]]
            print(f"Migrated legacy {pos_key} {tuple(saved.shape)} -> {tuple(target.shape)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not with_predictor:
        missing = [k for k in missing if not k.startswith(PREDICTOR_PREFIX)]
    if missing or unexpected:
        raise SystemExit(f"checkpoint/config architecture mismatch: missing={missing}, "
                          f"unexpected={unexpected}")
    epoch = ckpt.get("epoch")
    print(f"Loaded {checkpoint} (epoch {epoch}; predictor {'loaded' if with_predictor else 'skipped'})")
    return model, cfg, epoch


def load_lift_model(checkpoint, device):
    sys.path.insert(0, str(LIFT_ROOT))
    from lift import load_lift_module  # lazy: pulls in lightning
    model = load_lift_module(ckpt_root=str(Path(checkpoint).resolve().parent),
                             ckpt_name=Path(checkpoint).name).to(device)
    print(f"Loaded LiFT checkpoint {checkpoint}")
    return model


def _feature_files(path):
    p = Path(path)
    if p.exists():
        return [p]
    shards = sorted(p.parent.glob(p.name + ".shard*.pt"),
                    key=lambda s: int(s.name.rsplit(".shard", 1)[1].split(".")[0]))
    if not shards:
        raise SystemExit(f"no features file or shards found for {path}")
    return shards


def load_matched_features(path, df, path_col="video_path"):
    """Features for the rows of manifest `df`, in df's row order. Handles sharded files
    and memory-maps them, so only the matched clips are read from disk. Rows with no
    features are dropped. Returns (features tensor, matched df with a fresh index)."""
    want = defaultdict(list)
    for i, vp in enumerate(df[path_col]):
        want[vp].append(i)
    parts, rows = [], []
    for f in _feature_files(path):
        try:
            data = torch.load(f, mmap=True)
        except RuntimeError:
            data = torch.load(f)
        file_idx, df_idx = [], []
        for j, vp in enumerate(data["video_paths"]):
            for i in want.get(vp, ()):
                file_idx.append(j)
                df_idx.append(i)
        if file_idx:
            parts.append(data["features"][file_idx])
            rows.extend(df_idx)
        del data
    if not rows:
        raise SystemExit(f"no rows matched between {path} and the manifest")
    order = np.argsort(rows, kind="stable")
    feats = torch.cat(parts)[torch.from_numpy(order)]
    matched = df.iloc[np.asarray(rows)[order]].reset_index(drop=True)
    print(f"Matched {len(matched)}/{len(df)} manifest rows to {path}")
    return feats, matched


def reduce_raw(features, mode):
    """Raw (untrained) per-frame latents straight from a features tensor."""
    if mode == "cls":
        if features.dim() != 3:
            raise SystemExit(f"raw 'cls' needs (N,T,D) features, got {tuple(features.shape)}; "
                              f"use 'mean'/'flatten' for patch features")
        return features.numpy()
    if features.dim() != 4:
        raise SystemExit(f"raw '{mode}' needs (N,T,P,D) patch features, got {tuple(features.shape)}")
    if mode == "mean":
        return features.mean(dim=2).numpy()
    n, t, p, d = features.shape
    return features.reshape(n, t, p * d).numpy()


@torch.no_grad()
def encode_world_model(model, features, device, batch_size=64):
    """Per-frame latents z_agg = model.aggregate(model.encoder(x)) -- the same per-frame
    vector the straightening/VICReg losses see in training. (N, T, D) numpy.

    token_mode='raw' checkpoints have a model.pixel_encoder that must run first (raw
    pixel frames -> feature_dim vector) before model.encoder -- same order as
    ProceduralWorldModel.forward() itself. features is (N,T,3,H,W) uint8/float pixels
    for those checkpoints, (N,T,D) precomputed features otherwise."""
    out = []
    for s in range(0, len(features), batch_size):
        x = features[s:s + batch_size].to(device)
        if model.pixel_encoder is not None:
            x = model.pixel_encoder(x)
        out.append(model.aggregate(model.encoder(x)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_lift(model, features, device, batch_size=64):
    """LiFT's reconstructed per-frame trajectory in DINO-feature space. (N, T, D) numpy."""
    from lift import compute_lift_embeddings
    if features.dim() != 3:
        raise SystemExit(f"LiFT needs plain (N,T,D) DINOv2 CLS features, got {tuple(features.shape)}")
    out = []
    for s in range(0, len(features), batch_size):
        out.append(compute_lift_embeddings(features[s:s + batch_size].to(device), model,
                                           device=device)["reconstructed"].numpy())
    return np.concatenate(out)
