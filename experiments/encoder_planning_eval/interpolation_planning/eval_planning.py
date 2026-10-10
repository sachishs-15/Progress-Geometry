"""Evaluate one encoder's support for start->goal planning (see ../README.md): predict
every intermediate frame's latent from (z1, zT) alone, two ways --

  - interpolation (training-free): z_t_hat = z1 + alpha*(zT - z1). A direct test of
    whether the encoder's own latent space is already "straight" between start and goal.
  - trained (optional, --predictor-checkpoint): PlanningTransformer's learned, bidirectional
    mask-token infilling (see train_predictor.py). Same eval set, same metrics -- the gap
    between the two is how much straightening alone leaves on the table for this encoder.

Usage (interpolation only, no training needed):
    python eval_planning.py \\
        --features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt \\
        --manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \\
        --out results/planning_dinov2_vits14_interp.json

Usage (adds the trained predictor):
    python eval_planning.py --features ... --manifest ... \\
        --predictor-checkpoint checkpoints/dinov2_vits14/best.ckpt \\
        --out results/planning_dinov2_vits14.json

Usage (a trained ProceduralWorldModel checkpoint's own z as the ENCODER under test,
instead of a raw features file -- --source-checkpoint/--source-config run --features
through that checkpoint's own encoder first, in memory, no separate extraction step):
    python eval_planning.py \\
        --features .../dinov2_noreg/ssv2_validation.pt --manifest val_split.csv \\
        --source-checkpoint ../predictive_world_model/checkpoints/pwm_patch_channel_k400_10pct/epoch39.ckpt \\
        --source-config ../predictive_world_model/checkpoints/pwm_patch_channel_k400_10pct/config.yaml \\
        --out results/planning_pwm_patch_channel_k400_10pct_epoch39.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PlanningDataset
from interpolate import interpolate_sequence
from metrics import summarize
from predictors.predictor import build_predictor

def compute_planning_dict(z_all, device, predictor_checkpoint=None, batch_size=512):
    """Everything __main__ does once it already has z_all (N,T,D) in memory -- no
    PlanningDataset, no --features file path. Lets a caller with an in-memory encoding
    (e.g. a checkpoint's own encoder output, pooled some non-standard way, never written
    to disk) get the full interpolation(+trained) dict directly.
    z_all: (N, T, D) tensor, any device -- moved to `device` internally in batches.
    predictor_checkpoint: optional path to a train_predictor.py checkpoint; omit for
    interpolation-only.
    Returns the same dict __main__ builds (n_clips, T, feature_dim, n_examples,
    interpolation, and -- if predictor_checkpoint given -- trained/predictor_checkpoint/
    predictor_type/predictor_epoch/gap_closed_by_training), minus the
    features/manifest/source_checkpoint path metadata, which only make sense for a CLI
    invocation with real file paths behind them.
    """
    n, t, d = z_all.shape
    model, ckpt = None, None
    if predictor_checkpoint:
        ckpt = torch.load(predictor_checkpoint, map_location=device)
        if ckpt["feature_dim"] != d:
            raise SystemExit(f"--predictor-checkpoint was trained on feature_dim={ckpt['feature_dim']} but got D={d}")
        if ckpt["max_len"] != t:
            raise SystemExit(f"--predictor-checkpoint was trained on T={ckpt['max_len']} but got T={t}")
        model = build_predictor(ckpt["predictor_type"], dim=d, max_len=t,
                                 **ckpt["arch_kwargs"]).to(device).eval()
        model.load_state_dict(ckpt["model"])

    interp_hat, trained_hat, z_true_all = [], [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            z = z_all[s:s + batch_size].to(device)  # (B, T, D)
            z1, zT = z[:, 0], z[:, -1]
            z_true_all.append(z[:, 1:-1].reshape(-1, d).cpu())
            interp_hat.append(interpolate_sequence(z1, zT, t)[:, 1:-1].reshape(-1, d).cpu())
            if model is not None:
                z_pred = model(z1, zT, t)
                trained_hat.append(z_pred[:, 1:-1].reshape(-1, d).cpu())

    z_true = torch.cat(z_true_all)
    interp_hat = torch.cat(interp_hat)

    result = {
        "n_clips": n, "T": t, "feature_dim": d, "n_examples": len(z_true),
        "interpolation": summarize(interp_hat, z_true),
    }
    if model is not None:
        trained_hat = torch.cat(trained_hat)
        result["trained"] = summarize(trained_hat, z_true)
        result["predictor_checkpoint"] = predictor_checkpoint
        result["predictor_type"] = ckpt["predictor_type"]
        result["predictor_epoch"] = ckpt["epoch"]
        gap = 1 - result["interpolation"]["cosine_sim_median"]
        result["gap_closed_by_training"] = (
            (result["trained"]["cosine_sim_median"] - result["interpolation"]["cosine_sim_median"]) / gap
            if gap > 1e-8 else None
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--source-checkpoint", default=None,
                         help="optional trained ProceduralWorldModel checkpoint -- when given, --features "
                              "is run through ITS encoder first (see dataset.py) before anything else here, "
                              "so ITS z is the encoder under test. Requires --source-config.")
    parser.add_argument("--source-config", default=None,
                         help="config.yaml alongside --source-checkpoint")
    parser.add_argument("--predictor-checkpoint", default=None,
                         help="optional trained predictor checkpoint (train_predictor.py's "
                              "best.ckpt/last.ckpt, any --predictor-type -- the checkpoint records "
                              "its own type/arch_kwargs); omit to report interpolation only")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = PlanningDataset(args.features, args.manifest, path_col=args.path_col,
                          checkpoint=args.source_checkpoint, config=args.source_config, device=args.device)
    print(f"{ds.n_clips} clips, feature_dim={ds.D}, T={ds.T}")

    metrics = compute_planning_dict(ds.features, device, predictor_checkpoint=args.predictor_checkpoint,
                                     batch_size=args.batch_size)
    result = {
        "features": args.features, "manifest": args.manifest,
        "source_checkpoint": args.source_checkpoint, "source_config": args.source_config,
        **metrics,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}")
    print(f"\ninterpolation: {result['interpolation']}")
    if "trained" in result:
        print(f"trained:       {result['trained']}")
        print(f"gap_closed_by_training: {result['gap_closed_by_training']}")
