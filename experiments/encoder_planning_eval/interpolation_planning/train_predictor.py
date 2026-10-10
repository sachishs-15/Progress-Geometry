"""Train a planning predictor for one encoder's precomputed features -- fixed
architecture and training hyperparameters across every encoder under test for a given
--predictor-type (see predictors/predictor.py's docstring and ../README.md for why that
fixedness is the point).

Frozen features only: nothing here touches or fine-tunes the encoder itself, matching the
task's "predict intermediate latents without training on actions [or the encoder]"
framing -- the only thing ever trained is the small predictor, filling in z[1:-1] from
z[0]/z[-1] alone.

--predictor-type selects the architecture (predictors/predictor.py's PREDICTOR_REGISTRY):
  mlp          -- independent feedforward regression per intermediate position.
  transformer  -- bidirectional mask-token infilling over the whole sequence (default;
                   mirrors LiFT's MaskedLatentPredictor, see predictors/predictor.py).
Only the flags relevant to the chosen type are actually used (the others are simply
ignored) -- same convention predictive_world_model/train.py uses for its cls-only vs.
patch-only flags.

Usage (raw encoder):
    python train_predictor.py --predictor-type transformer \\
        --train-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt \\
        --train-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \\
        --val-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt \\
        --val-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \\
        --checkpoint-dir checkpoints/dinov2_vits14_transformer

Usage (a trained ProceduralWorldModel checkpoint's own z instead of a raw encoder --
--source-checkpoint/--source-config run the SOURCE features through that checkpoint's own
encoder first, in memory, no separate extraction step -- see dataset.py):
    python train_predictor.py --predictor-type transformer \\
        --train-features .../dinov2_noreg/ssv2_validation.pt --train-manifest train_split.csv \\
        --val-features .../dinov2_noreg/ssv2_validation.pt --val-manifest val_split.csv \\
        --source-checkpoint ../predictive_world_model/checkpoints/pwm_patch_channel_k400_10pct/epoch39.ckpt \\
        --source-config ../predictive_world_model/checkpoints/pwm_patch_channel_k400_10pct/config.yaml \\
        --checkpoint-dir checkpoints/pwm_patch_channel_k400_10pct_epoch39_transformer
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PlanningDataset
from predictors.predictor import PREDICTOR_REGISTRY, build_predictor


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            z = batch["z"].to(device)  # (B, T, D)
            t_len = z.shape[1]
            z_pred = model(z[:, 0], z[:, -1], t_len)
            loss = F.l1_loss(z_pred[:, 1:-1], z[:, 1:-1])  # intermediate positions only
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--source-checkpoint", default=None,
                         help="optional trained ProceduralWorldModel checkpoint (predictive_world_model/"
                              "train.py's best.ckpt/last.ckpt) -- when given, --train-features/--val-features "
                              "are run through ITS encoder first (see dataset.py), so its own z is what gets "
                              "trained on here, not the raw features file directly. Requires --source-config.")
    parser.add_argument("--source-config", default=None,
                         help="config.yaml alongside --source-checkpoint")

    parser.add_argument("--predictor-type", choices=sorted(PREDICTOR_REGISTRY), default="transformer")
    # mlp-only
    parser.add_argument("--hidden", type=int, default=PREDICTOR_REGISTRY["mlp"][1]["hidden"])
    parser.add_argument("--depth", type=int, default=PREDICTOR_REGISTRY["mlp"][1]["depth"])
    parser.add_argument("--embed-dim", type=int, default=PREDICTOR_REGISTRY["mlp"][1]["embed_dim"])
    # transformer-only
    parser.add_argument("--nhead", type=int, default=PREDICTOR_REGISTRY["transformer"][1]["nhead"])
    parser.add_argument("--num-layers", type=int, default=PREDICTOR_REGISTRY["transformer"][1]["num_layers"])
    # shared
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PlanningDataset(args.train_features, args.train_manifest, path_col=args.path_col,
                                checkpoint=args.source_checkpoint, config=args.source_config, device=args.device)
    val_ds = PlanningDataset(args.val_features, args.val_manifest, path_col=args.path_col,
                              checkpoint=args.source_checkpoint, config=args.source_config, device=args.device)
    if train_ds.D != val_ds.D:
        raise SystemExit(f"train/val feature dims differ: {train_ds.D} vs {val_ds.D} -- "
                          f"--train-features and --val-features must be the same encoder")
    if train_ds.T != val_ds.T:
        raise SystemExit(f"train/val clip lengths differ: T={train_ds.T} vs T={val_ds.T}")
    print(f"Train: {train_ds.n_clips} clips. Val: {val_ds.n_clips} clips. "
          f"feature_dim={train_ds.D}, T={train_ds.T}, predictor_type={args.predictor_type}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Only pass the flags relevant to the chosen predictor_type (same convention
    # predictive_world_model/train.py uses for its cls-only vs. patch-only args) -- the
    # rest of PREDICTOR_REGISTRY[type][1]'s keys, pulled straight off parsed args.
    arch_keys = PREDICTOR_REGISTRY[args.predictor_type][1].keys()
    arch_kwargs = {k: getattr(args, k) for k in arch_keys}
    model = build_predictor(args.predictor_type, dim=train_ds.D, max_len=train_ds.T,
                             **arch_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.safe_dump({**vars(args), "feature_dim": train_ds.D, "max_len": train_ds.T}, f)

    log_path = ckpt_dir / "train_log.csv"
    best_val = float("inf")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_l1", "val_l1"])
        writer.writeheader()

        for epoch in range(args.epochs):
            train_l1 = run_epoch(model, train_loader, device, optimizer)
            val_l1 = run_epoch(model, val_loader, device, optimizer=None)
            writer.writerow({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})
            log_file.flush()

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch}: train_l1={train_l1:.4f}, val_l1={val_l1:.4f}")

            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "feature_dim": train_ds.D, "max_len": train_ds.T,
                        "predictor_type": args.predictor_type, "arch_kwargs": arch_kwargs},
                       ckpt_dir / "last.ckpt")
            if val_l1 < best_val:
                best_val = val_l1
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "feature_dim": train_ds.D, "max_len": train_ds.T,
                            "predictor_type": args.predictor_type, "arch_kwargs": arch_kwargs},
                           ckpt_dir / "best.ckpt")

    print(f"Done. Checkpoint + config + log -> {ckpt_dir}")
