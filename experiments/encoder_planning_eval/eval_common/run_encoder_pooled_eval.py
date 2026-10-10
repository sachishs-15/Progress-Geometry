"""Evaluates a checkpoint's own ENCODER output z = model.encoder(x) -- BEFORE the trained
AggregationHead -- instead of model.aggregate(z)'s learned MLP, fully in memory, no
intermediate --features .pt file ever written for this representation (only the final
per-checkpoint/per-pooling metrics JSON is saved, same as every other run in this
project). Two checkpoint families, two pooling vocabularies:
  - token_mode='patch', projector_type='channel': z is (B,T,P,D), a per-patch grid; pool
    over the patch dimension with --poolings mean/max/concat.
  - token_mode='cls'/'raw' with cls_agg_head=true: z is already (B,T,D), one token/frame
    (this checkpoint family's AggregationHead has num_patches=1 -- it doesn't do any
    spatial reduction, so bypassing it doesn't need one either); use --poolings raw,
    which passes z through unchanged.

Computes z ONCE per split (train/val) and derives all requested poolings from that same
forward pass, then runs progress_metrics + interpolation_planning (+ representation_probes,
if --train-features/--train-manifest given) against each pooled representation via the
compute_metrics_dict/compute_planning_dict/compute_probes_dict functions those scripts
already expose, unmodified from their normal --checkpoint/--config CLI path (only the
final json.dump target differs: one file per pooling mode here vs. one per CLI call there).

Usage (patch/channel):
    python run_encoder_pooled_eval.py --checkpoint .../best.ckpt --config .../config.yaml \
        --val-features .../dinov2_noreg/ssv2_validation.pt --val-manifest ssv2_validation.csv \
        --train-features .../dinov2_noreg/ssv2_train.pt --train-manifest ssv2_train.csv \
        --dataset ssv2 --out-dir results/encoder_pooled/<checkpoint_name>

Usage (cls/raw, cls_agg_head=true):
    python run_encoder_pooled_eval.py --checkpoint .../best.ckpt --config .../config.yaml \
        --val-features .../dinov2/ssv2_validation.pt --val-manifest ssv2_validation.csv \
        --train-features .../dinov2/ssv2_train.pt --train-manifest ssv2_train.csv \
        --dataset ssv2 --poolings raw --out-dir results/encoder_pooled/<checkpoint_name>
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PWM_ROOT = Path(__file__).resolve().parents[2] / "predictive_world_model"
EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(EVAL_ROOT / "progress_metrics"))
sys.path.insert(0, str(EVAL_ROOT / "representation_probes"))
sys.path.insert(0, str(EVAL_ROOT / "interpolation_planning"))

from eval_common.loading import load_world_model, load_matched_features  # noqa: E402
from compute_progress_metrics import compute_metrics_dict  # noqa: E402
from compute_probes import compute_probes_dict, prepare_label_splits, DATASETS  # noqa: E402
from eval_planning import compute_planning_dict  # noqa: E402


def encode_pooled(model, feats, device, batch_size, poolings):
    """z = model.encoder(x) ONCE per batch, then all requested poolings derived from
    that same forward pass -- BEFORE model.aggregate() (i.e. before the trained
    AggregationHead, whether that's patch/channel's spatial one or a cls_agg_head=True
    checkpoint's trivial num_patches=1 one). feats: (N,T,P,D) raw patch tokens for
    token_mode='patch', (N,T,D) for token_mode='cls', or (N,T,3,H,W) raw pixel frames for
    token_mode='raw' (run through model.pixel_encoder first, same order as
    ProceduralWorldModel.forward() -- see dataset.py's _run_checkpoint_encoder, which this
    mirrors) -- cls/raw already has one token/frame once encoded, so mean/max/concat are
    all identical to z itself; "raw" is the one pooling name valid there, requested
    explicitly (not inferred) so a caller is never silently handed the same tensor under
    three different names."""
    out = {p: [] for p in poolings}
    with torch.no_grad():
        for s in tqdm(range(0, len(feats), batch_size), leave=False):
            x = feats[s:s + batch_size].to(device)
            if model.pixel_encoder is not None:
                x = model.pixel_encoder(x)
            z = model.encoder(x)  # (B, T, P, out_dim) patch mode, or (B, T, out_dim) cls/raw
            if z.dim() == 3:
                if set(poolings) - {"raw"}:
                    raise SystemExit(f"model.encoder(x) is already (B,T,D) for this checkpoint "
                                      f"(token_mode={model.token_mode!r}, one token/frame) -- "
                                      f"mean/max/concat would pool over the feature dimension "
                                      f"itself, not a patch dimension. Pass --poolings raw.")
                out["raw"].append(z.cpu())
                continue
            if "mean" in out:
                out["mean"].append(z.mean(dim=2).cpu())
            if "max" in out:
                out["max"].append(z.max(dim=2).values.cpu())
            if "concat" in out:
                b, t, pn, d = z.shape
                out["concat"].append(z.reshape(b, t, pn * d).cpu())
    return {p: torch.cat(chunks, dim=0) for p, chunks in out.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--train-features", default=None, help="omit to skip representation_probes")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--dataset", choices=list(DATASETS), default="ssv2")
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--poolings", nargs="+", default=["mean", "max", "concat"],
                         choices=["mean", "max", "concat", "raw"],
                         help="mean/max/concat: token_mode=patch,projector_type=channel only "
                              "(pools over the patch dimension). raw: token_mode=cls/raw with "
                              "cls_agg_head=true only (model.encoder(x) is already one "
                              "token/frame, used as-is -- mutually exclusive with the other three).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-val-per-class", type=int, default=5)
    parser.add_argument("--max-train-per-class", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--chiral-clf", choices=["ridge", "logistic", "kernel_svm"], default="ridge")
    parser.add_argument("--edge-frames", type=int, default=4)
    parser.add_argument("--tcc-pairs", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, epoch = load_world_model(args.checkpoint, args.config, device)
    model.eval()
    is_patch_channel = cfg.get("token_mode") == "patch" and cfg.get("projector_type") == "channel"
    is_cls_agghead = cfg.get("token_mode") in ("cls", "raw") and cfg.get("cls_agg_head")
    if not (is_patch_channel or is_cls_agghead):
        raise SystemExit(f"expected token_mode=patch,projector_type=channel (mean/max/concat "
                          f"pooling) or token_mode=cls/raw,cls_agg_head=true (raw pooling); got "
                          f"token_mode={cfg.get('token_mode')!r}, projector_type={cfg.get('projector_type')!r}, "
                          f"cls_agg_head={cfg.get('cls_agg_head')!r}")
    if is_patch_channel and set(args.poolings) - {"mean", "max", "concat"}:
        raise SystemExit("token_mode=patch,projector_type=channel only supports --poolings mean/max/concat")
    if is_cls_agghead and args.poolings != ["raw"]:
        raise SystemExit("token_mode=cls/raw,cls_agg_head=true only supports --poolings raw "
                          "(model.encoder(x) is already one token/frame)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- validation split: encode once, all poolings, used by progress_metrics + planning ---
    va_df = pd.read_csv(args.val_manifest, dtype=str)
    va_feats, va_matched = load_matched_features(args.val_features, va_df, args.path_col)
    print(f"[val] {len(va_feats)} clips, encoding with {args.poolings} pooling(s)...")
    va_pooled = encode_pooled(model, va_feats, device, args.batch_size, args.poolings)

    # --- train split (only if probes requested): same, once, all poolings ---
    tr_pooled, tr_df_split, tr_full_split, classes, dropped = None, None, None, None, None
    if args.train_features:
        spec = DATASETS[args.dataset]
        tr_df_split, va_df_split, tr_full_split, classes, dropped = prepare_label_splits(
            args.train_manifest, args.val_manifest, spec,
            args.min_train_per_class, args.min_val_per_class, args.max_train_per_class, args.seed,
        )
        print(f"[probes] {len(classes)} classes kept, dropped {dropped}; "
              f"train {len(tr_df_split)} clips, val {len(va_df_split)} clips")
        tr_feats, tr_matched = load_matched_features(args.train_features, tr_df_split, args.path_col)
        print(f"[train] {len(tr_feats)} clips, encoding with {args.poolings} pooling(s)...")
        tr_pooled = encode_pooled(model, tr_feats, device, args.batch_size, args.poolings)
        # raw DINOv2 per-frame reconstruction target, same convention as get_latents():
        # mean over the patch dim for a 4D (N,T,P,D) --features file, used as-is if it's
        # already (N,T,D) (cls/raw mode's --val-features/--train-features are plain CLS
        # features, no patch dim to pool). None for a 5D (N,T,3,H,W) raw-pixel --features
        # file -- there's no DINOv2 target to reconstruct back to for a token_mode='raw'
        # checkpoint trained directly on pixels (same guard as get_latents()'s
        # `feats.dim() in (3, 4)`); compute_probes_dict skips reconstruction_r2 when the
        # target is None. Without this guard the full (N,T,3,H,W) pixel tensor gets used
        # as the "target" and mlp_reconstruction_r2 OOMs trying to normalize it.
        tgt_tr = None if tr_feats.dim() == 5 else (tr_feats.mean(dim=2) if tr_feats.dim() == 4 else tr_feats).numpy()
        tr_full_feats, tr_full_matched = load_matched_features(args.train_features, tr_full_split, args.path_col)
        print(f"[train_full] {len(tr_full_feats)} clips (uncapped, for chirality), encoding...")
        tr_full_pooled = encode_pooled(model, tr_full_feats, device, args.batch_size, args.poolings)
        tgt_va = None if va_feats.dim() == 5 else (va_feats.mean(dim=2) if va_feats.dim() == 4 else va_feats).numpy()

    for pooling in args.poolings:
        print(f"\n=== pooling={pooling} ===")
        z_val = va_pooled[pooling]  # (N, T, D')

        pm = compute_metrics_dict(z_val.numpy(), va_matched)
        pm_out = {"checkpoint": args.checkpoint, "checkpoint_epoch": epoch,
                  "val_features": args.val_features, "pooling": pooling, **pm}
        (out_dir / f"progress_{pooling}.json").write_text(json.dumps(pm_out, indent=2))
        print(f"progress_metrics -> effective_rank_fraction={pm['effective_rank_fraction']:.4f}  "
              f"straightness={pm['summary_mean']['trajectory_straightness_raw']:.4f}")

        pl = compute_planning_dict(z_val, device)
        pl_out = {"checkpoint": args.checkpoint, "val_features": args.val_features,
                  "pooling": pooling, "interpolation": pl["interpolation"], **{k: v for k, v in pl.items() if k != "interpolation"}}
        (out_dir / f"planning_{pooling}.json").write_text(json.dumps(pl_out, indent=2))
        print(f"interpolation_planning -> R2={pl['interpolation']['r2']:.4f}")

        if tr_pooled is not None:
            z_tr, z_va2 = tr_pooled[pooling].numpy(), z_val.numpy()
            z_full = tr_full_pooled[pooling].numpy()
            pr = compute_probes_dict(
                z_tr, tgt_tr, tr_df_split, z_va2, tgt_va, va_df_split, classes, dropped,
                args, spec, device, encoder_label=f"{args.checkpoint}:encoder_only:{pooling}", epoch=epoch,
                z_full=z_full, tr_full=tr_full_split,
            )
            (out_dir / f"probes_{pooling}.json").write_text(json.dumps(pr, indent=2))
            print(f"representation_probes -> content.linear.balanced_acc="
                  f"{pr['probes']['content']['linear']['balanced_acc']:.4f}  "
                  f"chirality={pr['chirality'].get('concat_probe', {}).get('mean_accuracy', 'N/A')}")

    print(f"\nAll results -> {out_dir}/")
