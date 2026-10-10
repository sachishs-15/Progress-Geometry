"""Train ONE objective on frozen per-frame features and write a checkpoint that
experiments/encoder_planning_eval/ can score without modification.

Everything except the objective is held fixed -- encoder (pwm_bridge.ClsMlpEncoder),
optimizer, schedule, data, split, and evaluation -- so a difference between two runs is
attributable to the objective (or its hyperparameters) and nothing else. That is the
entire design constraint of this project; see README.md.

    ./train.sh configs/rank_n_contrast.yaml
    python train.py --config configs/tcc.yaml --checkpoint-dir checkpoints/tcc_t0.1 \
        --objective-kwargs '{"temperature": 0.1}'

Config keys map 1:1 onto the flags below (unknown keys are an error, not a silent
ignore), with two nested exceptions:

  objective_kwargs:  passed to the selected objective's constructor. Validated against
                     that class's own signature, so a typo names the valid keys.
  aux_weights:       weights for the geometric regularizers borrowed from
                     predictive_world_model/losses/world_model_loss.py -- straighten,
                     chord, vicreg_std, vicreg_cov, sigreg. Composable with ANY objective,
                     which is how this project tests "does objective X need an
                     anti-collapse term, or does it resist collapse on its own?".

Checkpoint selection. A training loss is not comparable across objectives (VIP's is a
log-mean-exp of distances, cross-entropy is nats, ranking is a margin), so "best" cannot
mean "lowest loss" here the way it does in predictive_world_model/train.py. With
--eval-every > 0 (the default), best.ckpt is instead selected on an actual ordinality
metric computed on held-out clips with LiFT's own clip_progress_metrics -- the same
function the final comparison table uses -- so model selection and evaluation agree.
Set --eval-every 0 to fall back to loss-based selection.
"""
import argparse
import csv
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_data import ObjectiveClipDataset, deterministic_subset, progress_from_batch
from objectives import OBJECTIVE_REGISTRY
from regularizers import ordinal_entropy
from pwm_bridge import ClsMlpEncoder, FeatureProvider, WorldModelLoss, load_clip_progress_metrics

#: Auxiliary regularizer name -> the WorldModelLoss method that computes it. Keys are what
#: a config's `aux_weights:` may contain; anything else is a config error.
AUX_LOSSES = {
    "straighten": "straightening_loss",
    "chord": "chord_loss",
    "vicreg_std": "vicreg_std_loss",
    "vicreg_cov": "vicreg_cov_loss",
}
#: Regularizers of this project's own that need the progress LABEL, which none of
#: WorldModelLoss's terms ever see (see regularizers.py). Weighted independently because
#: the paper they come from has a separate lambda for each.
LABEL_AUX = ("ordinal_entropy_d", "ordinal_entropy_t")
_SCALARS = {"sigreg_num_slices"}

#: Architecture keys that must appear in the written config.yaml for
#: encoder_planning_eval/progress_metrics/compute_progress_metrics.py to rebuild a
#: matching encoder from it (its own ARCH_KEYS). token_mode is pinned to 'cls' because
#: that is the only encoder this project trains.
EVAL_ARCH_DEFAULTS = {"token_mode": "cls", "projector_type": "global"}


class _Tee:
    """Duplicates writes to every given stream. Copied from
    predictive_world_model/train.py (see its docstring for why the log is owned here
    rather than by an external `tee` in train.sh)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _has_training_artifacts(ckpt_dir: Path) -> bool:
    markers = ("config.yaml", "train_log.csv")
    return any((ckpt_dir / m).exists() for m in markers) or any(ckpt_dir.glob("*.ckpt"))


def resolve_checkpoint_dir(base: Path) -> Path:
    """base if unused, else base_1/base_2/... Same policy (and same reasoning) as
    predictive_world_model/train.py's: there is no --resume, so a directory with training
    artifacts in it can only be a previous run's leftovers."""
    ckpt_dir = base
    i = 1
    while _has_training_artifacts(ckpt_dir):
        ckpt_dir = base.with_name(f"{base.name}_{i}")
        i += 1
    return ckpt_dir


def build_objective(name, latent_dim, kwargs):
    if name not in OBJECTIVE_REGISTRY:
        raise SystemExit(f"unknown objective {name!r}. Available: {sorted(OBJECTIVE_REGISTRY)}")
    cls = OBJECTIVE_REGISTRY[name]
    valid = set(inspect.signature(cls.__init__).parameters) - {"self", "latent_dim"}
    unknown = set(kwargs) - valid
    if unknown:
        raise SystemExit(f"objective_kwargs for {name!r} has unknown key(s) {sorted(unknown)}. "
                         f"Valid: {sorted(valid)}")
    return cls(latent_dim=latent_dim, **kwargs)


def build_aux(aux_weights):
    """A WorldModelLoss used purely as a bag of regularizer functions (its own forward is
    never called -- the prediction term belongs to objectives/next_step.py)."""
    valid = set(AUX_LOSSES) | set(LABEL_AUX) | {"sigreg"} | _SCALARS
    unknown = set(aux_weights) - valid
    if unknown:
        raise SystemExit(f"aux_weights has unknown key(s) {sorted(unknown)}. "
                         f"Valid: {sorted(valid)}")
    return WorldModelLoss(sigreg_num_slices=int(aux_weights.get("sigreg_num_slices", 1024)))


def aux_terms(criterion, aux_weights, z, progress):
    """Sum of weighted regularizers plus their individual detached values."""
    total, parts = None, {}
    # Computed once even if both weights are set -- the two terms share the per-label
    # centre computation, which is the expensive part.
    ent = None
    if any(aux_weights.get(k) for k in LABEL_AUX):
        ent = dict(zip(LABEL_AUX, ordinal_entropy(z, progress)))
    for name, weight in aux_weights.items():
        if name in _SCALARS or not weight:
            continue
        if name in LABEL_AUX:
            value = ent[name]
        elif name == "sigreg":
            from losses.sigreg_loss import sigreg_loss  # predictive_world_model's, via pwm_bridge's path
            value = sigreg_loss(z, num_slices=criterion.sigreg_num_slices)
        else:
            value = getattr(criterion, AUX_LOSSES[name])(z)
        total = value * weight if total is None else total + value * weight
        parts[f"aux_{name}"] = value.detach()
    return total, parts


def run_epoch(encoder, objective, criterion, aux_weights, loader, device, optimizer=None,
              grad_clip=0.0, desc="", progress_every=0, amp=False):
    """One pass over `loader`. optimizer=None -> evaluation (no grad, no step).

    Metric tensors stay on-device all epoch and are converted to Python floats exactly
    once at the end, same reason as predictive_world_model/train.py's run_epoch: an
    .item() per term per step forces a host<->device sync that serializes the GPU."""
    train = optimizer is not None
    encoder.train(train)
    objective.train(train)
    totals, n_batches = {}, 0
    total_batches = len(loader)
    amp_enabled = amp and device.type == "cuda"
    with torch.set_grad_enabled(train):
        for batch in loader:
            data = progress_from_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                z = encoder(data["features"])
                z_pair = encoder(data["features_pair"]) if "features_pair" in data else None
                loss, metrics = objective(
                    z, data["progress"], frame_index=data.get("frame_index"),
                    z_pair=z_pair, progress_pair=data.get("progress_pair"),
                    frame_index_pair=data.get("frame_index_pair"),
                )
                metrics = {f"obj_{k}": v for k, v in metrics.items()}
                metrics["objective"] = loss.detach()
                aux, aux_parts = aux_terms(criterion, aux_weights, z, data["progress"])
                if aux is not None:
                    loss = loss + aux
                    metrics.update(aux_parts)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    params = list(encoder.parameters()) + list(objective.parameters())
                    metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, grad_clip).detach()
                optimizer.step()
            metrics["total"] = loss.detach()
            # Diagnostic, not part of the objective -- several losses here are
            # scale-sensitive (TCN's margin, VIP's distances) while others are not, so
            # without this the curves can't distinguish "the loss moved" from "||z|| moved".
            metrics["z_norm"] = z.detach().norm(dim=-1).mean()
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
            if progress_every > 0 and (n_batches % progress_every == 0 or n_batches == total_batches):
                avg = {k: (v / n_batches).item() for k, v in totals.items()}
                print(f"  [{desc}] batch {n_batches}/{total_batches} "
                      f"({100 * n_batches / total_batches:.1f}%): "
                      + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()), flush=True)
    return {k: (v / n_batches).item() for k, v in totals.items()}


@torch.no_grad()
def evaluate_ordinality(encoder, features, device, batch_size=256):
    """Median P-JEPA-style ordinality metrics over `features` (M, T, D), via LiFT's own
    clip_progress_metrics -- literally the same function
    encoder_planning_eval/progress_metrics/compute_progress_metrics.py reports, so an
    in-training monitor value and a final table value mean the same thing.

    Runs on full, uncropped clips regardless of --crop-size: the metric is defined on a
    whole trajectory, and measuring it on random crops would confound "the space got more
    ordinal" with "this epoch happened to draw easier crops"."""
    clip_progress_metrics = load_clip_progress_metrics()
    encoder.eval()
    per_clip = []
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start:start + batch_size].to(device)
        z = encoder(chunk).float().cpu().numpy()
        for zi in z:
            m = clip_progress_metrics(zi)
            if m is not None:
                per_clip.append(m)
    if not per_clip:
        return {}
    keys = per_clip[0].keys()
    return {k: float(np.median([m[k] for m in per_clip])) for k in keys}


def build_monitor_features(provider, manifest, path_col, limit, seed):
    """Fixed set of full-length clips for the in-training monitor. Materialized once, in
    RAM, in a deterministic order -- a resampled subset each epoch would make the monitor
    curve partly noise about which clips were drawn."""
    ds = ObjectiveClipDataset.from_manifest(manifest, provider, path_col=path_col, crop_size=0)
    idx = deterministic_subset(len(ds), limit, seed=seed)
    return torch.stack([ds[int(i)]["features"] for i in idx])


def save_checkpoint(path, encoder, objective, epoch, objective_name, extra=None):
    """Written in exactly the layout compute_progress_metrics.py expects.

    "model" holds ONLY the encoder's weights, prefixed `encoder.` -- which is what that
    script gets when it rebuilds a ProceduralWorldModel and drops everything starting
    with `predictor.`: the remaining keys line up because pwm_bridge imports the very
    same ClsMlpEncoder class. The objective's own head goes in a separate "head" entry
    that the evaluator never looks at, so a head can be any shape it likes without
    breaking evaluation (and so nothing about head capacity can leak into the scored
    model)."""
    torch.save({
        "model": {f"encoder.{k}": v for k, v in encoder.state_dict().items()},
        "head": objective.state_dict(),
        "epoch": epoch,
        "objective": objective_name,
        # Travels with the checkpoint because it is needed to READ the result: head size
        # varies from 0 to 3.5M across objectives (see heads.py), and a big head is a way
        # to satisfy a loss without moving the encoder at all.
        "head_params": sum(p.numel() for p in objective.parameters()),
        **(extra or {}),
    }, path)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="yaml overriding every default below")

    p.add_argument("--objective", default="progress_regression",
                   choices=sorted(OBJECTIVE_REGISTRY),
                   help="which training objective to compare (see objectives/__init__.py)")
    p.add_argument("--objective-kwargs", default={}, type=json.loads,
                   help="JSON dict of the objective's own hyperparameters; normally set as "
                        "a nested `objective_kwargs:` block in --config, with this flag for "
                        "one-off sweep overrides")
    p.add_argument("--aux-weights", default={}, type=json.loads,
                   help="JSON dict of geometric regularizer weights (straighten, chord, "
                        "vicreg_std, vicreg_cov, sigreg), composable with any objective")

    p.add_argument("--train-manifest", required=False, default=None,
                   help="CSV with a video_path column; omit to train on every video in --features")
    p.add_argument("--val-manifest", default=None)
    p.add_argument("--path-col", default="video_path")
    p.add_argument("--pair-col", default=None,
                   help="manifest column to draw same-activity clip pairs from (`verb` for "
                        "SSv2). Required by tcc(view='pair') / scl(view='pair'); ignored "
                        "otherwise, at the cost of a second feature lookup per item")
    p.add_argument("--features", default=None, help="precomputed bulk .pt (LiFT format)")
    p.add_argument("--val-features", default=None,
                   help="separate precomputed .pt for the val manifest; defaults to --features")
    p.add_argument("--live-cache-dir", default=None)

    p.add_argument("--crop-size", type=int, default=0,
                   help="0 (default): train on whole n_frames clips. >0: a random contiguous "
                        "span of this many frames per example -- the control for whether an "
                        "objective is learning progress or learning the 16-slot layout "
                        "(see clip_data.py / README.md's 'The crop confound')")
    p.add_argument("--crop-label", choices=["global", "local"], default="global",
                   help="global: labels stay on the video's clock. local: renormalized to "
                        "[0,1] within the crop")

    p.add_argument("--feature-dim", type=int, default=384)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--use-projector", dest="use_projector", action="store_true", default=True)
    p.add_argument("--no-projector", dest="use_projector", action="store_false")
    p.add_argument("--norm-type", choices=["layernorm", "clip_rmsnorm", "batchnorm", "none"],
                   default="layernorm",
                   help="encoder output normalization (predictive_world_model/models/norms.py)")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--model-id", default="vit_small_patch14_reg4_dinov2.lvd142m")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=None,
                   help="separate LR for the objective's head; defaults to --lr. Worth "
                        "raising for the sequence-head objectives, whose head starts from "
                        "scratch while the encoder starts from a good frozen feature")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0,
                   help="also save a standalone epoch{N}.ckpt every N epochs; 0 disables")
    p.add_argument("--eval-every", type=int, default=5,
                   help="compute held-out ordinality metrics every N epochs and select "
                        "best.ckpt on --monitor-metric; 0 disables (best.ckpt then falls "
                        "back to lowest loss, which is NOT comparable across objectives)")
    p.add_argument("--eval-clips", type=int, default=512,
                   help="how many held-out clips the in-training monitor scores")
    p.add_argument("--monitor-source", choices=["train", "val"], default="train",
                   help="which split the monitor clips come from. 'train' (default) keeps "
                        "checkpoint selection OFF the split the final comparison table is "
                        "scored on -- make_eval_spec.py reports on ssv2_validation, so "
                        "selecting on validation would mean every row was chosen by peeking "
                        "at its own test metric, and would flatter noisier objectives more "
                        "than stable ones. 'val' is available if you would rather select on "
                        "held-out data, but then report last.ckpt in the table, not best.ckpt.")
    p.add_argument("--monitor-metric", default="time_progress_spearman",
                   help="clip_progress_metrics key to select best.ckpt on")
    p.add_argument("--monitor-mode", choices=["max", "min"], default="max")

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in p._actions}
        unknown = set(cfg) - valid
        if unknown:
            raise SystemExit(f"{pre_args.config}: unknown config key(s) {sorted(unknown)}. "
                             f"Valid keys: {sorted(valid - {'help'})}")
        p.set_defaults(**cfg)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint_dir:
        raise SystemExit("--checkpoint-dir is required (pass it, or set checkpoint_dir: in --config)")
    if not args.train_manifest and not args.features:
        raise SystemExit("pass --train-manifest, --features, or both")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # The dataset draws crops/pairs from the module-level `random` (see clip_data.py's
    # "Randomness"). DataLoader reseeds that per worker per epoch from a base seed derived
    # from the torch generator seeded above; this call covers the num_workers=0 path,
    # where __getitem__ runs in this process and nothing else would seed it.
    random.seed(args.seed)
    device = torch.device(args.device)
    ckpt_dir = resolve_checkpoint_dir(Path(args.checkpoint_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(ckpt_dir / "train.log", "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    sys.stderr = _Tee(sys.__stderr__, log_handle)
    print(f"Logging to {ckpt_dir / 'train.log'}")
    if str(ckpt_dir) != args.checkpoint_dir:
        print(f"--checkpoint-dir {args.checkpoint_dir} already has content -- using {ckpt_dir}")

    provider = FeatureProvider(
        features_path=args.features, live_cache_dir=args.live_cache_dir or str(ckpt_dir / "live_cache"),
        n_frames=args.n_frames, model_id=args.model_id, device=args.device, token_mode="cls",
    )
    val_provider = provider
    if args.val_features and args.val_features != args.features:
        val_provider = FeatureProvider(
            features_path=args.val_features,
            live_cache_dir=args.live_cache_dir or str(ckpt_dir / "live_cache"),
            n_frames=args.n_frames, model_id=args.model_id, device=args.device, token_mode="cls",
        )

    objective = build_objective(args.objective, args.latent_dim, dict(args.objective_kwargs))
    pair_col = args.pair_col
    if getattr(objective, "requires_pair", False) and not pair_col:
        raise SystemExit(f"objective {args.objective!r} with these kwargs needs paired clips -- "
                         f"pass --pair-col (e.g. `verb` for SSv2's manifests)")
    if getattr(objective, "requires_pair", False) and not args.train_manifest:
        raise SystemExit(f"objective {args.objective!r} needs paired clips, which are drawn from a "
                         f"manifest column -- pass --train-manifest as well as --pair-col")
    if pair_col and not getattr(objective, "requires_pair", False):
        # Pairing costs a second feature lookup per item and drops singleton groups from
        # the split; skip it rather than paying for something unused.
        print(f"note: --pair-col {pair_col!r} is set but {args.objective!r} does not use pairs "
              f"-- pairing disabled for this run")

    ds_kwargs = dict(path_col=args.path_col, pair_col=pair_col if getattr(objective, "requires_pair", False) else None,
                     crop_size=args.crop_size, crop_label=args.crop_label)
    if args.train_manifest:
        train_ds = ObjectiveClipDataset.from_manifest(args.train_manifest, provider, **ds_kwargs)
    else:
        train_ds = ObjectiveClipDataset(provider.video_paths, provider, crop_size=args.crop_size,
                                        crop_label=args.crop_label)
    print(f"Train set: {len(train_ds)} clips")

    val_loader = None
    if args.val_manifest:
        val_ds = ObjectiveClipDataset.from_manifest(args.val_manifest, val_provider, **ds_kwargs)
        print(f"Val set: {len(val_ds)} clips")

    encoder = ClsMlpEncoder(feature_dim=args.feature_dim, latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim, use_projector=args.use_projector,
                            norm_type=args.norm_type).to(device)
    objective = objective.to(device)
    criterion = build_aux(dict(args.aux_weights))
    param_groups = [{"params": encoder.parameters(), "lr": args.lr}]
    head_params = list(objective.parameters())
    if head_params:
        # `is not None`, not `or`: head_lr=0.0 is a legitimate way to freeze the head and
        # train the encoder through a fixed random readout, which `or` would silently turn
        # back into --lr. Same convention as predictive_world_model's encoder_lr/predictor_lr.
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        param_groups.append({"params": head_params, "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    n_head = sum(p.numel() for p in head_params)
    print(f"Objective {args.objective!r}: {n_head} head params, "
          f"encoder {sum(p.numel() for p in encoder.parameters())} params")

    loader_kwargs = {"multiprocessing_context": "spawn"} if args.num_workers > 0 else {}
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin_memory, **loader_kwargs)
    if args.val_manifest:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin_memory, **loader_kwargs)

    monitor_features = None
    if args.eval_every > 0:
        if args.monitor_source == "val" and args.val_manifest:
            monitor_manifest, monitor_provider = args.val_manifest, val_provider
        else:
            monitor_manifest, monitor_provider = args.train_manifest, provider
        if monitor_manifest is None:
            print("note: --eval-every > 0 but no manifest to draw monitor clips from; "
                  "falling back to loss-based checkpoint selection")
            args.eval_every = 0
        else:
            monitor_features = build_monitor_features(
                monitor_provider, monitor_manifest, args.path_col, args.eval_clips, args.seed)
            print(f"Ordinality monitor: {monitor_features.shape[0]} full clips from "
                  f"{monitor_manifest} (--monitor-source {args.monitor_source}), selecting on "
                  f"{args.monitor_metric} ({args.monitor_mode})")

    cfg_out = dict(vars(args))
    cfg_out.update(EVAL_ARCH_DEFAULTS)
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_out, f)

    log_path, log_file, writer = ckpt_dir / "train_log.csv", None, None
    best = None
    print(f"Training started: {args.epochs} epochs, {len(train_loader)} batches/epoch, "
          f"device={device}", flush=True)

    for epoch in range(args.epochs):
        train_metrics = run_epoch(encoder, objective, criterion, args.aux_weights, train_loader,
                                  device, optimizer, grad_clip=args.grad_clip,
                                  desc=f"epoch {epoch} train", progress_every=args.progress_every,
                                  amp=args.amp)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        monitor_value = train_metrics["total"]
        better = lambda new, old: old is None or new < old

        if val_loader is not None:
            val_metrics = run_epoch(encoder, objective, criterion, args.aux_weights, val_loader,
                                    device, optimizer=None, desc=f"epoch {epoch} val",
                                    progress_every=args.progress_every, amp=args.amp)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            monitor_value = val_metrics["total"]

        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs - 1):
            ord_metrics = evaluate_ordinality(encoder, monitor_features, device, args.batch_size)
            row.update({f"ord_{k}": v for k, v in ord_metrics.items()})
            if args.monitor_metric not in ord_metrics:
                raise SystemExit(f"--monitor-metric {args.monitor_metric!r} is not one of "
                                 f"{sorted(ord_metrics)}")
            monitor_value = ord_metrics[args.monitor_metric]
            better = (lambda new, old: old is None or new > old) if args.monitor_mode == "max" \
                else (lambda new, old: old is None or new < old)
            save_checkpoint(ckpt_dir / "last.ckpt", encoder, objective, epoch, args.objective,
                            extra={"ordinality": ord_metrics})
            if better(monitor_value, best):
                best = monitor_value
                save_checkpoint(ckpt_dir / "best.ckpt", encoder, objective, epoch, args.objective,
                                extra={"ordinality": ord_metrics})
        else:
            save_checkpoint(ckpt_dir / "last.ckpt", encoder, objective, epoch, args.objective)
            if args.eval_every == 0 and better(monitor_value, best):
                best = monitor_value
                save_checkpoint(ckpt_dir / "best.ckpt", encoder, objective, epoch, args.objective)

        if writer is None:
            log_file = open(log_path, "w", newline="")
            writer = csv.DictWriter(log_file, fieldnames=list(row), extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        log_file.flush()
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print("epoch {}: ".format(epoch)
                  + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch{epoch}.ckpt", encoder, objective, epoch, args.objective)

    if log_file is not None:
        log_file.close()
    print(f"Done. best {args.monitor_metric if args.eval_every > 0 else 'loss'} = {best}")
    print(f"Checkpoints + config + log -> {ckpt_dir}")


if __name__ == "__main__":
    main()
