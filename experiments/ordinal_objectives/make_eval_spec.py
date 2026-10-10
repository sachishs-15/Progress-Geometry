"""Turn this project's trained checkpoints into a spec that
../encoder_planning_eval/progress_metrics/run_progress_metrics_batch.py can run as-is.

The point of emitting a spec rather than writing an evaluator here: evaluation must not
be reimplemented in this folder. Every objective in this project has to be scored by the
SAME code that scored predictive_world_model's checkpoints and the raw frozen encoders,
or its row in the comparison table means something subtly different from the rows above
it. compute_progress_metrics.py already loads our checkpoints unmodified (see
train.py's save_checkpoint and tests/test_objectives.py's last test), so all that is
missing is the paperwork, which is this file.

Checkpoint/config paths in the emitted spec are relative to the progress_metrics
directory, matching the convention of its own
progress_metrics_experiments.example.json -- so the spec is run from there:

    python make_eval_spec.py --out ../encoder_planning_eval/progress_metrics/ordinal_val.json
    cd ../encoder_planning_eval/progress_metrics
    /home/sachish/.conda/envs/lift/bin/python run_progress_metrics_batch.py \
        --spec ordinal_val.json --results-dir results/ordinal_val \
        --csv-out results/ordinal_objectives_val.csv --workers 2

ALWAYS pass --results-dir, and a DIFFERENT one per split. The batch runner names each
row's per-clip json after the experiment, so two specs that share experiment names --
which train-split and validation-split specs necessarily do -- write to the same files
and the second silently overwrites the first. The summary CSV still comes out right,
because it is written from each subprocess's own return value, so nothing fails and
nothing warns; what is lost is the per-clip data, which is exactly what
analyze_results.py needs to get past the median's quantization. (Learned the hard way.)

Baselines are included by default (--no-baselines to drop them): the raw frozen DINOv2
features this project trains on top of, and predictive_world_model's own K400 checkpoint.
A table of objectives with no untrained baseline in it cannot answer the first question
anyone will ask of it, which is whether ANY of them beat doing nothing.
"""
import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROGRESS_METRICS_DIR = HERE.parent / "encoder_planning_eval" / "progress_metrics"

SSV2_VAL_FEATURES = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt"
SSV2_VAL_MANIFEST = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv"

BASELINES = [
    # Doing nothing at all: the frozen DINOv2 CLS features every objective here starts from.
    {"name": "baseline_raw_dinov2", "raw": "cls"},
    # The sibling project's own trained world model, for continuity with its results.
    {"name": "baseline_pwm_cls_mlp_k400",
     "checkpoint": "../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/best.ckpt",
     "config": "../../predictive_world_model/checkpoints/pwm_cls_mlp_k400/config.yaml"},
]


def last_epoch(run_dir: Path):
    """Last epoch in train_log.csv, or None if it has none yet."""
    log = run_dir / "train_log.csv"
    if not log.exists():
        return None
    rows = log.read_text().strip().splitlines()
    if len(rows) < 2:
        return None
    try:
        return int(rows[-1].split(",")[0])
    except ValueError:
        return None


def discover(checkpoints_dir: Path, ckpt_name: str, allow_partial: bool = False,
             pattern: str = "*"):
    """Every subdirectory holding a config.yaml, the requested checkpoint, AND a finished
    run.

    The completeness check is not optional bookkeeping. best.ckpt is written from epoch 0
    onward, so a run that is still training -- or that died halfway -- has one sitting on
    disk that looks exactly like a finished run's to anything that only globs for files.
    Tabulating it would put a half-trained encoder in the comparison under the same name
    as every fully-trained one, and nothing downstream could tell.
    """
    found = []
    for cfg in sorted(checkpoints_dir.glob(f"{pattern}/config.yaml")):
        run_dir = cfg.parent
        ckpt = run_dir / ckpt_name
        if not ckpt.exists():
            print(f"skipping {run_dir.name}: no {ckpt_name} (not started?)")
            continue
        want = yaml.safe_load(cfg.read_text()).get("epochs")
        got = last_epoch(run_dir)
        if not allow_partial and want is not None and (got is None or got < want - 1):
            print(f"skipping {run_dir.name}: epoch {got}/{want} -- still running or died "
                  f"(pass --allow-partial to include it anyway)")
            continue
        found.append((run_dir.name, ckpt, cfg))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--ckpt-name", default="best.ckpt", choices=["best.ckpt", "last.ckpt"])
    p.add_argument("--features", default=SSV2_VAL_FEATURES,
                   help="features to score on. Defaults to SSv2 VALIDATION -- the split "
                        "these configs train on is ssv2_train, and scoring ordinality on "
                        "the training split would reward memorising the 16-slot layout, "
                        "which is the exact confound crop_size exists to control for")
    p.add_argument("--manifest", default=SSV2_VAL_MANIFEST)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--allow-partial", action="store_true",
                   help="include runs that have not reached their configured epoch count")
    p.add_argument("--pattern", default="*",
                   help="glob restricting which checkpoints/ subdirectories are discovered "
                        "(e.g. 'ego_*' to pick only the Ego4D-GoalStep sweep out of a "
                        "checkpoints/ that also holds the SSv2 runs)")
    p.add_argument("--out", default=str(PROGRESS_METRICS_DIR / "ordinal_objectives.json"))
    args = p.parse_args()

    ckpt_dir = (HERE / args.checkpoints_dir).resolve()
    if not ckpt_dir.exists():
        raise SystemExit(f"{ckpt_dir} does not exist -- nothing trained yet")

    experiments = list(BASELINES) if not args.no_baselines else []
    for name, ckpt, cfg in discover(ckpt_dir, args.ckpt_name, args.allow_partial, args.pattern):
        experiments.append({
            "name": name,
            # Relative to the progress_metrics directory the spec is run from.
            "checkpoint": os_relpath(ckpt, PROGRESS_METRICS_DIR),
            "config": os_relpath(cfg, PROGRESS_METRICS_DIR),
        })

    spec = {"defaults": {"features": args.features, "manifest": args.manifest},
            "experiments": experiments}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {len(experiments)} experiments -> {out}")
    tag = out.stem
    print(f"\ncd {PROGRESS_METRICS_DIR}\n"
          f"/home/sachish/.conda/envs/lift/bin/python run_progress_metrics_batch.py \\\n"
          f"    --spec {out.name} --results-dir results/{tag} \\\n"
          f"    --csv-out results/{tag}_summary.csv --workers 2\n"
          f"# --results-dir is NOT optional when running more than one split: per-row json\n"
          f"# filenames are the experiment names, so specs would overwrite each other.")


def os_relpath(path: Path, start: Path) -> str:
    import os
    return os.path.relpath(path, start)


if __name__ == "__main__":
    main()
