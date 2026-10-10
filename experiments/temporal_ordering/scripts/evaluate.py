"""Aggregate runs/<experiment>/metrics.json into the final results table (section 17) and
generate the qualitative PCA + distance-vs-progress plots (sections 14-15). Usage:
    python scripts/evaluate.py --experiments baseline timestamp shuffle_k8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import CachedFeatureDataset  # noqa: E402
from data.splits import get_or_create_splits  # noqa: E402
from evaluation.latent_geometry import distance_progress_analysis  # noqa: E402
from models.encoder import FeatureEncoder  # noqa: E402
from train.train_progress_probe import build_video_data  # noqa: E402
from utils.config import load_config  # noqa: E402
from visualization.plot_latent import plot_distance_vs_progress, plot_pca_trajectories  # noqa: E402

KEY_METRICS = ["test_spearman", "test_pairwise_ordering_accuracy", "test_distance_progress_spearman"]


def load_metrics_table(runs_dir: str, experiments: list) -> list:
    rows = []
    for name in experiments:
        with open(Path(runs_dir) / name / "metrics.json") as f:
            m = json.load(f)
        row = {"experiment": name, "representation": m["representation"]}
        for split in ("val", "test"):
            for k, v in m[split].items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    return rows


def build_test_representations(runs_dir: str, experiment_name: str, device: str):
    cfg = load_config(str(Path(runs_dir) / experiment_name / "config.yaml"))
    dataset = CachedFeatureDataset(cfg["dataset"]["cache_dir"])
    splits = get_or_create_splits(
        cfg.get("splits_dir", "splits"), dataset,
        train=cfg["split"]["train"], val=cfg["split"]["val"], test=cfg["split"]["test"],
        seed=cfg["split"].get("seed", 42),
    )
    representation = cfg["probe"].get("representation", "raw")
    encoder = None
    if representation == "encoder":
        encoder = FeatureEncoder.load(str(Path(runs_dir) / experiment_name / "encoder.pt"), device=device)
        encoder.eval()
    progress_type = cfg["dataset"].get("progress_type", "timestamp")

    data = build_video_data(dataset, sorted(splits["test"]), progress_type, representation, encoder, device)
    reprs = {v: z for v, (z, _p) in data.items()}
    progress = {v: p for v, (_z, p) in data.items()}
    return reprs, progress


def main(args):
    rows = load_metrics_table(args.runs_dir, args.experiments)
    df = pd.DataFrame(rows)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.results_dir) / "results_table.csv", index=False)
    with open(Path(args.results_dir) / "results_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(df.to_string(index=False))
    print("\n=== Ordinality columns (the ones that answer the research question) ===")
    print(df[["experiment"] + KEY_METRICS].to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    for name in args.experiments:
        reprs, progress = build_test_representations(args.runs_dir, name, device)
        plot_pca_trajectories(
            reprs, progress, out_path=f"{args.plots_dir}/pca_{name}.png",
            title=f"{name} — test-video PCA trajectories", n_videos=args.n_pca_videos, seed=args.seed,
        )
        geo = distance_progress_analysis(reprs, progress, rng=np.random.RandomState(args.seed))
        plot_distance_vs_progress(
            geo["distances"], geo["progress_diffs"], geo["spearman"],
            out_path=f"{args.plots_dir}/distance_progress_{name}.png",
            title=f"{name} — latent distance vs. progress difference",
        )
    print(f"\nresults table -> {args.results_dir}/results_table.csv")
    print(f"plots -> {args.plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["baseline", "timestamp", "shuffle_k8"])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--n-pca-videos", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
