"""The one progress-probe training routine used for all three conditions (section 5 / 11):
same `nn.Linear(D, 1)` architecture, optimizer, epochs, batch size, split and eval protocol
regardless of whether the input representation is raw features (Model A / baseline) or a
frozen encoder's output (Model B / timestamp, Model C / shuffle) — only `probe.representation`
and (for B/C) which run's `encoder.pt` to freeze and read from differ.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import CachedFeatureDataset  # noqa: E402
from data.progress import compute_progress  # noqa: E402
from data.splits import get_or_create_splits  # noqa: E402
from evaluation.latent_geometry import distance_progress_analysis  # noqa: E402
from evaluation.progress_metrics import all_progress_metrics  # noqa: E402
from evaluation.ranking_metrics import pairwise_ordering_accuracy  # noqa: E402
from models.encoder import FeatureEncoder  # noqa: E402
from models.progress_probe import ProgressProbe  # noqa: E402
from utils.config import load_config, prepare_run_dir  # noqa: E402
from utils.seed import set_seed  # noqa: E402

VideoData = Dict[str, Tuple[np.ndarray, np.ndarray]]  # vid -> (representation [T,D'], progress [T])


def build_video_data(dataset: CachedFeatureDataset, video_ids: List[str], progress_type: str,
                      representation: str, encoder, device: str) -> VideoData:
    data = {}
    for vid in video_ids:
        rec = dataset.load(vid)
        progress = compute_progress(rec, progress_type)
        if representation == "encoder":
            with torch.no_grad():
                z = encoder(torch.from_numpy(rec.features).float().to(device)).cpu().numpy()
        else:
            z = rec.features
        data[vid] = (z.astype(np.float32), progress.astype(np.float32))
    return data


def _flatten(data: VideoData, video_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([data[v][0] for v in video_ids], axis=0)
    ys = np.concatenate([data[v][1] for v in video_ids], axis=0)
    return xs, ys


def evaluate_split(probe, data: VideoData, video_ids: List[str], device: str, rng: np.random.RandomState) -> dict:
    probe.eval()
    video_preds = {}
    with torch.no_grad():
        for vid in video_ids:
            z, _ = data[vid]
            video_preds[vid] = probe(torch.from_numpy(z).to(device)).cpu().numpy()

    pred_flat = np.concatenate([video_preds[v] for v in video_ids])
    true_flat = np.concatenate([data[v][1] for v in video_ids])
    metrics = all_progress_metrics(pred_flat, true_flat)
    metrics["pairwise_ordering_accuracy"] = pairwise_ordering_accuracy(video_preds, rng=rng)

    reprs = {v: data[v][0] for v in video_ids}
    progress = {v: data[v][1] for v in video_ids}
    geo = distance_progress_analysis(reprs, progress, rng=rng)
    metrics["distance_progress_spearman"] = geo["spearman"]
    return metrics


def train_progress_probe(config: dict, config_path: str, runs_dir: str = "runs") -> Path:
    set_seed(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = prepare_run_dir(runs_dir, config["experiment_name"], config_path)

    ds_cfg = config["dataset"]
    dataset = CachedFeatureDataset(ds_cfg["cache_dir"])
    splits = get_or_create_splits(
        config.get("splits_dir", "splits"), dataset,
        train=config["split"]["train"], val=config["split"]["val"], test=config["split"]["test"],
        seed=config["split"].get("seed", 42),
    )

    train_frac = config.get("train_frac", 1.0)
    train_ids = sorted(splits["train"])
    if train_frac < 1.0:
        n_keep = max(1, int(round(len(train_ids) * train_frac)))
        train_ids = list(np.random.RandomState(config["split"].get("seed", 42)).choice(
            train_ids, size=n_keep, replace=False))

    probe_cfg = config["probe"]
    representation = probe_cfg.get("representation", "raw")
    encoder = None
    if representation == "encoder":
        encoder = FeatureEncoder.load(run_dir / "encoder.pt", device=device)
        encoder.eval()

    progress_type = ds_cfg.get("progress_type", "timestamp")
    train_data = build_video_data(dataset, train_ids, progress_type, representation, encoder, device)
    val_data = build_video_data(dataset, splits["val"], progress_type, representation, encoder, device)
    test_data = build_video_data(dataset, splits["test"], progress_type, representation, encoder, device)

    x_train, y_train = _flatten(train_data, train_ids)
    input_dim = x_train.shape[1]
    probe = ProgressProbe(input_dim).to(device)
    probe.set_normalization(x_train.mean(axis=0), x_train.std(axis=0))
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_cfg.get("lr", 1e-2),
                                  weight_decay=probe_cfg.get("weight_decay", 0.0))

    epochs = probe_cfg.get("epochs", 50)
    batch_size = probe_cfg.get("batch_size", 64)
    rng = np.random.RandomState(config.get("seed", 42))
    n = len(x_train)
    x_val, y_val = _flatten(val_data, splits["val"])

    history = []
    best_val_mse, best_state = float("inf"), None
    for epoch in range(epochs):
        probe.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            pred = probe(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        probe.eval()
        with torch.no_grad():
            val_mse = torch.nn.functional.mse_loss(
                probe(torch.from_numpy(x_val).to(device)), torch.from_numpy(y_val).to(device)
            ).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss / n_batches, "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    # Best-val-MSE checkpoint selection (identical protocol for every condition): with a
    # fixed epoch budget and no other regularization, a linear probe on a highly
    # progress-predictive representation can drive train loss to ~0 in a few epochs and
    # then extrapolate to huge values on held-out videos in the following ones — this
    # guards every condition against that equally rather than tuning it away per-condition.
    if best_state is not None:
        probe.load_state_dict(best_state)

    eval_rng = np.random.RandomState(config.get("seed", 42) + 1)
    val_metrics = evaluate_split(probe, val_data, splits["val"], device, eval_rng)
    test_metrics = evaluate_split(probe, test_data, sorted(splits["test"]), device, eval_rng)

    torch.save({"input_dim": input_dim, "state_dict": probe.state_dict()}, run_dir / "probe.pt")
    metrics = {
        "experiment_name": config["experiment_name"],
        "representation": representation,
        "progress_type": progress_type,
        "train_frac": train_frac,
        "n_train_videos": len(train_ids),
        "val": val_metrics,
        "test": test_metrics,
        "probe_train_log": history,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{config['experiment_name']}] test metrics: "
          + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    print(f"[{config['experiment_name']}] saved -> {run_dir / 'metrics.json'}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_progress_probe(cfg, args.config, args.runs_dir)
