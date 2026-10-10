"""Representation-learning stage for the timestamp-control and shuffle-ordering
conditions (Model B / Model C). Model A (baseline) skips this entirely and probes raw
features directly — see train_progress_probe.py.

For each training step we sample a batch of B videos and K chronologically-sorted
timesteps per video. The `timestamp` objective regresses each sampled timestep's true
normalized position in its source video, idx / (T - 1). The `shuffle` objective shuffles
the K timesteps and asks a small Transformer to recover each one's rank within the
sampled subset.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import CachedFeatureDataset  # noqa: E402
from data.splits import get_or_create_splits  # noqa: E402
from losses.ordering_loss import ordering_loss, shuffle_permutation  # noqa: E402
from models.encoder import FeatureEncoder  # noqa: E402
from models.ordering_model import OrderingTransformer, TimestampHead  # noqa: E402
from utils.config import load_config, prepare_run_dir  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def _load_eligible(dataset: CachedFeatureDataset, video_ids, k: int, cache: dict):
    eligible = []
    for vid in video_ids:
        rec = cache.setdefault(vid, dataset.load(vid))
        if rec.features.shape[0] >= k:
            eligible.append(vid)
    return eligible


def _sample_batch(dataset, eligible_ids, cache, batch_size, k, rng):
    """Return features, source indices, and source lengths for a sampled batch.

    Features are chronological within each row. Keeping the sampled indices is
    essential for timestamp supervision: rank within a random K-frame subset is not
    the same quantity as absolute normalized time in the original T-frame video.
    """
    chosen = rng.choice(eligible_ids, size=batch_size, replace=len(eligible_ids) < batch_size)
    rows, sampled_indices, source_lengths = [], [], []
    for vid in chosen:
        rec = cache[vid]
        T = rec.features.shape[0]
        idx = np.sort(rng.choice(T, size=k, replace=False))
        rows.append(rec.features[idx])
        sampled_indices.append(idx)
        source_lengths.append(T)
    return (np.stack(rows).astype(np.float32), np.stack(sampled_indices),
            np.asarray(source_lengths, dtype=np.int64))


def _timestamp_targets(sampled_indices, source_lengths):
    """True normalized source-video timestamps for [B, K] sampled indices."""
    denominators = np.maximum(np.asarray(source_lengths, dtype=np.float32) - 1, 1)
    return np.asarray(sampled_indices, dtype=np.float32) / denominators[:, None]


def train_representation(config: dict, config_path: str, runs_dir: str = "runs") -> Path:
    set_seed(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_cfg = config["dataset"]
    dataset = CachedFeatureDataset(ds_cfg["cache_dir"])
    splits = get_or_create_splits(
        config.get("splits_dir", "splits"), dataset,
        train=config["split"]["train"], val=config["split"]["val"], test=config["split"]["test"],
        seed=config["split"].get("seed", 42),
    )

    rep_cfg = config["representation"]
    objective = rep_cfg["objective"]
    assert objective in ("timestamp", "shuffle")
    k = rep_cfg.get("k", 8)

    train_frac = config.get("train_frac", 1.0)
    train_ids = sorted(splits["train"])
    if train_frac < 1.0:
        n_keep = max(1, int(round(len(train_ids) * train_frac)))
        train_ids = list(np.random.RandomState(config["split"].get("seed", 42)).choice(
            train_ids, size=n_keep, replace=False))

    cache: dict = {}
    train_eligible = _load_eligible(dataset, train_ids, k, cache)
    val_eligible = _load_eligible(dataset, splits["val"], k, cache)
    if not train_eligible:
        raise RuntimeError(f"No training videos have >= K={k} timesteps")

    encoder = FeatureEncoder(dataset.feature_dim, rep_cfg.get("latent_dim", 512)).to(device)
    if objective == "shuffle":
        head = OrderingTransformer(
            latent_dim=rep_cfg.get("latent_dim", 512), k=k,
            num_layers=rep_cfg.get("num_layers", 2), num_heads=rep_cfg.get("num_heads", 8),
        ).to(device)
    else:
        head = TimestampHead(latent_dim=rep_cfg.get("latent_dim", 512)).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=rep_cfg.get("lr", 1e-3))

    epochs = rep_cfg.get("epochs", 30)
    batch_size = rep_cfg.get("batch_size", 16)
    steps_per_epoch = rep_cfg.get("steps_per_epoch", max(1, len(train_eligible) // batch_size))
    rng = np.random.RandomState(config.get("seed", 42))
    shuffle_strength = rep_cfg.get("shuffle_strength", "strong")
    shuffle_sigma = rep_cfg.get("shuffle_sigma", None)

    history = []
    for epoch in range(epochs):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            chron, sampled_idx, source_lengths = _sample_batch(
                dataset, train_eligible, cache, batch_size, k, rng
            )
            chron_t = torch.from_numpy(chron).to(device)  # [B, K, D]

            if objective == "shuffle":
                perms = np.stack([shuffle_permutation(k, strength=shuffle_strength, sigma=shuffle_sigma, rng=rng)
                                   for _ in range(batch_size)])  # [B, K]
                perms_t = torch.from_numpy(perms).long().to(device)
                shuffled = torch.gather(
                    chron_t, 1, perms_t.unsqueeze(-1).expand(-1, -1, chron_t.shape[-1])
                )
                z = encoder(shuffled)
                logits = head(z)
                loss = ordering_loss(logits, perms_t)
            else:
                z = encoder(chron_t)
                pred = head(z)  # [B, K]
                target = torch.from_numpy(
                    _timestamp_targets(sampled_idx, source_lengths)
                ).to(device)
                loss = nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metric = _evaluate_representation(encoder, head, dataset, val_eligible, cache, batch_size,
                                               k, objective, shuffle_strength, shuffle_sigma, device,
                                               np.random.RandomState(0))
        history.append({"epoch": epoch, "train_loss": epoch_loss / steps_per_epoch, "val": val_metric})
        print(f"[{objective}] epoch {epoch + 1}/{epochs} train_loss={epoch_loss / steps_per_epoch:.4f} "
              f"val={val_metric:.4f}")

    run_dir = prepare_run_dir(runs_dir, config["experiment_name"], config_path)
    encoder.save(run_dir / "encoder.pt")
    torch.save(head.state_dict(), run_dir / "ordering_head.pt")
    with open(run_dir / "representation_train_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{objective}] saved encoder -> {run_dir / 'encoder.pt'}")
    return run_dir


@torch.no_grad()
def _evaluate_representation(encoder, head, dataset, eligible_ids, cache, batch_size, k, objective,
                              shuffle_strength, shuffle_sigma, device, rng) -> float:
    """Returns ordering accuracy (shuffle) or MSE (timestamp) on a held-out video pool."""
    if not eligible_ids:
        return float("nan")
    encoder.eval()
    head.eval()
    chron, sampled_idx, source_lengths = _sample_batch(
        dataset, eligible_ids, cache, min(batch_size, len(eligible_ids)), k, rng
    )
    chron_t = torch.from_numpy(chron).to(device)

    if objective == "shuffle":
        b = chron_t.shape[0]
        perms = np.stack([shuffle_permutation(k, strength=shuffle_strength, sigma=shuffle_sigma, rng=rng)
                           for _ in range(b)])
        perms_t = torch.from_numpy(perms).long().to(device)
        shuffled = torch.gather(chron_t, 1, perms_t.unsqueeze(-1).expand(-1, -1, chron_t.shape[-1]))
        logits = head(encoder(shuffled))
        pred_rank = logits.argmax(dim=-1)
        return (pred_rank == perms_t).float().mean().item()

    z = encoder(chron_t)
    pred = head(z)
    target = torch.from_numpy(_timestamp_targets(sampled_idx, source_lengths)).to(device)
    return nn.functional.mse_loss(pred, target).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_representation(cfg, args.config, args.runs_dir)
