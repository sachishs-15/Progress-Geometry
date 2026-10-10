"""Generate a synthetic dataset in the SAME unified cache format `prepare_crosstask.py`
produces (data/cache.py), so it exercises exactly the same downstream code path with no
special-casing. Purpose: prove the pipeline runs correctly end-to-end without requiring the
30GB CrossTask features download. This is a smoke test, NOT a test of the research
hypothesis — see README.

Each synthetic task gets a random smooth trajectory through D-dim space (a small sum of
sinusoids in normalized progress, D=3200 to match CrossTask exactly), so there IS a genuine
(if noisy) progress signal in the raw features, similar in spirit to how real instructional-
video features plausibly evolve smoothly over a task.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import write_manifest, write_video_cache  # noqa: E402
from utils.config import load_config  # noqa: E402


def make_task_generator(rng: np.random.RandomState, feature_dim: int, n_basis: int = 8):
    freqs = rng.uniform(0.5, 3.0, size=n_basis)
    phases = rng.uniform(0, 2 * np.pi, size=n_basis)
    coeffs = rng.normal(0, 1, size=(n_basis, feature_dim)) / np.sqrt(n_basis)
    offset = rng.normal(0, 1, size=feature_dim)

    def gen(progress: np.ndarray) -> np.ndarray:
        basis = np.stack([np.sin(f * progress * np.pi + p) for f, p in zip(freqs, phases)], axis=1)  # [T, n_basis]
        return offset[None, :] + basis @ coeffs  # [T, D]

    return gen


def main(args):
    rng = np.random.RandomState(args.seed)
    video_to_task = {}

    task_ids = [f"synth_task_{i:02d}" for i in range(args.n_tasks)]
    for task_id in task_ids:
        gen = make_task_generator(rng, args.feature_dim, n_basis=args.n_basis)
        n_videos = rng.randint(args.min_videos_per_task, args.max_videos_per_task + 1)
        n_steps = rng.randint(3, 7)
        for v in range(n_videos):
            video_id = f"{task_id}_vid{v:02d}"
            t = rng.randint(args.min_frames, args.max_frames + 1)
            progress = np.arange(t) / max(t - 1, 1)
            clean = gen(progress)
            noise = rng.normal(0, args.noise_std, size=clean.shape)
            features = (clean + noise).astype(np.float32)
            step_labels = np.minimum((progress * n_steps).astype(np.int64), n_steps - 1)

            write_video_cache(args.out, video_id, task_id, features, step_labels)
            video_to_task[video_id] = task_id

    write_manifest(args.out, name="synthetic", feature_dim=args.feature_dim, video_to_task=video_to_task)
    print(f"Wrote {len(video_to_task)} synthetic videos across {len(task_ids)} tasks -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="optional yaml overriding the CLI defaults below")
    parser.add_argument("--out", default="data_cache/synthetic")
    parser.add_argument("--n-tasks", type=int, default=6)
    parser.add_argument("--min-videos-per-task", type=int, default=6)
    parser.add_argument("--max-videos-per-task", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--feature-dim", type=int, default=3200, help="3200 to match CrossTask exactly")
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()
    if cli_args.config:
        for key, value in load_config(cli_args.config).items():
            setattr(cli_args, key, value)
    main(cli_args)
