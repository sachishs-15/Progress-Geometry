"""Mean over the per-patch latent: (L, P*D) -> (L, D), from the per-patch files extract_pwm_features.py wrote.

  python make_mean_latent.py --root <data root> --src pwm_ep19_z --dst pwm_ep19_mean [--patches 196 --dim 8]

Reads <root>/<src>/<uid>.npy (L, patches*dim) and writes <root>/<dst>/<uid>.npy (L, dim) float16, the
average over the patch axis of each frame's encoder output. Skips finished videos.
"""
import argparse
from pathlib import Path

import numpy as np


def main(args):
    src, dst = Path(args.root) / args.src, Path(args.root) / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.npy") if ".tmp" not in f.name)
    for f in files:
        out = dst / f.name
        if out.exists():
            continue
        z = np.load(f, mmap_mode="r")
        mean = np.concatenate([np.asarray(z[i:i + 4096], dtype=np.float32).reshape(-1, args.patches, args.dim).mean(1)
                               for i in range(0, len(z), 4096)])
        np.save(out.with_suffix(".tmp.npy"), mean.astype(np.float16))
        out.with_suffix(".tmp.npy").rename(out)
    print(f"{len(files)} videos -> {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--patches", type=int, default=196)
    p.add_argument("--dim", type=int, default=8)
    main(p.parse_args())
