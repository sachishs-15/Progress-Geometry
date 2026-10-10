"""Expose live-cache files to LSTR's own data layer, which reads <root>/<feature>/<uid>.npy.

  python link_live_cache.py --cache-dir <live cache> --data-info <data_info.json> --root <data_root>

Symlinks <root>/dinov2_noreg_tokens/<uid>.npy to the cache file, and writes a one-column
dummy <root>/motion_dummy/<uid>.npy because LSTR's data layer always loads a second
stream. Videos not in the cache yet are listed, not linked: run fill_live_cache.py first.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dense_provider import DensePatchProvider


def main(args):
    info = json.load(open(args.data_info))["GoalStep"]
    uids = info["train_session_set"] + info["test_session_set"]
    provider = DensePatchProvider(args.cache_dir)
    tok, mot = Path(args.root) / "dinov2_noreg_tokens", Path(args.root) / "motion_dummy"
    tok.mkdir(parents=True, exist_ok=True)
    mot.mkdir(parents=True, exist_ok=True)
    missing = []
    for uid in uids:
        src = provider.cached_path(uid)
        if src is None:
            missing.append(uid)
            continue
        link = tok / f"{uid}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        np.save(mot / f"{uid}.npy", np.zeros((np.load(src, mmap_mode="r").shape[0], 1), dtype=np.uint8))
    print(f"{len(uids) - len(missing)} linked, {len(missing)} missing from the cache")
    for uid in missing[:10]:
        print("MISSING", uid)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--data-info", required=True)
    p.add_argument("--root", required=True)
    main(p.parse_args())
