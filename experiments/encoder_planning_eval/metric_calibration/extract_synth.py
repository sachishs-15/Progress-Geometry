"""Encode synthetic clips (synth_videos.py) with registry encoders; also save raw-pixel 'features'
and a contact sheet.

    python extract_synth.py --encoders dinov2:cls dinov3:cls clip:cls vjepa2.1:mean_patch --n 200 --out-dir results/synth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "experiments" / "frozen_encoders"))
import synth_videos as sv  # noqa: E402


def pixel_features(clips, size=24):
    out = []
    for c in clips:
        out.append(np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR), np.float32).ravel() / 255
                             for f in c]))
    return torch.from_numpy(np.stack(out))


def contact_sheet(data, path, k=4):
    rows = []
    for cond in sv.CONDITIONS:
        rows.append(np.concatenate([np.asarray(Image.fromarray(data[cond][0][t]).resize((112, 112)))
                                    for t in np.linspace(0, sv.T - 1, 6).astype(int)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=[], help="name:pooling pairs")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--amplitude", action="store_true", help="render the object-size sweep (AMPLITUDE_CONDITIONS) instead")
    a = ap.parse_args()
    if a.amplitude:
        sv.CONDITIONS = sv.AMPLITUDE_CONDITIONS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {c: sv.make_dataset(c, a.n, a.seed + i) for i, c in enumerate(sv.CONDITIONS)}
    contact_sheet(data, out / "examples.png")
    if not (out / "pixel.pt").exists():
        torch.save({c: pixel_features(v) for c, v in data.items()}, out / "pixel.pt")
        print("saved pixel baseline")

    from common.base import build  # noqa: E402
    import models  # noqa: E402,F401
    for spec in a.encoders:
        name, pooling = spec.split(":")
        enc = build(name, device="cuda", pooling=pooling)
        feats = {}
        with enc:
            for cond, clips in data.items():
                feats[cond] = torch.stack([torch.from_numpy(enc.encode([Image.fromarray(f) for f in c]).embeddings)
                                           for c in tqdm(clips, desc=f"{name}/{cond}", leave=False)])
        torch.save(feats, out / f"{name}.pt")
        print("saved", name, {k: tuple(v.shape) for k, v in list(feats.items())[:2]}, flush=True)
