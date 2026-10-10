"""Check ope.py against GTCC's own get_trueprogress / get_cum_matrix on real pouring labels.
Run with the GTCC env:  ~/gtcc-venv/bin/python test_ope.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GTCC = HERE.parents[1] / "external" / "GTCC_CVPR2024"
for k in ("DATASET_PATH", "OUTPUT_PATH", "JSON_DPATH"):
    os.environ.setdefault(k, "/tmp")
sys.path.insert(0, str(GTCC))
from utils.tensorops import get_cum_matrix, get_trueprogress  # noqa: E402

sys.path.insert(0, str(HERE))
from ope import cum_path, gt_progress, segments  # noqa: E402
from pouring_data import load_split  # noqa: E402

rng = np.random.default_rng(0)
n = 0
for v in load_split("train") + load_split("val"):
    segs = segments(v.labels)
    tdict = {"step": [s[0] for s in segs], "start_frame": [s[1] for s in segs],
             "end_frame": [s[2] for s in segs], "name": v.name}
    theirs = get_trueprogress(tdict).numpy()
    ours = gt_progress(v.labels)
    assert theirs.shape == ours.shape and np.allclose(theirs, ours, atol=1e-6), v.name
    z = rng.normal(size=(len(v.labels), 16)).astype(np.float32)
    assert np.allclose(get_cum_matrix(torch.from_numpy(z)).numpy(), cum_path(z), rtol=1e-4), v.name
    n += 1
print(f"OK: gt_progress and cum_path match GTCC on {n} videos")
