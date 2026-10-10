"""Combining runs must reject ambiguous or differently sampled results."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "combine_results.py"


def make_run(root, name, readout, seed=0):
    run = root / name
    run.mkdir()
    pd.DataFrame([{"dataset": "toy", "readout": readout, "injection": "reverse",
                   "score": "cv_mean", "auroc": 0.5}]).to_csv(run / "toy_auroc.csv", index=False)
    (run / "config.json").write_text(json.dumps({"args": {"seed": seed, "n_clips": 20}, "meta": {}}))
    return run


def test_combiner_rejects_duplicate_readouts_and_sampling_mismatch():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = make_run(root, "a", "raw")
        b = make_run(root, "b", "model")
        c = make_run(root, "c", "raw")
        d = make_run(root, "d", "other", seed=1)

        def combine(*runs):
            return subprocess.run([sys.executable, str(SCRIPT), *(str(r) for r in runs),
                                   "--out", str(root / "combined.md")], capture_output=True, text=True)

        assert combine(a, b).returncode == 0
        duplicate = combine(a, c)
        assert duplicate.returncode != 0 and "duplicate dataset/readout" in duplicate.stderr
        mismatch = combine(a, d)
        assert mismatch.returncode != 0 and "sampling mismatch" in mismatch.stderr


if __name__ == "__main__":
    test_combiner_rejects_duplicate_readouts_and_sampling_mismatch()
    print("PASS  test_combiner_rejects_duplicate_readouts_and_sampling_mismatch")
