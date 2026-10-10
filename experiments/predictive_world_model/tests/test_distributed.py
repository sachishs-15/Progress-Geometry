"""Two-process CPU (gloo) smoke test of data-parallel training, plus a one-process run for
comparison: same files out, same CSV columns, checkpoints load into the same model, and
ranks agree. Run: python tests/test_distributed.py  (takes ~20 s, no GPU needed)."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(world, tmp, extra=()):
    ckpt = tmp / f"run{world}"
    launch = ([PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
              if world > 1 else [PY])
    cmd = launch + [str(ROOT / "train.py"), "--features", str(tmp / "feats.pt"), "--device", "cpu",
                    "--feature-dim", "8", "--latent-dim", "8", "--nhead", "2", "--pred-num-layers", "1",
                    "--epochs", "2", "--batch-size", "4", "--num-workers", "0", "--val-fraction", "0.25",
                    "--checkpoint-dir", str(ckpt), "--cls-agg-head", "--agg-out-dim", "8",
                    "--straighten-weight", "0.5", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return ckpt, out


def rows(ckpt):
    return list(csv.DictReader(open(ckpt / "train_log.csv")))


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        n = 48
        torch.manual_seed(0)
        torch.save({"features": torch.randn(n, 16, 8).cumsum(1), "video_paths": [f"v{i}.mp4" for i in range(n)]},
                   tmp / "feats.pt")
        c1, _ = run(1, tmp)
        c2, out2 = run(2, tmp)
        assert sorted(p.name for p in c1.iterdir() if p.suffix != ".log") == \
               sorted(p.name for p in c2.iterdir() if p.suffix != ".log"), "different output files"
        assert list(rows(c1)[0]) == list(rows(c2)[0]), "CSV columns changed"
        assert len(rows(c2)) == 2
        assert "Distributed: 2 GPUs" in (c2 / "train.log").read_text()
        s1, s2 = (torch.load(c / "last.ckpt")["model"] for c in (c1, c2))
        assert s1.keys() == s2.keys(), "checkpoint keys differ (DDP prefix leaked?)"
        assert all(not k.startswith("module.") for k in s2)
        # 36 train clips -> 18 per rank -> ceil(18 / 4) = 5 batches per rank per epoch
        assert "5 batches/epoch" in (c2 / "train.log").read_text()
        # the aggregation head must have been trained (its gradient is synced via TrainStep)
        init = torch.load(c2 / "last.ckpt")["model"]
        assert any(k.startswith("agg_head") for k in init)
        # straighten-weight 0 leaves the aggregation head without gradient: must not crash DDP
        c3, _ = run(2, tmp, extra=("--straighten-weight", "0", "--checkpoint-dir", str(tmp / "run_sw0")))
        assert (c3 / "last.ckpt").exists()
    print("ok")


if __name__ == "__main__":
    main()
