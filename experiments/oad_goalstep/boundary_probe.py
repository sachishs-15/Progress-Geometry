"""Do causal surprise signals spike at Goal-Step step/substep boundaries? (no training)

  python boundary_probe.py --run-dir <pwm checkpoint dir> --ckpt best.ckpt --cache-dir <dense live cache> --uids u1 u2

Signals per feature t (all use only frames <= t):
  pe      predictor error: the predictor sees the previous <=15 latents (a window starting at
          position 0, as in training) and its last output is scored against z_t
  z_jump  ||z_t - z_{t-1}||, mean over patch tokens
  raw_jump / cls_jump  the same on the raw DINOv2 patch tokens / CLS cosine distance
Ground truth: a feature is positive when it lies within one feature spacing of a
start or end time of any step/substep (all levels), or of a top-level step only.
Run at stride 1 (0.53 s spacing) and 2 (0.93 s, nearer the clip spacing the model trained on).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

EVAL = Path("/home/sachish/proc-planning/experiments/encoder_planning_eval")
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "world_model_eval"))
from compute_prediction_skill import position_error  # noqa: E402
from eval_common.loading import load_world_model  # noqa: E402

from dense_provider import DensePatchProvider  # noqa: E402
from goalstep_common import FPS, STRIDE, flatten_steps, load_split  # noqa: E402

W = 15


@torch.no_grad()
def signals(model, kind, tokens, stride, device, batch=4):
    x = torch.from_numpy(np.asarray(tokens[::stride], dtype=np.float32)).to(device)   # (L, 197, 384)
    cls, patches = x[:, 0], x[:, 1:]
    z = torch.cat([model.encoder(patches[None, i:i + 256])[0] for i in range(0, len(x), 256)])  # (L, P, D)
    L = len(z)
    pe = torch.full((L,), float("nan"), device=device)
    for t in range(1, min(W, L)):                                         # short prefixes
        pred = model.predictor(z[None, :t])[0, -1]
        pe[t] = position_error(pred[None, None], z[None, t:t + 1], kind)[0, 0]
    if L > W:
        wins = z.unfold(0, W, 1)[:-1].permute(0, 3, 1, 2)                  # windows z[t-W:t] for t=W..L-1
        for i in range(0, len(wins), batch):
            pred = model.predictor(wins[i:i + batch])[:, -1]
            tgt = z[W + i:W + i + len(pred)]
            pe[W + i:W + i + len(pred)] = position_error(pred[:, None], tgt[:, None], kind)[:, 0]
    copy = torch.full((L,), float("nan"), device=device)
    copy[1:] = position_error(z[:-1][:, None], z[1:][:, None], kind)[:, 0]
    nan = torch.full((1,), float("nan"), device=device)
    out = {
        "pe": pe,
        "z_jump": torch.cat([nan, (z[1:] - z[:-1]).norm(dim=-1).mean(-1)]),
        "raw_jump": torch.cat([nan, (patches[1:] - patches[:-1]).norm(dim=-1).mean(-1)]),
        "cls_jump": torch.cat([nan, 1 - torch.nn.functional.cosine_similarity(cls[1:], cls[:-1], dim=-1)]),
    }
    return {k: v.cpu().numpy() for k, v in out.items()}, pe.cpu().numpy(), copy.cpu().numpy()


def boundaries(video, top_only):
    segs = [(s["start_time"], s["end_time"]) for s in video["segments"]] if top_only else \
           [(a, b) for a, b, _ in flatten_steps(video["segments"])]
    return np.array(sorted({t for ab in segs for t in ab}))


def main(args):
    device = "cuda"
    run = Path(args.run_dir)
    model, _, _ = load_world_model(run / args.ckpt, run / "config.yaml", device, with_predictor=True)
    import yaml
    kind = yaml.safe_load(open(run / "config.yaml")).get("pred_loss", "l2")
    provider = DensePatchProvider(args.cache_dir, device=device)
    videos = {v["video_uid"]: v for s in ("train", "val") for v in load_split(s)}
    for stride in (1, 2):
        rows = {k: [] for k in ("pe", "z_jump", "raw_jump", "cls_jump")}
        ys = {"all": [], "top": []}
        skill_pe, skill_copy = [], []
        for uid in args.uids:
            sig, pe, copy = signals(model, kind, provider.get(uid), stride, device)
            L = len(pe)
            t = np.arange(L) * stride * STRIDE / FPS
            tol = stride * STRIDE / FPS
            for name, top in (("all", False), ("top", True)):
                b = boundaries(videos[uid], top)
                ys[name].append((np.abs(t[:, None] - b[None, :]).min(1) <= tol)[1:])
            for k in rows:
                rows[k].append(sig[k][1:])
            skill_pe.append(pe[1:]); skill_copy.append(copy[1:])
        print(f"\n=== stride {stride} ({stride * STRIDE / FPS:.2f}s spacing), videos={len(args.uids)}, "
              f"features={sum(len(r) for r in rows['pe'])} ===")
        skill = 1 - np.nanmean(np.concatenate(skill_pe)) / np.nanmean(np.concatenate(skill_copy))
        print(f"predictor skill vs copy-previous: {skill:.3f}")
        for name in ("all", "top"):
            y = np.concatenate(ys[name])
            print(f"-- boundaries: {name}-levels, positive rate {y.mean():.3f} (AP of a random score = that)")
            for k in rows:
                s = np.concatenate(rows[k]); ok = ~np.isnan(s)
                print(f"   {k:9s} AUROC {roc_auc_score(y[ok], s[ok]):.3f}   AP {average_precision_score(y[ok], s[ok]):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.ckpt")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--uids", nargs="+", required=True)
    main(p.parse_args())
