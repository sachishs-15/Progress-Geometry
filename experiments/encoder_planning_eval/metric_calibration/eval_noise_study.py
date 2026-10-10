"""Score trained checkpoints (CLS family) on SSv2 val with the full calibrated metric set, one
JSON of scalars + one npz of per-clip arrays per checkpoint, for seed / epoch noise analysis.

    python eval_noise_study.py --ckpt ../../predictive_world_model/checkpoints/noise_study/vicreg_s1/last.ckpt \
        --name vicreg_s1_last --out-dir results/noise_study
Special name 'RAW' scores the raw DINOv2 CLS features instead (no checkpoint).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import calibrate_planning as cpl  # noqa: E402
import calibrate_probes as cpr  # noqa: E402
from eval_common.loading import encode_world_model, load_world_model  # noqa: E402
from metrics_batch import dataset_metrics, per_clip_metrics  # noqa: E402

PWM = HERE.parents[1] / "predictive_world_model"


def build_untrained(config, seed):
    """Same architecture as a trained checkpoint (from its config.yaml), random init, never trained."""
    import inspect
    import yaml
    from models.world_model import ProceduralWorldModel
    cfg = yaml.safe_load(open(config))
    params = inspect.signature(ProceduralWorldModel.__init__).parameters
    kwargs = {k: v for k, v in cfg.items() if k in params}
    window = cfg.get("window_size") or 0
    kwargs["num_frames"] = window + 1 if window > 0 else cfg.get("n_frames", 16)
    torch.manual_seed(seed)
    return ProceduralWorldModel(**kwargs).eval()


def get_z(ckpt, config, split):
    f = cpr.FEATS.format("dinov2", split)
    d = torch.load(f, map_location="cpu")
    return d, None


def run(ckpt, config, name, out_dir, probes=True):
    tr_m, va_m, classes, ftr, fva = cpr.load_split("dinov2")
    if ckpt == "RAW":
        ztr, zva = ftr, fva
        epoch = None
    elif ckpt.startswith("UNTRAINED:"):
        seed = int(ckpt.split(":")[1])
        model = build_untrained(config, seed)
        epoch = -1
        ztr = encode_world_model(model, torch.from_numpy(ftr).float(), torch.device("cpu"), 256).astype(np.float64)
        zva = encode_world_model(model, torch.from_numpy(fva).float(), torch.device("cpu"), 256).astype(np.float64)
    else:
        model, _, epoch = load_world_model(ckpt, config, torch.device("cpu"))
        ztr = encode_world_model(model, torch.from_numpy(ftr).float(), torch.device("cpu"), 256).astype(np.float64)
        zva = encode_world_model(model, torch.from_numpy(fva).float(), torch.device("cpu"), 256).astype(np.float64)
    out = {"name": name, "epoch": epoch, "latent_dim": int(zva.shape[-1])}
    pc, keep = per_clip_metrics(zva)
    out["trajectory"] = {k: float(v[keep].mean()) for k, v in pc.items()}
    out["trajectory_median"] = {k: float(np.median(v[keep])) for k, v in pc.items()}
    out["dataset"] = dataset_metrics(zva)
    rows = cpl.calibrate_arrays(name, zva, None, boots=200, with_ridge=False)
    out["planning"] = {r["predictor"]: {k: r[k] for k in ("r2", "cos_mean", "cos_median", "rel_l2_median", "mse") if k in r}
                       for r in rows}
    if probes:
        res = cpr.probe_suite(ztr, tr_m, zva, va_m, classes, light=True)
        res.pop("_pred_linear_seed0", None), res.pop("_pred_knn_seed0", None)
        out["probes"] = res
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(out_dir) / f"{name}.json", "w"), default=float, indent=1)
    np.savez_compressed(Path(out_dir) / f"{name}.perclip.npz", keep=keep, **pc)
    print("saved", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--also", default="", help="comma list of extra checkpoint stems in the same dir (e.g. epoch29,epoch39), scored without probes")
    a = ap.parse_args()
    cfg = a.config or (str(Path(a.ckpt).parent / "config.yaml") if a.ckpt != "RAW" else None)
    run(a.ckpt, cfg, a.name, a.out_dir, probes=not a.no_probes)
    for extra in filter(None, a.also.split(",")):
        run(str(Path(a.ckpt).parent / f"{extra}.ckpt"), cfg, f"{a.name}_{extra}", a.out_dir, probes=False)
