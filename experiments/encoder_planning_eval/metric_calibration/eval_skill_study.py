"""Skill ladder for raw CLS latents and for each CLS-family checkpoint (teacher-forced one step)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import calibrate_probes as cpr  # noqa: E402
import calibrate_skill as sk  # noqa: E402
from eval_common.loading import load_world_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    _, _, _, ftr, fva = cpr.load_split("dinov2")
    dev = torch.device("cpu")
    if a.ckpt == "RAW":
        ztr, zva, pred = ftr, fva, None
    else:
        model, cfg, ep = load_world_model(a.ckpt, str(Path(a.ckpt).parent / "config.yaml"), dev, with_predictor=True)
        with torch.no_grad():
            enc = lambda f: torch.cat([model.encoder(torch.from_numpy(f[i:i + 256]).float()) for i in range(0, len(f), 256)])
            zt, zv = enc(ftr), enc(fva)
            pred = torch.cat([model.predictor(zv[i:i + 256])[:, :-1] for i in range(0, len(zv), 256)]).numpy().astype(np.float64)
        ztr, zva = zt.numpy().astype(np.float64), zv.numpy().astype(np.float64)
    errs, nv = sk.ladder(zva, ztr, pred)
    out = sk.summarize(errs, nv)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out_dir) / f"{a.name}.json", "w"), indent=1)
    np.savez_compressed(Path(a.out_dir) / f"{a.name}.errs.npz", **errs)
    print(a.name, {k: round(v["skill"], 4) for k, v in out.items()})
