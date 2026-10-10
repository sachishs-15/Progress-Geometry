"""Score raw DINOv2 CLS or concatenated patch features on an existing SSv2 split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from injections import INJECTIONS, PIXEL_ONLY
from run_egoper_style_eval import evaluate, mark_negative_controls, write_summary
import run_injection_eval as injection_runner
from run_injection_eval import DATASETS, RawReadout, run_dataset


class CLSReadout(RawReadout):
    name = "dinov2_cls"


class PatchConcatReadout(RawReadout):
    name = "dinov2_patch_concat"

    @torch.no_grad()
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"expected (B,T,P,D) patch features, got {tuple(x.shape)}")
        flattened = x.flatten(2)
        tokstep = (x[:, 1:] - x[:, :-1]).norm(dim=-1).mean(-1)
        return flattened.cpu().numpy(), None, tokstep.cpu().numpy(), None


def concat_geometry_scores(z):
    """Equivalent geometry in float32 to avoid expanding 75k-D paths to float64."""
    z = np.asarray(z, dtype=np.float32)
    eps = 1e-8
    norm = lambda x: np.sqrt(np.einsum("btd,btd->bt", x, x, optimize=True))
    d = norm(np.diff(z, axis=1))
    med = np.median(d, axis=1)
    out = {"jump_max": d.max(1), "jump_spike": d.max(1) / (med + eps)}
    cv = norm(z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2])
    out["cv_max"], out["cv_mean"] = cv.max(1), cv.mean(1)
    rel = z - z[:, :1]
    u = z[:, -1] - z[:, 0]
    p = np.einsum("btd,bd->bt", rel, u, optimize=True) / (np.einsum("bd,bd->b", u, u)[:, None] + eps)
    dev = norm(rel - p[..., None] * u[:, None])
    out["chord_dev"] = dev.max(1) / (d.mean(1) + eps)
    out["backslide"] = (np.maximum.accumulate(p, axis=1) - p).max(1)
    out["step_cv"] = d.std(1) / (d.mean(1) + eps)
    out["stall"] = 1 - d.min(1) / (med + eps)
    return out, d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--representation", choices=("cls", "patch_concat"), default="cls")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.features is None:
        folder = "dinov2" if args.representation == "cls" else "dinov2_noreg"
        args.features = Path("/mnt/rdata7/proprogress/data/SSv2/precomputed_features") / folder / "ssv2_validation.pt"
    protocol = json.loads((args.reference_run / "protocol.json").read_text())
    if protocol["args"]["dataset"] != "ssv2":
        parser.error("reference run must use SSv2")
    seed = protocol["args"]["seed"]
    paths = protocol["calibration_paths"] + protocol["test_paths"]
    if len(paths) != len(set(paths)):
        parser.error("reference paths are not unique")
    data = torch.load(args.features, mmap=True, weights_only=False)
    features = data["features"]
    expected_ndim = 3 if args.representation == "cls" else 4
    if features.ndim != expected_ndim or features.shape[1] != 16:
        parser.error(f"expected 16-frame {args.representation} features, got {tuple(features.shape)}")
    lookup = {path: i for i, path in enumerate(data["video_paths"])}
    missing = [path for path in paths if path not in lookup]
    if missing:
        parser.error(f"{len(missing)} reference clips are missing from CLS features")
    manifest = pd.read_csv(DATASETS["ssv2"]["manifest"], dtype=str)
    label_of = dict(zip(manifest["video_path"], manifest["verb"]))
    sample = features[torch.tensor([lookup[path] for path in paths])].float()
    n_cal = len(protocol["calibration_paths"])
    readout = CLSReadout() if args.representation == "cls" else PatchConcatReadout()
    if args.representation == "patch_concat":
        injection_runner.geometry_scores = concat_geometry_scores
    injections = protocol["args"]["injections"] or [i for i in INJECTIONS if i not in PIXEL_ONLY]
    records = {}
    for split, indices, conditions in (("calibration", slice(None, n_cal), []),
                                       ("test", slice(n_cal, None), injections)):
        split_paths = paths[indices]
        split_sample = (sample[indices], [label_of[p] for p in split_paths], split_paths)
        split_args = argparse.Namespace(n_clips=len(split_paths), seed=seed,
                                        batch_size=args.batch_size, injections=conditions)
        records[split], _, _ = run_dataset(f"ssv2/{args.representation}/{split}", DATASETS["ssv2"],
                                            [readout], split_args, torch.device("cpu"),
                                            sample=split_sample)
    segment, frame, loc = evaluate("ssv2", records["calibration"], records["test"],
                                    [readout], injections, protocol["args"]["n_boot"], seed)
    segment = mark_negative_controls(segment)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    segment.to_csv(out / "segment_metrics.csv", index=False)
    frame.to_csv(out / "frame_metrics.csv", index=False)
    loc.to_csv(out / "localization.csv", index=False)
    arrays = {f"{split}|{name}|{condition}|{field}": value
              for split, table in records.items()
              for (name, condition), result in table.items()
              for field, value in result.items() if field != "seams"}
    np.savez_compressed(out / "per_clip_scores.npz", **arrays)
    (out / "protocol.json").write_text(json.dumps({"features": str(args.features),
        "representation": args.representation,
        "reference_run": str(args.reference_run), "seed": seed,
        "calibration_paths": protocol["calibration_paths"],
        "test_paths": protocol["test_paths"], "injections": injections}, indent=2))
    write_summary(out, "ssv2", n_cal, len(paths) - n_cal, segment, frame, loc)
    print(f"saved -> {out} ({n_cal} calibration, {len(paths) - n_cal} test clips)")


if __name__ == "__main__":
    main()
