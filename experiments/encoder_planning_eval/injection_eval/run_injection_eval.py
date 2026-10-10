"""Mistake-injection evaluation: do a world model's latent path and predictor flag
clips that have been deliberately broken, and does it do so better than raw features?

For each sampled clip the script builds a clean version and a set of perturbed versions
(injections.py), encodes all of them with every readout, scores them (scores.py) and
reports, per injection, how well each score separates perturbed from clean clips (AUROC,
bootstrap CI over clips) and whether the largest error lands at the seam.

Readouts
    raw     the input features themselves, mean-pooled over patches (no predictor)
    <name>  a ProceduralWorldModel checkpoint: encoder tokens mean-pooled over patches for
            the geometry scores, and the predictor's one-step error in its native token
            space (the checkpoint's own pred_loss) for the predictor scores

    python run_injection_eval.py --dataset k400 --dataset ssv2 \\
        --model vicreg3=../../predictive_world_model/checkpoints/pwm_patch_channel_paper_window0_vicreg_3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "world_model_eval"))
from compute_prediction_skill import position_error  # noqa: E402
from eval_common.loading import load_world_model  # noqa: E402
from injections import (INJECTIONS, NEEDS_DONOR, PIXEL_ONLY, REFERENCE, SEAM_INJECTIONS,  # noqa: E402
                        clean_window, rng_for)
from scores import (GEOMETRY_SCORES, PREDICTOR_SCORES, compare, geometry_scores,  # noqa: E402
                    localization, predictor_scores)

DATA = Path("/mnt/rdata7/proprogress/data")
PROC = Path(__file__).resolve().parents[3]
DATASETS = {
    "k400": dict(
        # _clean: re-extracted after the K400 copy was completed. The older
        # precomputed_features/ file was built from an incomplete copy, so ~40% of its clips
        # are black frames; it also lists 136 videos that cannot be decoded and are absent here.
        features=DATA / "kinetics-dataset/precomputed_features_clean/dinov2_noreg/k400_train_10pct.pt",
        manifest=PROC / "data/kinetics-400/k400_train_10pct.csv", label_col="label", n_classes=50),
    "ssv2": dict(
        features=DATA / "SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt",
        manifest=PROC / "data/chirality_in_action/SSv2/ssv2_validation.csv", label_col="verb",
        n_classes=None),
}
VICREG3 = PROC / "experiments/predictive_world_model/checkpoints/dump/pwm_patch_channel_paper_window0_vicreg_3"
DEFAULT_MODELS = [("vicreg3", VICREG3, "best.ckpt"), ("vicreg3_last", VICREG3, "last.ckpt")]

# (predictor score, geometry score) reported in the headline table, chosen in advance
HEADLINE = {
    "reverse": ("pe_mean", "cv_mean"), "swap": ("pe_max", "jump_spike"),
    "shuffle": ("pe_max", "jump_spike"), "drop": ("pe_max", "jump_spike"),
    "splice_other": ("pe_max", "chord_dev"), "splice_same": ("pe_max", "chord_dev"),
    "freeze": ("pe_max", "stall"), "speed_up": ("pe_mean", "step_cv"),
    "slow_down": ("pe_mean", "step_cv"), "appearance": ("pe_mean", "jump_max"),
    "blur": ("pe_mean", "jump_max"), "null_split": ("pe_mean", "jump_max"),
}


def sample_clips(paths, labels, n, n_classes, seed):
    """Indices of ~n clips, stratified over classes so that same-class donors exist."""
    if n < 2:
        raise ValueError("n_clips must be at least 2 to sample same-class donors")
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(len(paths), min(n, len(paths)), replace=False))
    by = {}
    for i, lab in enumerate(labels):
        if lab is not None:
            by.setdefault(lab, []).append(i)
    classes = [c for c, v in by.items() if len(v) >= 2]
    if not classes:
        raise ValueError("no classes have two clips for same-class donors")
    rng.shuffle(classes)
    if n_classes:
        classes = classes[:n_classes]
    classes = classes[:n // 2]
    per = int(np.ceil(n / len(classes)))
    base, extras = [], []
    for c in classes:
        chosen = [int(i) for i in rng.choice(by[c], min(per, len(by[c])), replace=False)]
        base.extend(chosen[:2])
        extras.extend(chosen[2:])
    idx = base + [int(i) for i in rng.choice(extras, min(n - len(base), len(extras)), replace=False)]
    return np.sort(np.asarray(idx))


def sample_indices(cfg, n_clips, seed):
    """The sampled clips (same for feature and pixel runs, so their perturbations line up)."""
    data = torch.load(cfg["features"], mmap=True)
    paths = data["video_paths"]
    manifest = pd.read_csv(cfg["manifest"], dtype=str)
    lab_of = dict(zip(manifest["video_path"], manifest[cfg["label_col"]]))
    labels = [lab_of.get(p) for p in paths]
    idx = sample_clips(paths, labels, n_clips, cfg["n_classes"], seed)
    return data, idx, [labels[i] for i in idx], [paths[i] for i in idx]


def load_sample(cfg, n_clips, seed):
    data, idx, labels, paths = sample_indices(cfg, n_clips, seed)
    return data["features"][torch.from_numpy(idx)].float(), labels, paths


def load_sample_pixels(name, cfg, n_clips, seed, cache_dir, workers=8, img_size=224, n_frames=16):
    """The same clips as load_sample, as decoded uint8 RGB frames (n, T, 3, H, W): the 16
    uniformly spaced frames FeatureProvider(token_mode='raw') would give. Cached on disk."""
    _, idx, labels, paths = sample_indices(cfg, n_clips, seed)
    cache = Path(cache_dir) / f"frames_{name}_n{len(idx)}_seed{seed}_{img_size}.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["paths"] == paths:
            return saved["frames"], labels, paths
    from concurrent.futures import ThreadPoolExecutor
    from data.feature_provider import FeatureProvider
    provider = FeatureProvider(None, live_cache_dir=None, n_frames=n_frames, token_mode="raw",
                               raw_img_size=img_size, device="cpu")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        frames = torch.stack(list(pool.map(provider._compute_live_raw, paths)))
    print(f"[{name}] decoded {len(paths)} clips in {time.time() - t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": frames, "paths": paths}, cache)
    return frames, labels, paths


def pick_donors(labels, seed):
    """Per clip: a donor from another class (or any other clip) and one from its own class."""
    other, same = [], []
    for i, lab in enumerate(labels):
        rng = rng_for(seed, "donor", i)
        diff = [j for j in range(len(labels)) if labels[j] != lab]
        eq = [j for j in range(len(labels)) if j != i and labels[j] == lab]
        other.append(int(rng.choice(diff)) if diff else None)
        same.append(int(rng.choice(eq)) if eq else None)
    return other, same


class RawReadout:
    name = "raw"
    has_predictor = False

    @torch.no_grad()
    def __call__(self, x):
        pooled = x.mean(2) if x.dim() == 4 else x
        tok = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
        tok = tok.mean(-1) if tok.dim() == 3 else tok
        return pooled.cpu().numpy(), None, tok.cpu().numpy(), None


class PixelBaselineReadout:
    """Raw baseline for RGB inputs: frames average-pooled to 28x28 and flattened. No predictor."""
    name = "pixel"
    has_predictor = False

    @torch.no_grad()
    def __call__(self, x):
        b, t = x.shape[:2]
        small = torch.nn.functional.adaptive_avg_pool2d(x.flatten(0, 1), 28).reshape(b, t, -1)
        tok = (small[:, 1:] - small[:, :-1]).norm(dim=-1)
        return small.cpu().numpy(), None, tok.cpu().numpy(), None


class ModelReadout:
    has_predictor = True

    def __init__(self, name, model, cfg, geometry_pooling="mean", compute_predictor=True):
        self.name, self.model = name, model
        self.kind = cfg.get("pred_loss", "l2")
        self.patch = cfg.get("token_mode", "cls") == "patch"
        if geometry_pooling not in ("mean", "concat"):
            raise ValueError(f"unknown geometry pooling: {geometry_pooling}")
        self.geometry_pooling = geometry_pooling
        self.has_predictor = compute_predictor

    @torch.no_grad()
    def __call__(self, x):
        if getattr(self.model, "pixel_encoder", None) is not None:   # token_mode='raw': RGB frames in
            x = self.model.pixel_encoder(x)
        z = self.model.encoder(x)                                   # (B,T,P,D) or (B,T,D)
        if self.has_predictor:
            pred = self.model.predictor(z[:, :-1] if self.patch else z)
            pred = pred if self.patch else pred[:, :-1]             # pred[:, t] predicts z[:, t+1]
            pe = position_error(pred, z[:, 1:], self.kind)
            copy = position_error(z[:, :-1], z[:, 1:], self.kind)   # "nothing changes" baseline
        else:
            pe = copy = None
        pooled = (z.mean(2) if self.geometry_pooling == "mean" else z.flatten(2)) if z.dim() == 4 else z
        tok = (z[:, 1:] - z[:, :-1]).norm(dim=-1)
        tok = tok.mean(-1) if tok.dim() == 3 else tok
        return (pooled.cpu().numpy(), pe.cpu().numpy() if pe is not None else None,
                tok.cpu().numpy(), copy.cpu().numpy() if copy is not None else None)


def as_float(t):
    return t.float() / 255.0 if t.dtype == torch.uint8 else t


def build_condition(cond, X, rows, labels, donors, seed):
    ys, seams, valid = [], [], []
    for i in rows:
        x = as_float(X[i])
        rng = rng_for(seed, cond, i)
        if cond == "clean16":
            out = (x, [])
        elif cond == "clean12":
            out = clean_window(x, rng)
        else:
            kind = NEEDS_DONOR.get(cond)
            d_idx = donors[kind][i] if kind else None
            out = INJECTIONS[cond](x, rng, as_float(X[d_idx]) if d_idx is not None else None)
        valid.append(out is not None)
        y, s = out if out is not None else (x, [])
        ys.append(y)
        seams.append(s)
    return torch.stack(ys), seams, valid


def run_dataset(name, cfg, readouts, args, device, pixels=False, sample=None):
    t0 = time.time()
    if sample is not None:
        X, labels, _ = sample
    elif pixels:
        X, labels, _ = load_sample_pixels(name, cfg, args.n_clips, args.seed, args.frame_cache)
    else:
        X, labels, _ = load_sample(cfg, args.n_clips, args.seed)
    n = len(X)
    donors = dict(zip(("other", "same"), pick_donors(labels, args.seed)))
    conds = ["clean16", "clean12"] + list(args.injections)
    store = {(r.name, c): {"seams": [], "valid": [], "step": [], "tokstep": [], "pe": [], "copy": [],
                           **{k: [] for k in GEOMETRY_SCORES + PREDICTOR_SCORES}}
             for r in readouts for c in conds}
    print(f"[{name}] {n} clips, {len(set(labels))} classes, loaded in {time.time() - t0:.0f}s")
    for b0 in range(0, n, args.batch_size):
        rows = list(range(b0, min(n, b0 + args.batch_size)))
        for cond in conds:
            y, seams, valid = build_condition(cond, X, rows, labels, donors, args.seed)
            y = y.to(device)
            for r in readouts:
                pooled, pe, tok, copy = r(y)
                geo, step = geometry_scores(pooled)
                rec = store[(r.name, cond)]
                rec["seams"] += seams
                rec["valid"] += valid
                rec["step"].append(step)
                rec["tokstep"].append(tok)
                for k, v in geo.items():
                    rec[k].append(v)
                if pe is not None:
                    rec["pe"].append(pe)
                    rec["copy"].append(copy)
                    for k, v in predictor_scores(pe).items():
                        rec[k].append(v)
        print(f"[{name}] {min(n, b0 + args.batch_size)}/{n} clips  ({time.time() - t0:.0f}s)")
    out = {}
    for key, rec in store.items():
        o = {"seams": rec["seams"], "valid": np.asarray(rec["valid"])}
        for k, v in rec.items():
            if k not in ("seams", "valid") and v:
                o[k] = np.concatenate(v)
        out[key] = o
    return out, labels, time.time() - t0


def analyse(name, store, readouts, args):
    rows, loc_rows = [], []
    for r in readouts:
        score_names = list(GEOMETRY_SCORES) + (list(PREDICTOR_SCORES) if r.has_predictor else [])
        for inj in args.injections:
            p, c = store[(r.name, inj)], store[(r.name, REFERENCE[inj])]
            for s in score_names:
                pert = np.where(p["valid"], p[s], np.nan)
                rows.append({"dataset": name, "readout": r.name, "injection": inj, "score": s,
                             **compare(pert, c[s], args.n_boot, args.seed)})
            if inj in SEAM_INJECTIONS:
                seams = [s if v else None for s, v in zip(p["seams"], p["valid"])]
                signals = {"step": p["step"], "tokstep": p["tokstep"]}
                if r.has_predictor:
                    signals["pe"] = p["pe"]
                for sig, arr in signals.items():
                    loc_rows.append({"dataset": name, "readout": r.name, "injection": inj,
                                     "signal": sig, **localization(arr, seams)})
        base = store[(r.name, "clean16")]
        order = np.random.default_rng(args.seed).permutation(len(base["valid"]))
        a, b = order[: len(order) // 2], order[len(order) // 2:]
        m = min(len(a), len(b))
        for s in score_names:
            rows.append({"dataset": name, "readout": r.name, "injection": "null_split", "score": s,
                         **compare(base[s][a[:m]], base[s][b[:m]], args.n_boot, args.seed),
                         })
    df = pd.DataFrame(rows)
    df.loc[df.injection == "null_split", "paired_win"] = np.nan
    skill = {}
    for r in readouts:
        if r.has_predictor:
            c = store[(r.name, "clean16")]
            skill[r.name] = {"model_error": float(c["pe"].mean()), "copy_error": float(c["copy"].mean()),
                             "skill": float(1 - c["pe"].sum() / c["copy"].sum())}
    return df, pd.DataFrame(loc_rows), skill


def fmt(row):
    if row is None or not np.isfinite(row["auroc"]):
        return "-"
    return f"{row['auroc']:.2f} [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"


def headline_markdown(df, loc, skill, readouts, name):
    lines = [f"### {name}: predictor skill on clean 16-frame clips (1 - model error / copy-previous-frame error)", "",
             "| readout | model error | copy error | skill |", "|---|---|---|---|"]
    for rn, v in skill.items():
        lines.append(f"| {rn} | {v['model_error']:.4f} | {v['copy_error']:.4f} | {v['skill']:.2f} |")
    lines += ["", "Skill below 0 means the predictor is worse than assuming nothing changes; its error then says little "
              "about whether a clip is anomalous.", "",
              f"### {name}: AUROC of perturbed vs clean (0.5 = chance), 95% CI over clips", ""]
    head = ["injection", "reference"]
    cols = []
    for r in readouts:
        cols.append((r.name, "geom"))
        if r.has_predictor:
            cols.append((r.name, "pred"))
    head += [f"{rn} {'predictor' if k == 'pred' else 'geometry'}" for rn, k in cols]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for inj in list(HEADLINE):
        if inj != "null_split" and inj not in set(df.injection):
            continue
        ps, gs = HEADLINE[inj]
        cells = [f"**{inj}**", REFERENCE.get(inj, "other half")]
        for rn, k in cols:
            sc = ps if k == "pred" else gs
            sel = df[(df.readout == rn) & (df.injection == inj) & (df.score == sc)]
            cells.append(fmt(sel.iloc[0] if len(sel) else None) + f" ({sc})")
        lines.append("| " + " | ".join(cells) + " |")
    if len(loc):
        lines += ["", f"### {name}: does the largest per-transition value land within one step of the seam?",
                  "", "| injection | chance | " + " | ".join(
                      f"{r.name} {s}" for r in readouts for s in (("step", "tokstep", "pe") if r.has_predictor
                                                                    else ("step", "tokstep"))) + " |",
                  "|" + "---|" * (2 + sum(3 if r.has_predictor else 2 for r in readouts))]
        for inj in SEAM_INJECTIONS:
            sub = loc[loc.injection == inj]
            if sub.empty:
                continue
            cells = [f"**{inj}**", f"{sub.chance.iloc[0]:.2f}"]
            for r in readouts:
                for s in (("step", "tokstep", "pe") if r.has_predictor else ("step", "tokstep")):
                    q = sub[(sub.readout == r.name) & (sub.signal == s)]
                    cells.append(f"{q.hit.iloc[0]:.2f}" if len(q) and np.isfinite(q.hit.iloc[0]) else "-")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(dfs, readouts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dfs)
    injs = [i for i in HEADLINE if i in set(pd.concat(dfs.values()).injection)]
    base = next(r.name for r in readouts if not r.has_predictor) if any(not r.has_predictor for r in readouts) else None
    series = [(f"{base} geometry", base, 1, "#7a8b96")] if base else []
    for r in readouts:
        if r.has_predictor:
            series += [(f"{r.name} geometry", r.name, 1, "#1B6E8C"), (f"{r.name} predictor", r.name, 0, "#5545B8")]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names)), squeeze=False)
    w = 0.8 / len(series)
    for ax, nm in zip(axes[:, 0], names):
        df = dfs[nm]
        for k, (label, rn, geom, color) in enumerate(series):
            ys, lo, hi = [], [], []
            for inj in injs:
                sc = HEADLINE[inj][geom]
                q = df[(df.readout == rn) & (df.injection == inj) & (df.score == sc)]
                v = q.auroc.iloc[0] if len(q) else np.nan
                ys.append(v)
                lo.append(max(0.0, v - q.ci_lo.iloc[0]) if len(q) else 0)
                hi.append(max(0.0, q.ci_hi.iloc[0] - v) if len(q) else 0)
            ax.bar(np.arange(len(injs)) + k * w, ys, w, yerr=[lo, hi], color=color, label=label, capsize=2)
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(injs)) + 0.4 - w / 2)
        ax.set_xticklabels(injs, rotation=20)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("AUROC")
        ax.set_title(nm)
        ax.legend(ncol=len(series), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(DATASETS), help="repeatable; default k400 and ssv2")
    ap.add_argument("--model", action="append", metavar="NAME=RUN_DIR[@CKPT]",
                    help="repeatable; the run directory holds config.yaml and the checkpoint "
                         "(default best.ckpt). Default: vicreg3 (best) and vicreg3_last (last)")
    ap.add_argument("--ckpt-name", default="best.ckpt", help="checkpoint used when a --model has no @CKPT")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--injections", nargs="+", default=None, choices=list(INJECTIONS),
                    help="default: all, except the pixel-only ones for feature inputs")
    ap.add_argument("--frame-cache", default=str(HERE / "results" / "frame_cache"),
                    help="where decoded RGB frames are cached for token_mode='raw' checkpoints")
    ap.add_argument("--n-clips", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "vicreg3_vs_raw"))
    args = ap.parse_args()

    device = torch.device(args.device)
    specs = []
    for m in (args.model or []):
        nm, rest = m.split("=", 1)
        d, _, ck = rest.partition("@")
        specs.append((nm, Path(d), ck or args.ckpt_name))
    specs = specs or DEFAULT_MODELS
    readouts, meta = [], {}
    for nm, d, ck in specs:
        model, cfg, epoch = load_world_model(d / ck, d / "config.yaml", device, with_predictor=True)
        readouts.append(ModelReadout(nm, model, cfg))
        meta[nm] = {"checkpoint": str(d / ck), "epoch": epoch,
                    "straighten_weight": cfg.get("straighten_weight"), "pred_loss": cfg.get("pred_loss", "l2")}
    kinds = {getattr(r.model, "pixel_encoder", None) is not None for r in readouts}
    if len(kinds) > 1:
        raise SystemExit("do not mix token_mode='raw' (RGB input) checkpoints with feature-input ones in one run")
    pixels = kinds == {True}
    if not args.no_raw:
        readouts.insert(0, PixelBaselineReadout() if pixels else RawReadout())
    if args.injections is None:
        args.injections = [i for i in INJECTIONS if pixels or i not in PIXEL_ONLY]
    unsupported = set(args.injections) & set(PIXEL_ONLY) if not pixels else set()
    if unsupported:
        raise SystemExit(f"pixel-only injections require a token_mode='raw' checkpoint: {sorted(unsupported)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs, locs, md, npz = {}, {}, [], {}
    for name in (args.dataset or list(DATASETS)):
        store, labels, secs = run_dataset(name, DATASETS[name], readouts, args, device, pixels)
        dfs[name], locs[name], skill = analyse(name, store, readouts, args)
        dfs[name].to_csv(out_dir / f"{name}_auroc.csv", index=False)
        locs[name].to_csv(out_dir / f"{name}_localization.csv", index=False)
        md.append(headline_markdown(dfs[name], locs[name], skill, readouts, name))
        meta[name] = {"n_clips": len(labels), "n_classes": len(set(labels)), "seconds": round(secs),
                      "predictor_skill": skill}
        for (rn, c), rec in store.items():
            for k in list(GEOMETRY_SCORES) + list(PREDICTOR_SCORES):
                if k in rec:
                    npz[f"{name}|{rn}|{c}|{k}"] = rec[k]
            npz[f"{name}|{rn}|{c}|valid"] = rec["valid"]
    np.savez_compressed(out_dir / "scores.npz", **npz)
    (out_dir / "summary.md").write_text("\n".join(md))
    (out_dir / "config.json").write_text(json.dumps({"args": vars(args), "meta": meta}, indent=2, default=str))
    try:
        plot(dfs, readouts, out_dir / "injection_eval.png")
    except Exception as e:  # plotting must never lose the numbers
        print("plot skipped:", repr(e))
    print("\n" + "\n".join(md))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
