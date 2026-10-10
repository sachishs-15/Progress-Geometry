"""Does the representation still carry meaning? Label- and train-split-based probes on
frozen per-frame latents:

1. Semantic probes -- k-NN (k=20, cosine, majority vote) and a linear probe (logistic
   regression, balanced class weights, C by 3-fold CV), fit on train, scored on val.
   SSv2 label = verb; Ego4D GoalStep label = leaf step family (the leaf_step_category
   prefix before ':').
2. Chirality (SSv2 only) -- Chirality in Action's protocol on the concatenation of all
   frame latents: one binary forward/reverse classifier per chiral triplet (default
   RidgeClassifier), fit on all train clips of the triplet, accuracy averaged over
   triplets; plus whether predictions flip when a clip is reversed.
3. Cross-clip consistency -- do clips with the same label move in the same direction
   (change-vector cosine, same vs different label), and can their frames be aligned in
   order (TCC-style Kendall's tau of nearest-neighbour frame alignment)?
4. Reconstruction R² (checkpoints only) -- how much of the raw DINOv2 per-frame feature an
   MLP decoder can recover from the latent (fit on train frames, R² on val frames).

Each clip's latent z (T, D) becomes two vectors: content = mean over frames (what is in
the clip; order-blind) and change = mean(last --edge-frames) − mean(first --edge-frames)
(which way it moved; averaged at each end because per-frame noise is heavy). Probes are
run on content, change, and both concatenated.

Encoded latents are cached under --cache-dir so reruns skip re-encoding.

Usage (raw DINOv2 CLS, SSv2):
    python compute_probes.py --dataset ssv2 --raw cls \\
        --train-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_train.pt \\
        --train-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_train.csv \\
        --val-features /mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2/ssv2_validation.pt \\
        --val-manifest /home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_validation.csv \\
        --out results/ssv2_raw_dinov2_cls.json
Checkpoint: replace --raw with --checkpoint <best.ckpt> --config <config.yaml> and point
the features at the patch files (dinov2_noreg) the checkpoint was trained on.
"""
import argparse
import hashlib
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_common.loading import (RAW_MODES, encode_world_model, load_matched_features,  # noqa: E402
                                 load_world_model, reduce_raw)

DATASETS = {
    "ssv2": {"label": lambda d: d["verb"], "group": lambda d: d["video_path"], "pair": "chiral_triplet_id"},
    "ego4d": {"label": lambda d: d["leaf_step_category"].str.split(":").str[0].str.strip(),
              "group": lambda d: d["video_uid"], "pair": None},
}


@lru_cache(maxsize=None)
def _sha256(path_string):
    """Content identity for the comparatively small checkpoint/config inputs."""
    h = hashlib.sha256()
    with open(path_string, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _feature_identity(path_string):
    """Identity without hashing multi-GB tensors: resolved path, size and nanosecond mtime."""
    path = Path(path_string)
    files = [path] if path.exists() else sorted(path.parent.glob(path.name + ".shard*.pt"))
    return [{"path": str(p.resolve()), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in files]


def _cache_metadata(split, df, features_path, args, keep_target):
    metadata = {
        "schema_version": 2,
        "split": split,
        "requested_video_paths": [str(path) for path in df[args.path_col]],
        "path_col": args.path_col,
        "features": _feature_identity(features_path),
        "raw_mode": args.raw,
        "keep_target": bool(keep_target),
    }
    if not args.raw:
        metadata.update({
            "checkpoint": {"path": str(Path(args.checkpoint).resolve()),
                           "sha256": _sha256(str(Path(args.checkpoint).resolve()))},
            "config": {"path": str(Path(args.config).resolve()),
                       "sha256": _sha256(str(Path(args.config).resolve()))},
        })
    return metadata


def get_latents(split, df, features_path, args, model, device, keep_target=True, chunk=1500):
    """Per-frame latents z (N,T,D), raw per-frame DINOv2 target (N,T,384) for the
    reconstruction probe (checkpoint mode only, keep_target=True), and the matched manifest
    rows. Encodes `chunk` clips at a time so large splits never sit in memory whole."""
    cache = Path(args.cache_dir) / f"{args.cache_name}_{split}.npz"
    metadata_json = json.dumps(
        _cache_metadata(split, df, features_path, args, keep_target),
        sort_keys=True, separators=(",", ":"),
    )
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        cached_metadata = str(c["cache_metadata"].item()) if "cache_metadata" in c else None
        if cached_metadata == metadata_json:
            print(f"Loaded cached {split} latents from {cache}")
            matched = pd.read_json(io.StringIO(str(c["matched_df"].item())), orient="split",
                                   dtype=False)
            return c["z"], (c["target"] if c["target"].size else None), matched
        print(f"Ignoring stale {split} cache with a different extraction identity: {cache}")
    zs, targets, parts = [], [], []
    for s in range(0, len(df), chunk):
        feats, matched = load_matched_features(features_path, df.iloc[s:s + chunk], args.path_col)
        if args.raw:
            zs.append(reduce_raw(feats, args.raw))
        else:
            zs.append(encode_world_model(model, feats, device, args.batch_size))
            # keep_target only makes sense for a DINOv2 cls (3D) or patch (4D) --features
            # tensor -- there's no DINOv2 ground truth to reconstruct back to for
            # token_mode='raw' checkpoints (5D raw pixel input), and treating the pixel
            # tensor itself as the target OOMs the reconstruction MLP (31GB+ normalize).
            if keep_target and feats.dim() in (3, 4):
                targets.append((feats.mean(dim=2) if feats.dim() == 4 else feats).numpy())
        parts.append(matched)
        del feats
    z, matched = np.concatenate(zs), pd.concat(parts, ignore_index=True)
    target = np.concatenate(targets) if targets else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, z=z, target=target if target is not None else np.empty(0),
             video_paths=np.asarray(matched[args.path_col]), cache_metadata=metadata_json,
             matched_df=matched.to_json(orient="split"))
    return z, target, matched


def clip_vectors(z, edge):
    content = z.mean(1)
    change = z[:, -edge:].mean(1) - z[:, :edge].mean(1)
    return {"content": content, "change": change, "content+change": np.concatenate([content, change], 1)}


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def knn_predict(train_x, train_y, val_x, k, n_classes, blocks):
    """Cosine k-NN, majority vote. Multi-block inputs are L2-normalised per block so each
    block counts equally."""
    tr = np.concatenate([l2n(b) for b in np.split(train_x, blocks, axis=1)], 1)
    va = np.concatenate([l2n(b) for b in np.split(val_x, blocks, axis=1)], 1)
    preds = np.empty(len(va), dtype=int)
    for s in range(0, len(va), 1024):
        top = np.argpartition(-(va[s:s + 1024] @ tr.T), k, axis=1)[:, :k]
        preds[s:s + 1024] = [np.bincount(train_y[row], minlength=n_classes).argmax() for row in top]
    return preds


def fit_linear(train_x, train_y, seed):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    search = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                          scoring="balanced_accuracy", n_jobs=3)
    search.fit(train_x, train_y)
    return search.best_estimator_, search.best_params_["logisticregression__C"]


def bootstrap_ci(y, pred, groups, n, seed):
    """95% CI of balanced accuracy, resampling whole groups (videos) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        stats.append(balanced_accuracy_score(y[idx], pred[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def scores(y, pred, classes, general_mask=None):
    out = {"acc": float((y == pred).mean()), "balanced_acc": float(balanced_accuracy_score(y, pred))}
    if general_mask is not None:
        keep = ~general_mask[y]
        out["balanced_acc_excl_general"] = float(balanced_accuracy_score(y[keep], pred[keep]))
    return out


def make_chiral_clf(name):
    """Chirality in Action's classifiers (lift/feat_utils.py get_linear_probe_accuracy),
    same defaults: RidgeClassifier(), LogisticRegression(), SVC(kernel='rbf')."""
    if name == "ridge":
        return RidgeClassifier()
    if name == "logistic":
        return LogisticRegression()
    return SVC(kernel="rbf")


def chirality_concat_probe(z_tr, tr_df, z_va, va_df, clf_name):
    """Chirality in Action's chiral-action-recognition protocol on the concatenation of all
    T frame latents (T*D per clip, which keeps frame order): for each chiral triplet in
    both splits, fit a fresh binary classifier (chiral_label: 1 forward / 0 reverse) on
    all its train clips, score accuracy on its val clips, and average over triplets.
    Triplets with only one side in train are skipped, as in CiA.

    reversal_flip: among a triplet's correctly classified val clips, the fraction whose
    prediction flips to the other side when the clip's frames are reversed (exact for
    per-frame encoders, where reversing the video just reverses z)."""
    x_tr, x_va = z_tr.reshape(len(z_tr), -1), z_va.reshape(len(z_va), -1)
    x_rev = z_va[:, ::-1].reshape(len(z_va), -1)
    t_tr, t_va = tr_df.chiral_triplet_id.to_numpy(), va_df.chiral_triplet_id.to_numpy()
    y_tr, y_va = tr_df.chiral_label.astype(float).to_numpy(), va_df.chiral_label.astype(float).to_numpy()
    per, skipped = [], []
    for tid in sorted(set(t_tr) & set(t_va)):
        itr, iva = np.flatnonzero(t_tr == tid), np.flatnonzero(t_va == tid)
        if len(np.unique(y_tr[itr])) < 2:
            skipped.append(str(tid))
            continue
        clf = make_chiral_clf(clf_name).fit(x_tr[itr], y_tr[itr])
        pred, rev = clf.predict(x_va[iva]), clf.predict(x_rev[iva])
        ok = pred == y_va[iva]
        verbs = va_df.verb.to_numpy()[iva]
        per.append({
            "triplet_id": str(tid),
            "verb_forward": str(verbs[y_va[iva] == 1][0]) if (y_va[iva] == 1).any() else None,
            "verb_reverse": str(verbs[y_va[iva] == 0][0]) if (y_va[iva] == 0).any() else None,
            "accuracy": float(ok.mean()),
            "reversal_flip": float((rev[ok] != pred[ok]).mean()) if ok.any() else None,
            "n_train": int(len(itr)), "n_val": int(len(iva)),
        })
    flips = [r["reversal_flip"] for r in per if r["reversal_flip"] is not None]
    return {"classifier": clf_name, "features": f"concat of all frames ({x_tr.shape[1]} dims)",
            "mean_accuracy": float(np.mean([r["accuracy"] for r in per])),
            "mean_reversal_flip": float(np.mean(flips)) if flips else None,
            "n_triplets": len(per), "skipped_triplets": skipped, "per_triplet": per}


def change_consistency(change, y, groups):
    """Mean cosine between change vectors of different clips from different videos:
    same label vs different label."""
    u = l2n(change)
    sims = u @ u.T
    same_label = y[:, None] == y[None, :]
    valid = groups[:, None] != groups[None, :]
    within, between = sims[same_label & valid].mean(), sims[~same_label & valid].mean()
    return {"within_label_cos": float(within), "between_label_cos": float(between), "gap": float(within - between)}


def tcc_kendall_tau(z, y, groups, n_pairs, seed):
    """TCC-style alignment: for random same-label clip pairs (A, B) from different videos,
    subtract each clip's own mean latent, map each frame of A to its nearest (Euclidean)
    frame of B, and take Kendall's tau between A's frame order and the matched order.
    1 = frames align in order; ~0 = no temporal alignment. Centering removes each clip's
    overall appearance (object, background), which otherwise dwarfs within-clip movement
    and makes every frame of A match the same one or two frames of B.

    degenerate_fraction: pairs where all of A's frames match a single frame of B (tau
    undefined, counted as 0). median_distinct_matches: how many of B's T frames get used."""
    rng = np.random.default_rng(seed)
    by_label = pd.Series(np.arange(len(y))).groupby(y).apply(list)
    labels = [lab for lab, idx in by_label.items() if len(idx) >= 2]
    zc = z - z.mean(axis=1, keepdims=True)
    taus, distinct = [], []
    for _ in range(n_pairs * 5):
        if len(taus) >= n_pairs:
            break
        a, b = rng.choice(by_label[labels[rng.integers(len(labels))]], 2, replace=False)
        if groups[a] == groups[b]:
            continue
        nn = np.linalg.norm(zc[a][:, None] - zc[b][None], axis=-1).argmin(1)
        tau = kendalltau(np.arange(len(nn)), nn).statistic
        taus.append(0.0 if np.isnan(tau) else tau)
        distinct.append(len(np.unique(nn)))
    distinct = np.asarray(distinct)
    return {"mean": float(np.mean(taus)), "median": float(np.median(taus)), "n_pairs": len(taus),
            "degenerate_fraction": float(np.mean(distinct == 1)),
            "median_distinct_matches": float(np.median(distinct))}


def mlp_reconstruction_r2(z_tr, t_tr, z_va, t_va, device, seed, hidden=512, max_epochs=200, patience=10):
    """R² of an MLP decoder from per-frame latent to raw per-frame DINOv2 feature.
    z_*: (N, T, D) latents, t_*: (N, T, F) targets. Every frame is a sample. Inputs and
    targets are standardized with train statistics; 10% of train CLIPS (not frames, so
    frames of one clip never straddle the split) are held out for early stopping. R² is
    variance-weighted over the F outputs, in the original target units, on val frames."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hold = rng.permutation(len(z_tr))[: max(1, len(z_tr) // 10)]
    fit = np.setdiff1d(np.arange(len(z_tr)), hold)
    d, f = z_tr.shape[-1], t_tr.shape[-1]
    flat = lambda a, idx: torch.tensor(a[idx].reshape(-1, a.shape[-1]), dtype=torch.float32)
    x_fit, y_fit, x_hold, y_hold = flat(z_tr, fit), flat(t_tr, fit), flat(z_tr, hold), flat(t_tr, hold)
    xm, xs = x_fit.mean(0), x_fit.std(0) + 1e-6
    ym, ys = y_fit.mean(0), y_fit.std(0) + 1e-6
    norm = lambda x, y: (((x - xm) / xs).to(device), ((y - ym) / ys).to(device))
    x_fit, y_fit = norm(x_fit, y_fit)
    x_hold, y_hold = norm(x_hold, y_hold)
    mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.GELU(), torch.nn.Linear(hidden, f)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        mlp.train()
        for b in torch.randperm(len(x_fit), device=device).split(1024):
            opt.zero_grad()
            torch.nn.functional.mse_loss(mlp(x_fit[b]), y_fit[b]).backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(mlp(x_hold), y_hold).item()
        if loss < best - 1e-4:
            best, best_state, bad = loss, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    mlp.load_state_dict(best_state)
    with torch.no_grad():
        x_va = ((flat(z_va, np.arange(len(z_va))) - xm) / xs).to(device)
        pred = (mlp(x_va).cpu() * ys + ym).numpy()
    return float(r2_score(t_va.reshape(-1, f), pred, multioutput="variance_weighted"))


def prepare_label_splits(train_manifest, val_manifest, spec, min_train_per_class,
                          min_val_per_class, max_train_per_class, seed):
    """Labels, class filtering, per-class train subsample -- the same manifest-only setup
    __main__ does before touching any encoder, pulled out so it can be reused directly
    from Python (e.g. once per dataset, then reused across many encoders/poolings)."""
    tr_df = pd.read_csv(train_manifest, dtype=str)
    va_df = pd.read_csv(val_manifest, dtype=str)
    for d in (tr_df, va_df):
        d["label"], d["group"] = spec["label"](d), spec["group"](d)
    tr_counts, va_counts = tr_df.label.value_counts(), va_df.label.value_counts()
    classes = sorted(c for c in va_counts.index
                     if tr_counts.get(c, 0) >= min_train_per_class and va_counts[c] >= min_val_per_class)
    dropped = sorted(set(va_counts.index) - set(classes))
    tr_df = tr_df[tr_df.label.isin(classes)]
    tr_full = tr_df
    tr_df = tr_df.sample(frac=1, random_state=seed).groupby("label").head(max_train_per_class)
    va_df = va_df[va_df.label.isin(classes)]
    return tr_df, va_df, tr_full, classes, dropped


def compute_probes_dict(z_tr, tgt_tr, tr_df, z_va, tgt_va, va_df, classes, dropped,
                         args, spec, device, encoder_label, epoch=None,
                         z_full=None, tr_full=None):
    """Everything __main__ does once it already has z_tr/z_va (and optionally z_full,
    the un-capped full-train-set encoding used only for the chirality concat_probe) in
    memory -- no file path, no model, no get_latents() call. Lets a caller who already
    has an in-memory encoding (e.g. a checkpoint's own encoder output, pooled some
    non-standard way, never written to a --features file) get the full probes dict.
    z_full/tr_full may be omitted (chirality section is then skipped) if the caller has
    no reason to build the uncapped full-train encoding separately.
    """
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_tr, y_va = tr_df.label.map(cls_idx).to_numpy(), va_df.label.map(cls_idx).to_numpy()
    groups_va = va_df.group.to_numpy()
    general = np.array([c.startswith("General") for c in classes]) if args.dataset == "ego4d" else None

    v_tr, v_va = clip_vectors(z_tr, args.edge_frames), clip_vectors(z_va, args.edge_frames)
    result = {
        "encoder": encoder_label, "checkpoint_epoch": epoch,
        "dataset": args.dataset, "classes": classes, "dropped_classes": dropped,
        "n_train": len(y_tr), "n_val": len(y_va),
        "chance": {"uniform": 1 / len(classes), "majority": float(np.bincount(y_va).max() / len(y_va))},
        "probes": {}, "chirality": {},
    }
    for name in v_tr:
        blocks = 2 if name == "content+change" else 1
        knn = knn_predict(v_tr[name], y_tr, v_va[name], args.k, len(classes), blocks)
        lin, C = fit_linear(v_tr[name], y_tr, args.seed)
        lin_pred = lin.predict(v_va[name])
        result["probes"][name] = {
            "knn": scores(y_va, knn, classes, general),
            "linear": {**scores(y_va, lin_pred, classes, general), "C": C,
                       "balanced_acc_ci95": bootstrap_ci(y_va, lin_pred, groups_va, args.bootstrap, args.seed)},
        }
        print(f"[{name}] knn bal_acc {result['probes'][name]['knn']['balanced_acc']:.3f} | "
              f"linear bal_acc {result['probes'][name]['linear']['balanced_acc']:.3f}")

    if spec["pair"] and z_full is not None:
        result["chirality"]["concat_probe"] = chirality_concat_probe(z_full, tr_full, z_va, va_df, args.chiral_clf)
        c = result["chirality"]["concat_probe"]
        print(f"chirality concat probe ({c['classifier']}): mean accuracy {c['mean_accuracy']:.3f} over "
              f"{c['n_triplets']} triplets, reversal flip {c['mean_reversal_flip']:.3f}")

    result["consistency"] = {
        "change_direction": change_consistency(v_va["change"], y_va, groups_va),
        "tcc_kendall_tau": tcc_kendall_tau(z_va, y_va, groups_va, args.tcc_pairs, args.seed),
    }

    if tgt_tr is not None:
        result["reconstruction_r2"] = mlp_reconstruction_r2(z_tr, tgt_tr, z_va, tgt_va, device, args.seed)
        result["reconstruction_probe"] = "mlp: 2 hidden layers x 512, GELU, early-stopped"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--raw", choices=RAW_MODES)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--path-col", default="video_path")
    parser.add_argument("--max-train-per-class", type=int, default=200)
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-val-per-class", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--chiral-clf", choices=["ridge", "logistic", "kernel_svm"], default="ridge",
                        help="classifier for the per-triplet chirality probe (Chirality in Action default: ridge)")
    parser.add_argument("--edge-frames", type=int, default=4)
    parser.add_argument("--tcc-pairs", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "cache"))
    parser.add_argument("--cache-name", help="defaults to the --out file's stem")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.raw) == bool(args.checkpoint and args.config):
        raise SystemExit("pass exactly one of --raw or --checkpoint + --config")
    args.cache_name = args.cache_name or Path(args.out).stem
    spec = DATASETS[args.dataset]
    device = torch.device(args.device)

    tr_df, va_df, tr_full, classes, dropped = prepare_label_splits(
        args.train_manifest, args.val_manifest, spec,
        args.min_train_per_class, args.min_val_per_class, args.max_train_per_class, args.seed,
    )
    print(f"{len(classes)} classes kept, dropped {dropped}; train {len(tr_df)} clips, val {len(va_df)} clips")

    model, epoch = None, None
    if not args.raw:
        model, _, epoch = load_world_model(args.checkpoint, args.config, device)
    z_tr, tgt_tr, tr_df = get_latents("train", tr_df, args.train_features, args, model, device)
    z_va, tgt_va, va_df = get_latents("val", va_df, args.val_features, args, model, device)

    z_full, tr_full_out = None, None
    if spec["pair"]:
        # Chirality in Action protocol on concatenated frames, with ALL train clips
        z_full, _, tr_full_out = get_latents("train_full", tr_full, args.train_features, args, model, device,
                                             keep_target=False)

    result = compute_probes_dict(
        z_tr, tgt_tr, tr_df, z_va, tgt_va, va_df, classes, dropped, args, spec, device,
        encoder_label=f"raw:{args.raw}" if args.raw else args.checkpoint, epoch=epoch,
        z_full=z_full, tr_full=tr_full_out,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}")
    print(json.dumps({k: result[k] for k in ("chance", "consistency")}, indent=1))
    if "reconstruction_r2" in result:
        print("reconstruction_r2:", result["reconstruction_r2"])
