"""Calibrate the representation-probe metrics (SSv2): semantic probes, chirality, cross-clip
consistency.  Uses compute_probes.py's own functions unchanged wherever possible.

Per encoder, reports (see README for interpretation):
  semantic   linear / kNN balanced accuracy on content, change, content+change
             - noise: spread over 5 train-subsample seeds (the repo fixes seed 0), bootstrap CI over val clips
             - floor: label-permutation null (train labels shuffled, val labels true)
             - ceiling: same linear probe on ALL train clips; and a richer temporal-statistics feature set
  chirality  per-triplet ridge on concatenated frames
             - repo version vs feature-scale x0.1 / x10 (scale-sensitivity: ridge alpha=1, no scaling)
             - standardised + CV-alpha ridge (scale-free version)
             - order-blind control: every clip's frames randomly permuted before training and testing
             - floor: train labels shuffled within each triplet
             - CI: bootstrap over val clips within triplets
  consistency change-direction gap and TCC tau: spread over pair-sampling seeds; label-shuffled floor
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "representation_probes"))
sys.path.insert(0, str(HERE.parent))
import compute_probes as cp  # noqa: E402
from eval_common.loading import load_matched_features  # noqa: E402

MAN = "/home/sachish/proc-planning/data/chirality_in_action/SSv2/ssv2_{}.csv"
FEATS = "/mnt/rdata7/proprogress/data/SSv2/precomputed_features/{}/ssv2_{}.pt"
SPEC = cp.DATASETS["ssv2"]


def load_split(enc=None, z_override=None):
    """Returns tr_full_df, va_df, classes, z_tr_full (N,T,D), z_va. If z_override=(z_tr,z_va) is given
    they must be aligned with the *matched* manifests, i.e. built by this function's loader."""
    tr_df, va_df, tr_full, classes, dropped = cp.prepare_label_splits(MAN.format("train"), MAN.format("validation"),
                                                                       SPEC, 20, 5, 200, 0)
    ftr, tr_m = load_matched_features(FEATS.format(enc, "train"), tr_full)
    fva, va_m = load_matched_features(FEATS.format(enc, "validation"), va_df)
    return tr_m, va_m, classes, ftr.numpy().astype(np.float64), fva.numpy().astype(np.float64)


def subsample_idx(tr_df, seed, per_class=200):
    d = tr_df.reset_index(drop=True)
    return d.sample(frac=1, random_state=seed).groupby("label").head(per_class).index.to_numpy()


def lin_fit_predict(xtr, ytr, xva, C=None, seed=0):
    if C is None:
        m, C = cp.fit_linear(xtr, ytr, seed)
        return m.predict(xva), C
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=C)).fit(xtr, ytr)
    return m.predict(xva), C


def temporal_stats(z, edge=4):
    """Richer clip descriptor used only for the ceiling probe."""
    return np.concatenate([z.mean(1), z.std(1), z[:, :edge].mean(1), z[:, -edge:].mean(1),
                           z[:, -edge:].mean(1) - z[:, :edge].mean(1)], 1)


def bal_boot(y, pred, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    s = [balanced_accuracy_score(y[i], pred[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.quantile(s, .025)), float(np.quantile(s, .975))


def chirality(z_tr, tr_df, z_va, va_df, make_clf, scale=1.0, order_blind_seed=None, shuffle_labels_seed=None,
              boot=0):
    """mean per-triplet accuracy (+ optional bootstrap CI over val clips within triplets)."""
    ztr, zva = z_tr * scale, z_va * scale
    if order_blind_seed is not None:
        rng = np.random.default_rng(order_blind_seed)
        ztr = np.take_along_axis(ztr, np.argsort(rng.random(ztr.shape[:2]), 1)[:, :, None], 1)
        zva = np.take_along_axis(zva, np.argsort(rng.random(zva.shape[:2]), 1)[:, :, None], 1)
    xtr, xva = ztr.reshape(len(ztr), -1), zva.reshape(len(zva), -1)
    t_tr, t_va = tr_df.chiral_triplet_id.to_numpy(), va_df.chiral_triplet_id.to_numpy()
    y_tr, y_va = tr_df.chiral_label.astype(float).to_numpy(), va_df.chiral_label.astype(float).to_numpy()
    rng = np.random.default_rng(shuffle_labels_seed)
    accs, hits = [], {}
    for tid in sorted(set(t_tr) & set(t_va)):
        itr, iva = np.flatnonzero(t_tr == tid), np.flatnonzero(t_va == tid)
        if len(np.unique(y_tr[itr])) < 2:
            continue
        yt = y_tr[itr].copy()
        if shuffle_labels_seed is not None:
            yt = rng.permutation(yt)
            if len(np.unique(yt)) < 2:
                continue
        pred = make_clf().fit(xtr[itr], yt).predict(xva[iva])
        hit = (pred == y_va[iva]).astype(float)
        accs.append(hit.mean())
        hits[tid] = hit
    out = {"mean_accuracy": float(np.mean(accs)), "n_triplets": len(accs)}
    if boot:
        rng = np.random.default_rng(1)
        bs = [np.mean([h[rng.integers(0, len(h), len(h))].mean() for h in hits.values()]) for _ in range(boot)]
        out["ci_lo"], out["ci_hi"] = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    return out


def probe_suite(z_tr, tr_df, z_va, va_df, classes, light=False, seeds=5, perms=10, tcc_seeds=5):
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_va = va_df.label.map(cls_idx).to_numpy()
    y_full = tr_df.label.map(cls_idx).to_numpy()
    groups_va = va_df.group.to_numpy()
    res = {"n_classes": len(classes), "chance": 1 / len(classes)}
    v_va = cp.clip_vectors(z_va, 4)
    v_full = cp.clip_vectors(z_tr, 4)
    n_seed = 1 if light else seeds

    for name in ("content", "change", "content+change"):
        lin_acc, knn_acc, lin_pred0 = [], [], None
        for s in range(n_seed):
            idx = subsample_idx(tr_df, s)
            blocks = 2 if name == "content+change" else 1
            kn = cp.knn_predict(v_full[name][idx], y_full[idx], v_va[name], 20, len(classes), blocks)
            knn_acc.append(balanced_accuracy_score(y_va, kn))
            if name == "content+change" or not light:
                pr, C = lin_fit_predict(v_full[name][idx], y_full[idx], v_va[name], seed=s)
                lin_acc.append(balanced_accuracy_score(y_va, pr))
                if s == 0:
                    lin_pred0, C0, knn_pred0 = pr, C, kn
        r = {"knn_mean": float(np.mean(knn_acc)), "knn_sd": float(np.std(knn_acc, ddof=1)) if n_seed > 1 else None,
             "knn_seeds": knn_acc}
        if lin_acc:
            lo, hi = bal_boot(y_va, lin_pred0)
            r.update({"linear_mean": float(np.mean(lin_acc)), "linear_seed0": float(lin_acc[0]),
                      "linear_sd": float(np.std(lin_acc, ddof=1)) if n_seed > 1 else None, "linear_seeds": lin_acc,
                      "linear_boot_lo": lo, "linear_boot_hi": hi, "C": C0})
        res[name] = r
        if name == "content+change":
            res["_pred_linear_seed0"] = lin_pred0
            res["_pred_knn_seed0"] = knn_pred0
        if light:
            continue
        # floor: permuted train labels
        if name == "content+change":
            idx = subsample_idx(tr_df, 0)
            rng = np.random.default_rng(0)
            perm_acc = []
            for p in range(perms):
                yp = rng.permutation(y_full[idx])
                pr, _ = lin_fit_predict(v_full[name][idx], yp, v_va[name], C=C0)
                perm_acc.append(balanced_accuracy_score(y_va, pr))
            res[name]["perm_null_mean"], res[name]["perm_null_sd"] = float(np.mean(perm_acc)), float(np.std(perm_acc, ddof=1))
            # ceilings
            pr, _ = lin_fit_predict(v_full[name], y_full, v_va[name], C=C0)
            res[name]["ceiling_full_train_linear"] = float(balanced_accuracy_score(y_va, pr))
            ts_tr, ts_va = temporal_stats(z_tr), temporal_stats(z_va)
            pr, _ = lin_fit_predict(ts_tr, y_full, ts_va, C=0.01)
            res[name]["ceiling_temporal_stats_full_train"] = float(balanced_accuracy_score(y_va, pr))

    # chirality
    ridge = lambda: RidgeClassifier()
    ridge_cv = lambda: make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(0, 5, 8)))
    ch = {"repo_ridge": chirality(z_tr, tr_df, z_va, va_df, ridge, boot=0 if light else 500)}
    if not light:
        for sc in (0.1, 10.0):
            ch[f"repo_ridge_scale{sc:g}"] = chirality(z_tr, tr_df, z_va, va_df, ridge, scale=sc)
        ch["std_ridgecv"] = chirality(z_tr, tr_df, z_va, va_df, ridge_cv)
        ch["order_blind"] = chirality(z_tr, tr_df, z_va, va_df, ridge, order_blind_seed=0)
        nulls = [chirality(z_tr, tr_df, z_va, va_df, ridge, shuffle_labels_seed=s)["mean_accuracy"] for s in range(8)]
        ch["label_shuffle_null_mean"], ch["label_shuffle_null_sd"] = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
    res["chirality"] = ch

    # cross-clip consistency (val only)
    gaps, taus, taus_null = [], [], []
    for s in range(1 if light else tcc_seeds):
        taus.append(cp.tcc_kendall_tau(z_va, y_va, groups_va, 2000, s)["mean"])
    gap = cp.change_consistency(v_va["change"], y_va, groups_va)
    res["consistency"] = {"change_gap": gap["gap"], "tcc_tau_mean": float(np.mean(taus)),
                          "tcc_tau_sd": float(np.std(taus, ddof=1)) if len(taus) > 1 else None}
    if not light:
        rng = np.random.default_rng(0)
        gnull = [cp.change_consistency(v_va["change"], rng.permutation(y_va), groups_va)["gap"] for _ in range(10)]
        tnull = [cp.tcc_kendall_tau(z_va, rng.permutation(y_va), groups_va, 1000, s)["mean"] for s in range(3)]
        res["consistency"].update({"change_gap_null_mean": float(np.mean(gnull)), "change_gap_null_sd": float(np.std(gnull, ddof=1)),
                                   "tcc_tau_null_mean": float(np.mean(tnull))})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--light", action="store_true")
    a = ap.parse_args()
    tr_m, va_m, classes, ztr, zva = load_split(a.encoder)
    r = probe_suite(ztr, tr_m, zva, va_m, classes, light=a.light)
    r["encoder"] = a.encoder
    preds = {k: r.pop(k) for k in list(r) if k.startswith("_pred")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    np.savez(Path(a.out).with_suffix(".preds.npz"), y_va=va_m.label.map({c: i for i, c in enumerate(classes)}).to_numpy(), **preds)
    print("saved", a.out)
