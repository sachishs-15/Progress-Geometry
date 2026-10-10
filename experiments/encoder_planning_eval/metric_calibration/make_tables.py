"""Markdown tables for REPORT.md, generated from results/ so no number is transcribed by hand.
    python make_tables.py   ->  results/tables.md  (and returns dict name -> markdown for build_report.py)
"""
import functools
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import noise

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i, _ in enumerate(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def get(j, p):
    try:
        return functools.reduce(lambda a, k: a[k], p, j)
    except (KeyError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------------------------- T1
def win(c, k):
    if k == "direction_reversal_fraction":
        return "step at r≈1"
    row = c.loc["trajectory_straightness_raw"] if k == "trajectory_straightness" else c.loc[k]
    return f"{row.win_lo:.3g} – {row.win_hi:.3g}"


def t_card():
    c = pd.read_csv(RES / "metric_card_trajectory_dinov2.csv").set_index("metric")
    keys = ["time_progress_spearman", "pc1_time_spearman", "monotonic_progress_fraction", "progress_efficiency",
            "adjacent_temporal_nn_fraction", "distance_to_goal_spearman", "trajectory_straightness", "mean_velocity_cosine_raw",
            "mean_chord_alignment_raw", "direction_reversal_fraction"]
    rows = []
    for k in keys:
        r = c.loc[k]
        rows.append({"metric": k, "floor": r.floor, "DINOv2-S": r.real, "random walk": r.decoy_brownian, "smooth σ=1": r.gauss1,
                     "ceiling": r.ceiling, "position": r.pos, "MDD": r.mdd, "range/MDD": r.steps, "ICC": r.icc,
                     "sensitive window r (10–90%)": win(c, k), "whiten (×MDD)": r.d_whiten_mdd, "smooth σ=1 (% range)": 100 * r.d_gauss1_range,
                     "reversal-blind": "yes" if r.rev_sym else "no"})
    return md(pd.DataFrame(rows), 3)


# ---------------------------------------------------------------------------------------------- T2
def t_encoders():
    tc = pd.read_csv(RES / "trajectory_calibration.csv")
    encs = ["dinov2", "dinov2_base", "dinov2_large", "dinov2_giant", "dinov3", "clip", "siglip", "levljepa", "vjepa2", "vjepa2.1"]
    mets = [("time_progress_spearman", "time_prog."), ("trajectory_straightness", "straight."), ("mean_velocity_cosine_raw", "vel.cos"),
            ("direction_reversal_fraction", "dir.rev."), ("adjacent_temporal_nn_fraction", "adj.NN"), ("distance_to_goal_spearman", "dist→goal")]
    rows = []
    for e in encs:
        r = {"encoder": e}
        for m, lab in mets:
            v = lambda ref: float(tc[(tc.encoder == e) & (tc.ref == ref) & (tc.metric == m)].value.iloc[0])
            r[lab] = f"{v('REAL'):.3f}"
        rows.append(r)
    fl = {"encoder": "*floor (shuffled), range over encoders*"}
    sm = {"encoder": "*σ=1 smoothing of the same features, range*"}
    for m, lab in mets:
        vf = [float(tc[(tc.encoder == e) & (tc.ref == "shuffle_time") & (tc.metric == m)].value.iloc[0]) for e in encs]
        vs = [float(tc[(tc.encoder == e) & (tc.ref == "gauss1") & (tc.metric == m)].value.iloc[0]) for e in encs]
        rng2 = lambda v: f"{min(v):.2f}" if f"{min(v):.2f}" == f"{max(v):.2f}" else f"{min(v):.2f} to {max(v):.2f}"
        fl[lab] = rng2(vf)
        sm[lab] = rng2(vs)
    rows += [fl, sm]
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- T3
def t_planning():
    d = pd.read_csv(RES / "planning_calibration.csv")
    p = d[~d.predictor.str.startswith("PAIR")]
    r2 = p.pivot(index="encoder", columns="predictor", values="r2")
    cos = p.pivot(index="encoder", columns="predictor", values="cos_median")
    pair = d[d.predictor == "PAIR_interp_minus_mid"].set_index("encoder")
    order = r2["interp"].sort_values(ascending=False).index
    rows = []
    for e in order:
        rows.append({"encoder": e, "R² static midpoint": r2.loc[e, "mid_const"], "R² interp (reported)": r2.loc[e, "interp"],
                     "temporal part [95% CI]": f"+{pair.loc[e, 'r2']:.3f} [{pair.loc[e, 'r2_lo']:.3f}, {pair.loc[e, 'r2_hi']:.3f}]",
                     "R² ridge": r2.loc[e, "ridge_resid"], "R² adjacent-frame oracle": r2.loc[e, "neighbors"],
                     "R² noise ceiling": r2.loc[e, "noise_ceiling"],
                     "cos: mean-predictor": cos.loc[e, "mean"], "cos: interp": cos.loc[e, "interp"]})
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- T4
def t_noise_study():
    def L(n):
        return json.load(open(RES / f"noise_study/{n}.json"))
    P = [("time_progress_spearman", ["trajectory", "time_progress_spearman"]), ("trajectory_straightness", ["trajectory", "trajectory_straightness"]),
         ("mean_velocity_cosine_raw", ["trajectory", "mean_velocity_cosine_raw"]), ("direction_reversal_fraction", ["trajectory", "direction_reversal_fraction"]),
         ("adjacent_temporal_nn", ["trajectory", "adjacent_temporal_nn_fraction"]), ("distance_to_goal", ["trajectory", "distance_to_goal_spearman"]),
         ("planning R² (interp)", ["planning", "interp", "r2"]), ("planning R² (static midpoint)", ["planning", "mid_const", "r2"]),
         ("effective_rank_fraction", ["dataset", "effective_rank_fraction"]), ("linear probe c+c", ["probes", "content+change", "linear_seed0"]),
         ("kNN probe c+c", ["probes", "content+change", "knn_mean"]), ("chirality acc.", ["probes", "chirality", "repo_ridge", "mean_accuracy"]),
         ("change-direction gap", ["probes", "consistency", "change_gap"]), ("TCC τ", ["probes", "consistency", "tcc_tau_mean"])]
    li = json.load(open(RES / "reparam_audit_dinov2.json"))
    groups = [("untrained", "untrained_s{}"), ("no reg", "noreg_s{}"), ("VICReg sw0.1", "vicreg_s{}"), ("VICReg sw1", "vicreg_sw1_s{}"), ("SigReg", "sigreg_s{}")]
    raw = L("RAW_dinov2cls")
    rows = []
    for lab, p in P:
        r = {"metric": lab, "raw DINOv2": get(raw, p), "PCA-whitened raw": get(li["pca_whiten"], p)}
        for gl, pat in groups:
            v = [get(L(pat.format(s)), p) for s in range(1, 6) if glob.glob(str(RES / f"noise_study/{pat.format(s)}.json"))]
            r[gl] = f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}"
        r["raw DINOv2"] = f"{r['raw DINOv2']:.4f}"
        r["PCA-whitened raw"] = f"{r['PCA-whitened raw']:.4f}"
        rows.append(r)
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- T5
def t_probes():
    rows = []
    for f in sorted(glob.glob(str(RES / "probes/*.json"))):
        r = json.load(open(f))
        cc, ch, co = r["content+change"], r["chirality"], r["consistency"]
        rows.append({"encoder": r["encoder"], "chance": r["chance"], "perm-null": f"{cc['perm_null_mean']:.3f}±{cc['perm_null_sd']:.3f}",
                     "linear c+c": cc["linear_seed0"], "val-boot 95% CI": f"[{cc['linear_boot_lo']:.3f}, {cc['linear_boot_hi']:.3f}]",
                     "sd over 5 train subsamples": cc["linear_sd"], "ceiling (all train)": cc["ceiling_full_train_linear"],
                     "chirality": ch["repo_ridge"]["mean_accuracy"], "chir. CI": f"[{ch['repo_ridge']['ci_lo']:.3f}, {ch['repo_ridge']['ci_hi']:.3f}]",
                     "chir. order-blind": ch["order_blind"]["mean_accuracy"], "chir. shuffled-label": ch["label_shuffle_null_mean"],
                     "TCC τ": co["tcc_tau_mean"], "τ null": co["tcc_tau_null_mean"]})
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- T6
def t_skill():
    raw = json.load(open(RES / "skill/RAW.json"))
    rows = [{"system": "copy last frame", "skill": 0.0, "note": "by definition"},
            {"system": "EMA(0.5) of past frames (no learning)", "skill": raw["ema0.5"]["skill"], "note": "free denoiser"},
            {"system": "ridge AR(1), raw latents", "skill": raw["ridge_ar1"]["skill"], "note": "learned, linear"},
            {"system": "ridge AR(3), raw latents", "skill": raw["ridge_ar3"]["skill"], "note": "learned, linear"},
            {"system": "noise ceiling, raw latents", "skill": raw["noise_ceiling"]["skill"], "note": "upper bound on any predictor"}]
    for cfg, lab in (("noreg", "no reg"), ("vicreg", "VICReg sw0.1"), ("sigreg", "SigReg"), ("vicreg_sw1", "VICReg sw1")):
        js = [json.load(open(f)) for f in sorted(glob.glob(str(RES / f"skill/{cfg}_s[0-9].json")))]
        m = np.array([j["model"]["skill"] for j in js])
        a = np.array([j["ridge_ar3"]["skill"] for j in js])
        n = np.array([j["noise_ceiling"]["skill"] for j in js])
        rows.append({"system": f"trained predictor, {lab} (n={len(js)})", "skill": m.mean(),
                     "note": f"sd {m.std(ddof=1):.4f}; ridge AR(3) in its own latents {a.mean():.3f}; noise ceiling {n.mean():.3f}"})
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- T7
def t_synth():
    p = RES / "synth_metrics.csv"
    if not p.exists():
        return "_(synthetic results not available)_"
    d = pd.read_csv(p)
    conds = ["static_noise", "iid_scenes", "linear_clean", "linear_both", "accel_clean", "fill_level", "step_change"]
    encs = [e for e in ["dinov2", "dinov2_base", "dinov2_large", "dinov2_giant", "dinov3", "clip", "siglip", "levljepa", "vjepa2.1", "vjepa2", "pixel"] if e in set(d.encoder)]
    out = []
    for m, lab in [("time_progress_spearman", "time_progress_spearman"), ("trajectory_straightness", "trajectory_straightness"),
                   ("mean_velocity_cosine_raw", "mean_velocity_cosine_raw"), ("direction_reversal_fraction", "direction_reversal_fraction")]:
        piv = d.pivot(index="encoder", columns="cond", values=m).reindex(encs)[conds].reset_index()
        out.append(f"**{lab}**\n\n" + md(piv))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------------------------- T8
def t_audit():
    R = HERE.parent / "progress_metrics/results/"
    files = {"sw0.1": "ssv2_progress_pwm_patch_channel_k400_10pct_vicreg_window0_batch16_1_epoch49.json",
             "sw0.5": "ssv2_progress_pwm_patch_channel_k400_10pct_vicreg_sw0.5_window0_batch16.json",
             "sw1": "ssv2_progress_pwm_patch_channel_k400_10pct_vicreg_sw1_window0_batch16.json",
             "sw1.5": "ssv2_progress_pwm_patch_channel_k400_10pct_vicreg_sw1.5_window0_batch16.json",
             "sw2": "ssv2_progress_pwm_patch_channel_k400_10pct_vicreg_sw2_window0_batch16.json"}
    A = {}
    for k, f in files.items():
        pc = json.load(open(R / f))["per_clip"]
        A[k] = {m: np.array([c[m] for c in pc]) for m in pc[0] if m not in ("triplet_id", "verb")}
    rows = []
    for a, b in [("sw0.5", "sw0.1"), ("sw1", "sw0.5"), ("sw1.5", "sw1"), ("sw2", "sw1.5")]:
        for m in ["time_progress_spearman", "trajectory_straightness", "mean_velocity_cosine", "direction_reversal_fraction"]:
            d = noise.paired_diff_bootstrap(A[a][m], A[b][m], B=2000)
            rows.append({"contrast": f"{a} − {b}", "metric": m, "Δ": d["diff"], "95% CI (paired, clips)": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                         "excludes 0": "yes" if d["lo"] > 0 or d["hi"] < 0 else "**no**"})
    return md(pd.DataFrame(rows), 4)


def t_amp():
    p = RES / "synth_amp_metrics.csv"
    if not p.exists():
        return "_(not available)_"
    d = pd.read_csv(p)
    out = []
    for m in ["time_progress_spearman", "trajectory_straightness", "mean_velocity_cosine_raw", "direction_reversal_fraction"]:
        piv = d.pivot(index="encoder", columns="radius_px", values=m).reset_index()
        piv.columns = [str(c) if c == "encoder" else f"r={c}px" for c in piv.columns]
        out.append(f"**{m}**\n\n" + md(piv))
    return "\n\n".join(out)


def t_reliability():
    d = pd.read_csv(RES / "jitter_reliability.csv")
    piv = d.pivot(index="metric", columns="encoder", values="icc")
    piv["clip-level use"] = np.where(piv.min(axis=1) >= 0.85, "reliable", np.where(piv.min(axis=1) >= 0.7, "moderate", "unreliable"))
    piv = piv.sort_values("dinov2", ascending=False).reset_index()
    return md(piv)


TABLES = {"card": t_card, "encoders": t_encoders, "planning": t_planning, "noise_study": t_noise_study, "probes": t_probes,
          "skill": t_skill, "synth": t_synth, "audit": t_audit, "amp": t_amp, "reliability": t_reliability}

if __name__ == "__main__":
    out = []
    for k, f in TABLES.items():
        out.append(f"## {k}\n\n{f()}\n")
    (RES / "tables.md").write_text("\n".join(out))
    print("\n".join(out))
