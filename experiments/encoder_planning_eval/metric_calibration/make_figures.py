"""Figures for REPORT.md -> results/figures/*.png.  Light surface; blue = measured, orange = free baselines
(smoothing / whitening), aqua = ceilings/oracles, grey = floors/nulls. Everything is direct-labelled."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True, parents=True)
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c88", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#ececE8", "grid.linewidth": 0.8, "axes.axisbelow": True, "text.color": INK})


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------------------------- fig 1
def fig_ladder():
    tc = pd.read_csv(RES / "trajectory_calibration.csv")
    g = lambda ref, m, e="dinov2": float(tc[(tc.encoder == e) & (tc.ref == ref) & (tc.metric == m)].value.iloc[0])
    boot = tc[(tc.encoder == "dinov2") & (tc.ref == "REAL") & tc.metric.str.endswith("__boot")].set_index("metric")
    rows = [("shuffle_time", "frames shuffled in time (floor)", GREY), ("REAL", "DINOv2-S, real SSv2 clips", BLUE),
            ("brownian_signflip", "random walk, same step vectors", ORANGE), ("gauss1", "free smoothing, σ = 1 frame", ORANGE),
            ("cummean", "causal cumulative mean", ORANGE), ("line_endpoints", "straight line between endpoints", AQUA)]
    mets = [("time_progress_spearman", "time_progress\nSpearman"), ("trajectory_straightness", "trajectory\nstraightness"),
            ("mean_velocity_cosine_raw", "mean velocity\ncosine (raw)"), ("direction_reversal_fraction", "direction reversal\nfraction (lower = better)"),
            ("adjacent_temporal_nn_fraction", "adjacent temporal\nNN fraction"), ("distance_to_goal_spearman", "distance-to-goal\nSpearman")]
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.6), sharey=True)
    for ax, (m, title) in zip(axes, mets):
        for i, (ref, lab, col) in enumerate(rows):
            v = g(ref, m)
            ax.plot([v], [i], "o", color=col, ms=8, mec="#fcfcfb", mew=1.5, zorder=3)
            ax.annotate(f"{v:.2f}", (v, i), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
        lo, hi = boot.loc[m + "__boot", ["ci_lo", "ci_hi"]]
        ax.plot([lo, hi], [1, 1], "-", color=BLUE, lw=3, alpha=.5, zorder=2)
        ax.axhline(1, color="#ececE8", lw=0)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(len(rows) - .5, -.7)
        ax.margins(x=0.12)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows], fontsize=9)
    fig.suptitle("Where a real encoder sits between a floor and a ceiling (DINOv2-S CLS, SSv2 val, mean over 1430 clips)",
                 y=1.04, fontsize=11.5)
    save(fig, "fig1_reference_ladder.png")


# ---------------------------------------------------------------------------------------------- fig 2
def fig_planning():
    import calibrate_planning as cp
    d = pd.read_csv(RES / "planning_calibration.csv")
    d = d[~d.predictor.str.startswith("PAIR")]
    ladder = [("mean", "predict dataset mean"), ("copy_z1", "copy start frame"), ("mid_const", "static midpoint (z₁+z_T)/2"),
              ("interp", "linear interpolation  (reported metric)"), ("ridge_resid", "learned ridge ceiling"),
              ("neighbors", "oracle: adjacent frames"), ("noise_ceiling", "noise ceiling")]
    cols = {"mean": GREY, "copy_z1": GREY, "mid_const": ORANGE, "interp": BLUE, "ridge_resid": AQUA, "neighbors": AQUA,
            "noise_ceiling": AQUA}
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    e = d[d.encoder == "dinov2"].set_index("predictor")
    for i, (k, lab) in enumerate(ladder):
        v = float(e.loc[k, "r2"])
        ax.barh(i, max(v, 0), color=cols[k], height=.55)
        ax.annotate(f"{v:.3f}", (max(v, 0), i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(ladder)))
    ax.set_yticklabels([l for _, l in ladder])
    ax.set_ylim(len(ladder) - .4, -.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R² on the 14 intermediate frames")
    ax.set_title("What R² = 0.66 is made of (DINOv2-S)", fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    # per-position gain
    ax = fig.add_subplot(gs[1])
    zva, _ = cp.load("dinov2", "validation")
    n, t, dd = zva.shape
    true = zva[:, 1:-1]
    mu = true.reshape(-1, dd).mean(0)
    ip = cp.interp(zva[:, 0], zva[:, -1], t)[:, 1:-1]
    mid = np.broadcast_to(((zva[:, 0] + zva[:, -1]) / 2)[:, None], true.shape)
    tot = ((true - mu) ** 2).sum(-1)
    r2 = lambda p: np.array([1 - ((p[:, k] - true[:, k]) ** 2).sum() / tot[:, k].sum() for k in range(t - 2)])
    x = np.arange(1, t - 1)
    ax.plot(x, r2(mid), color=ORANGE, lw=2)
    ax.plot(x, r2(ip), color=BLUE, lw=2)
    ax.fill_between(x, r2(mid), r2(ip), color=BLUE, alpha=.12)
    ax.text(1.0, .93, "linear interpolation", color=BLUE, fontsize=9.5)
    ax.text(9.3, .455, "static midpoint", color=ORANGE, fontsize=9.5)
    ax.annotate("no gain over a constant\nin the middle of the clip", (7.7, .525), xytext=(4.6, .70), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.8))
    ax.set_xlabel("intermediate frame index t")
    ax.set_ylabel("R²")
    ax.set_ylim(.4, 1)
    ax.set_title("Interpolation only helps next to the endpoints", fontsize=10.5)
    # across encoders: static vs temporal part
    ax = fig.add_subplot(gs[2])
    piv = d.pivot(index="encoder", columns="predictor", values="r2")
    order = piv["interp"].sort_values().index
    for i, enc in enumerate(order):
        s, tpart = piv.loc[enc, "mid_const"], piv.loc[enc, "interp"] - piv.loc[enc, "mid_const"]
        ax.barh(i, s, color=ORANGE, height=.6)
        ax.barh(i, tpart, left=s, color=BLUE, height=.6)
        ax.annotate(f"+{tpart:.3f}", (s + tpart, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=BLUE)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("interpolation R²")
    ax.set_title("Static clip identity vs temporal part, by encoder", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="static midpoint (no time information)"),
                       Patch(color=BLUE, label="added by interpolation")], loc="upper center", bbox_to_anchor=(0.45, -0.17), frameon=False, fontsize=8.5, ncol=1)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig2_planning_ladder.png")


# ---------------------------------------------------------------------------------------------- fig 3
def fig_response():
    c = pd.read_csv(RES / "synthetic_latent_curves.csv")
    c = c[c.theta == 0].sort_values("r")
    tc = pd.read_csv(RES / "trajectory_calibration.csv")
    mets = [("time_progress_spearman", "time_progress Spearman"), ("distance_to_goal_spearman", "distance-to-goal Spearman"),
            ("adjacent_temporal_nn_fraction", "adjacent temporal NN"), ("trajectory_straightness_raw", "trajectory straightness"),
            ("mean_velocity_cosine_raw", "mean velocity cosine"), ("direction_reversal_fraction", "direction reversal fraction")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    for ax, (m, title) in zip(axes.ravel(), mets):
        ax.plot(c.r, c[m], color=BLUE, lw=2)
        for enc, col in (("dinov2", INK2), ("vjepa2.1", ORANGE)):
            v = float(tc[(tc.encoder == enc) & (tc.ref == "REAL") & (tc.metric == m)].value.iloc[0])
            ax.axhline(v, color=col, lw=1, ls="--")
            up = enc == "dinov2" if v < 0.995 and abs(v - float(tc[(tc.encoder == "dinov2") & (tc.ref == "REAL") & (tc.metric == m)].value.iloc[0])) < 1 else True
            hi = float(tc[(tc.encoder == "dinov2") & (tc.ref == "REAL") & (tc.metric == m)].value.iloc[0]) >= float(
                tc[(tc.encoder == "vjepa2.1") & (tc.ref == "REAL") & (tc.metric == m)].value.iloc[0])
            above = (enc == "dinov2") == hi  # the higher line's label goes above it, the lower one below
            ax.annotate(("DINOv2-S " if enc == "dinov2" else "V-JEPA2.1 ") + f"{v:.2f}", (c.r.max() * 0.8, v),
                        xytext=(0, 4 if above else -11), textcoords="offset points", fontsize=8.5, color=col, ha="right")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10.5)
    for ax in axes[1]:
        ax.set_xlabel("drift-to-noise ratio  r = ‖step drift‖² / (D σ²)")
    fig.suptitle("Metric response to signal-to-noise on straight-line + white-noise trajectories (D=384, T=16).  Dashed lines = real SSv2 values",
                 y=1.01, fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig3_snr_response.png")


# ---------------------------------------------------------------------------------------------- fig 4
def fig_noise():
    card = pd.read_csv(RES / "metric_card_trajectory_dinov2.csv").set_index("metric")
    mets = ["time_progress_spearman", "distance_to_goal_spearman", "adjacent_temporal_nn_fraction", "trajectory_straightness",
            "mean_velocity_cosine_raw", "direction_reversal_fraction"]
    names = ["time_progress", "distance_to_goal", "adjacent_NN", "straightness", "velocity_cosine", "direction_reversal"]
    rng = (card.ceiling - card.floor).abs()
    series = [("se_clip", "clip-sampling SE (single model)", BLUE, "o"), ("se_pair", "paired clip SE (model − raw)", BLUE, "D"),
              ("sd_seed", "training-seed sd (5 seeds)", AQUA, "s"), ("sd_epoch", "epoch 29→49 drift", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, m in enumerate(mets):
        for j, (k, lab, col, mk) in enumerate(series):
            v = card.loc[m, k] / rng[m] * 100
            ax.plot([v], [i], mk, color=col, ms=8, mec="#fcfcfb", mew=1, label=lab if i == 0 else None, zorder=3)
        # free baseline effects
        w = abs(card.loc[m, "d_whiten_mdd"] * card.loc[m, "mdd"]) / rng[m] * 100
        s = abs(card.loc[m, "d_gauss1_range"]) * 100
        ax.plot([w], [i], "P", color=ORANGE, ms=9, mec="#fcfcfb", mew=1, label="effect of free PCA-whitening" if i == 0 else None, zorder=3)
        ax.plot([s], [i], "X", color=ORANGE, ms=9, mec="#fcfcfb", mew=1, label="effect of σ=1 smoothing" if i == 0 else None, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(len(mets)))
    ax.set_yticklabels(names)
    ax.set_ylim(len(mets) - .4, -.6)
    ax.set_xlabel("% of the metric's formal range (shuffle floor → straight-line ceiling), log scale")
    ax.set_title("Every noise source is small next to the effect of a free baseline", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False, ncol=1)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig4_noise_vs_free_effects.png")


# ---------------------------------------------------------------------------------------------- fig 5
def fig_tradeoff():
    from matplotlib.lines import Line2D

    def L(n):
        return json.load(open(RES / f"noise_study/{n}.json"))
    sm, li = json.load(open(RES / "smooth_audit_dinov2.json")), json.load(open(RES / "reparam_audit_dinov2.json"))
    P = {"x1": ("trajectory", "mean_velocity_cosine_raw"), "x2": ("trajectory", "time_progress_spearman"),
         "y1": ("probes", "chirality", "repo_ridge", "mean_accuracy"), "y2": ("probes", "content+change", "linear_seed0")}

    def get(j, p):
        for k in p:
            j = j[k]
        return j
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (xk, yk, xl, yl) in zip(axes, [("x1", "y1", "mean velocity cosine (raw)  →  straighter", "chirality accuracy  →  keeps arrow-of-time"),
                                           ("x2", "y2", "time_progress Spearman  →  more ordered", "content+change linear probe  →  keeps semantics")]):
        smooth = [("0.5", sm["gauss0.5"]), ("1", sm["gauss1"]), ("2", sm["gauss2"]), ("cum", sm["cummean"])]
        xs = [get(li["identity"], P[xk])] + [get(d, P[xk]) for _, d in smooth]
        ys = [get(li["identity"], P[yk])] + [get(d, P[yk]) for _, d in smooth]
        ax.plot(xs, ys, "-", color=ORANGE, lw=1.5, alpha=.8, zorder=1)
        for (lab, _), x, y in zip(smooth, xs[1:], ys[1:]):
            ax.plot([x], [y], "o", color=ORANGE, ms=8, mec="#fcfcfb", mew=1.2, zorder=3)
            ax.annotate(lab, (x, y), xytext=(0, 9), textcoords="offset points", fontsize=8.5, color=ORANGE, ha="center")
        ax.plot([xs[0]], [ys[0]], "o", color=GREY, ms=11, mec="#fcfcfb", mew=1.2, zorder=3)
        wh = li["pca_whiten"]
        ax.plot([get(wh, P[xk])], [get(wh, P[yk])], "P", color=ORANGE, ms=11, mec="#fcfcfb", mew=1.2, zorder=3)
        for cfg, mk in (("vicreg", "o"), ("vicreg_sw1", "s"), ("noreg", "^")):
            v = [L(f"{cfg}_s{s}") for s in range(1, 6)]
            ax.scatter([get(j, P[xk]) for j in v], [get(j, P[yk]) for j in v], marker=mk, color=BLUE, s=40, zorder=4,
                       edgecolor="#fcfcfb", linewidth=.8)
        un = [L(f"untrained_s{s}") for s in range(1, 6)]
        ax.scatter([get(j, P[xk]) for j in un], [get(j, P[yk]) for j in un], color=AQUA, s=40, zorder=4, edgecolor="#fcfcfb", linewidth=.8)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    h = [Line2D([], [], marker="o", color=GREY, ls="", ms=9, label="raw DINOv2 CLS"),
         Line2D([], [], marker="o", color=AQUA, ls="", ms=7, label="untrained copy of the architecture (5 inits)"),
         Line2D([], [], marker="o", color=BLUE, ls="", ms=7, label="trained, VICReg sw0.1 (5 seeds)"),
         Line2D([], [], marker="s", color=BLUE, ls="", ms=7, label="trained, VICReg sw1 (5 seeds)"),
         Line2D([], [], marker="^", color=BLUE, ls="", ms=7, label="trained, no regulariser (5 seeds)"),
         Line2D([], [], marker="P", color=ORANGE, ls="", ms=9, label="raw, free PCA-whitening"),
         Line2D([], [], marker="o", color=ORANGE, ls="-", ms=7, label="raw, Gaussian smoothing σ = 0.5, 1, 2 frames → cumulative mean")]
    axes[0].legend(handles=h, loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle("Trained models vs free baselines: smoothing buys 'progress' by giving up arrow-of-time / semantics; whitening explains VICReg",
                 y=1.0, fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig5_tradeoff_frontier.png")


# ---------------------------------------------------------------------------------------------- fig 6
def fig_skill():
    raw = json.load(open(RES / "skill/RAW.json"))
    import glob
    fig, ax = plt.subplots(figsize=(10, 4.2))
    items = [("copy last frame", raw["copy"]["skill"], GREY), ("EMA(0.5) of past frames", raw["ema0.5"]["skill"], ORANGE),
             ("ridge AR(3) on the same latents", raw["ridge_ar3"]["skill"], AQUA), ("noise ceiling", raw["noise_ceiling"]["skill"], AQUA)]
    for i, (lab, v, col) in enumerate(items):
        ax.barh(i, v, color=col, height=.55)
        ax.annotate(f"{v:+.3f}", (max(v, 0), i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    base = len(items)
    for j, (cfg, lab) in enumerate((("vicreg", "trained predictor, VICReg"), ("sigreg", "trained predictor, SigReg"),
                                     ("vicreg_sw1", "trained predictor, VICReg sw1"), ("noreg", "trained predictor, no reg"))):
        vs = [json.load(open(f))["model"]["skill"] for f in sorted(glob.glob(str(RES / f"skill/{cfg}_s[0-9].json")))]
        i = base + j
        ax.barh(i, np.mean(vs), color=BLUE, height=.55)
        ax.plot(vs, [i] * len(vs), "o", color="#fcfcfb", ms=5, mec="#fcfcfb", zorder=4)
        ax.annotate(f"{np.mean(vs):+.3f}", (max(max(vs), np.mean(vs)), i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
        items.append((lab, np.mean(vs), BLUE))
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items])
    ax.set_ylim(len(items) - .4, -.6)
    ax.set_xlim(0, .68)
    ax.set_xlabel("one-step skill  =  1 − error / copy-error   (SSv2 val, CLS latents, positions k ≥ 3)")
    ax.set_title("Skill 1.0 is unreachable (noise ceiling ≈ 0.6); a 3-tap linear model beats the trained predictors", fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig6_skill_ladder.png")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    which = sys.argv[1:] or ["ladder", "planning", "response", "noise", "tradeoff", "skill"]
    for w in which:
        {"ladder": fig_ladder, "planning": fig_planning, "response": fig_response, "noise": fig_noise,
         "tradeoff": fig_tradeoff, "skill": fig_skill}[w]()
