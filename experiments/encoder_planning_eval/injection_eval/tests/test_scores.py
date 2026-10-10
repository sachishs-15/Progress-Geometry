"""Scores respond to the geometry they are meant to measure; AUROC is exact."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scores import auroc, compare, geometry_scores, localization, paired_win, predictor_scores  # noqa: E402


def line(n=16, d=6, speed=1.0, batch=3):
    t = np.arange(n)[None, :, None] * speed
    u = np.ones((1, 1, d)) / np.sqrt(d)
    return np.repeat(t * u, batch, axis=0)


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 50).astype(float)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert abs(auroc(a, b) - brute) < 1e-12
    assert auroc([2, 3], [0, 1]) == 1.0 and auroc([0, 1], [2, 3]) == 0.0 and auroc([1, 1], [1, 1]) == 0.5


def test_straight_constant_pace_path_scores_zero_everywhere():
    s, _ = geometry_scores(line())
    for k in ("cv_max", "cv_mean", "chord_dev", "backslide", "step_cv"):
        assert np.allclose(s[k], 0, atol=1e-6), k
    assert np.allclose(s["jump_spike"], 1, atol=1e-6) and np.allclose(s["stall"], 0, atol=1e-6)


def test_jump_detour_backslide_and_stall_are_detected():
    base = line(batch=1)
    jump = base.copy(); jump[:, 8:] += 6.0
    assert geometry_scores(jump)[0]["jump_spike"][0] > 5
    detour = base.copy(); detour[:, 8, 0] += 5.0; detour[:, 8, 1] -= 5.0
    assert geometry_scores(detour)[0]["chord_dev"][0] > 3
    back = base.copy(); back[:, 10:] -= 6.0
    assert geometry_scores(back)[0]["backslide"][0] > 0.2
    stall = base.copy(); stall[:, 5:9] = stall[:, 5:6]
    assert geometry_scores(stall)[0]["stall"][0] > 0.9
    pace = np.concatenate([line(8, batch=1), line(8, batch=1) * 3 + 7], axis=1)
    assert geometry_scores(pace)[0]["step_cv"][0] > 0.3


def test_predictor_scores():
    pe = np.ones((2, 15)); pe[1, 7] = 10
    s = predictor_scores(pe)
    assert s["pe_max"].tolist() == [1, 10]
    assert s["pe_spike"][1] > 9 and s["pe_spike"][0] < 1.1


def test_compare_drops_non_finite_and_localization_counts_hits():
    r = compare([1, 2, 3, np.nan] * 5, [0, 1, 2, 3] * 5, n_boot=20)
    assert r["n"] == 15 and r["ci_lo"] <= r["auroc"] <= r["ci_hi"]
    arr = np.zeros((4, 15)); arr[0, 5] = 1; arr[1, 9] = 1; arr[2, 5] = 1; arr[3, 5] = 1
    loc = localization(arr, [[5], [5], [6], None])
    assert loc["n"] == 3 and abs(loc["hit"] - 2 / 3) < 1e-9 and abs(loc["chance"] - 3 / 15) < 0.07
    assert paired_win([2, 2], [1, 2]) == 0.75


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
