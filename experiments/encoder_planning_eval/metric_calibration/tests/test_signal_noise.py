"""Noise estimators recover the known noise level of synthetic trajectories."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import signal_noise as sn  # noqa: E402


def make(n=800, t=16, d=128, sigma=0.5, seed=0, curved=True):
    rng = np.random.default_rng(seed)
    tt = np.linspace(0, 1, t)[None, :, None]
    a = rng.standard_normal((n, 1, d))
    b = rng.standard_normal((n, 1, d)) * (1.0 if curved else 0.0)
    sig = a * tt + b * tt ** 2  # smooth, curved signal
    return sig + sigma * rng.standard_normal((n, t, d)), sigma ** 2 * d


def test_recovers_noise():
    for sigma in (0.1, 0.5, 2.0):
        z, true = make(sigma=sigma)
        e_loo = sn.poly_loo_error(z, 2).mean()  # quadratic signal -> deg 2 correct
        e_res = sn.poly_resid_var(z, 2).mean()
        e_d4 = sn.diff_noise_var(z, 4).mean()
        assert abs(e_res / true - 1) < 0.03, (sigma, e_res, true)
        assert abs(e_d4 / true - 1) < 0.08, (sigma, e_d4, true)  # high order removes the signal
        assert e_loo > 0.9 * true and e_loo < 1.4 * true  # LOO error ~ noise (+ prediction variance)
    z, true = make(sigma=0.5)
    assert sn.diff_noise_var(z, 1).mean() > sn.diff_noise_var(z, 4).mean() * 0.99  # signal inflates low orders
    assert sn.best_degree(z)[0] in (1, 2)  # at this SNR the quadratic term is below the noise; LOO may prefer a line
    z, true = make(sigma=0.05)
    assert sn.best_degree(z)[0] == 2
    print("ok")


if __name__ == "__main__":
    test_recovers_noise()
