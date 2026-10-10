"""Regression checks for prediction-skill error semantics."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_prediction_skill import position_error, summarize  # noqa: E402


def test_angular_error_matches_training_loss_and_ignores_scale():
    target = torch.randn(2, 4, 3, 8)
    pred = 7 * target
    error = position_error(pred, target, "angular")
    assert error.shape == (2, 4)
    assert error.abs().max().item() < 1e-5
    assert position_error(pred, target, "l2").mean().item() > 1.0


def test_unknown_error_is_rejected():
    x = torch.randn(1, 2, 4)
    try:
        position_error(x, x, "angluar")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown loss names must not silently fall through to L2")


def test_exact_collapse_reports_undefined_skill_without_nan():
    result = summarize(np.zeros(3), np.zeros(3))
    assert result["skill"] is None
    assert result["skill_per_clip_median"] is None
    assert result["degenerate_copy_baseline"] is True
    assert result["n_skill_clips"] == 0
    assert result["n_degenerate_clips"] == 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
