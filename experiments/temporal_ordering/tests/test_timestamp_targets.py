"""Regression checks for absolute timestamp supervision."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_representation import _timestamp_targets  # noqa: E402


def test_targets_use_source_indices_and_lengths():
    indices = np.array([[3, 6, 11, 14], [0, 2, 5, 9]])
    lengths = np.array([16, 10])
    target = _timestamp_targets(indices, lengths)
    expected = np.array([[3 / 15, 6 / 15, 11 / 15, 14 / 15],
                         [0 / 9, 2 / 9, 5 / 9, 9 / 9]], dtype=np.float32)
    assert np.allclose(target, expected)
    assert not np.allclose(target[0], np.linspace(0, 1, 4)), (
        "targets must not collapse to rank within the sampled subset"
    )


if __name__ == "__main__":
    test_targets_use_source_indices_and_lengths()
    print("PASS  test_targets_use_source_indices_and_lengths")
