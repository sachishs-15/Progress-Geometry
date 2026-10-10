"""Check held-out source splitting and temporal labels for synthetic edits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_egoper_style_eval import altered_frames, frame_metrics, mark_negative_controls, split_by_source  # noqa: E402
import pandas as pd


def test_split_keeps_source_groups_disjoint_and_test_donors_available():
    labels = [f"class_{i // 5}" for i in range(30)]
    cal, test = split_by_source(labels, 0)
    assert len(cal) == 6 and len(test) == 24
    assert not set(cal) & set(test)
    assert sorted(set(cal) | set(test)) == list(range(30))
    assert all(sum(labels[i] == label for i in test) >= 2 for label in set(labels))


def test_altered_frame_labels_and_frame_scores():
    assert np.flatnonzero(altered_frames("splice_other", [4, 8], 16)).tolist() == [5, 6, 7, 8]
    assert np.flatnonzero(altered_frames("freeze", [8], 16)).tolist() == [6, 7, 8]
    p = np.zeros((1, 15)); p[0, 4:8] = 10
    c = np.zeros((1, 15))
    metrics = frame_metrics(p, c, [[4, 8]], [True])
    assert metrics["n_videos"] == 1 and metrics["positive_frames"] == 4
    assert metrics["frame_auc"] == 1.0


def test_benign_control_is_reported_as_false_alarm():
    df = pd.DataFrame({"injection": ["appearance", "splice_other"], "recall": [0.1, 0.8],
                       "precision": [0.5, 0.9], "f1": [0.2, 0.85],
                       "balanced_accuracy": [0.5, 0.9], "tp": [1, 8], "fn": [9, 2]})
    out = mark_negative_controls(df)
    assert out.control_flag_rate.iloc[0] == 0.1 and np.isnan(out.recall.iloc[0])
    assert out.role.iloc[1] == "mistake" and out.recall.iloc[1] == 0.8


if __name__ == "__main__":
    test_split_keeps_source_groups_disjoint_and_test_donors_available()
    test_altered_frame_labels_and_frame_scores()
    test_benign_control_is_reported_as_false_alarm()
    print("PASS  test_split_keeps_source_groups_disjoint_and_test_donors_available")
    print("PASS  test_altered_frame_labels_and_frame_scores")
    print("PASS  test_benign_control_is_reported_as_false_alarm")
