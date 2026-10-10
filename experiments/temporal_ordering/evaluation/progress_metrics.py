"""Section 5 / 17 regression metrics for predicted vs. true procedural progress."""
import numpy as np
from scipy.stats import spearmanr


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2 or np.all(pred == pred[0]):
        return float("nan")
    rho, _ = spearmanr(pred, true)
    return float(rho)


def all_progress_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred, true = np.asarray(pred), np.asarray(true)
    return {
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "r2": r2_score(pred, true),
        "spearman": spearman(pred, true),
    }
