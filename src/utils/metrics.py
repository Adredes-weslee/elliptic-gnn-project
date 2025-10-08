
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

def pr_auc_illicit(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # PR-AUC for illicit class (positive=1).
    return float(average_precision_score(y_true, y_score))

def roc_auc_illicit(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))

def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_pred = (y_score >= thr).astype(int)
    return float(f1_score(y_true, y_pred))

def pick_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    i = np.nanargmax(f1s)
    return float(thresholds[i]), float(f1s[i])

def pick_threshold_for_precision(y_true: np.ndarray, y_score: np.ndarray, target_p: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thr_candidates = np.append(thresholds, 1.0)
    mask = precision >= target_p
    if not np.any(mask):
        return pick_threshold_max_f1(y_true, y_score)[0]
    idx = np.argmax(mask)  # first index where condition holds
    return float(thr_candidates[idx])

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[idx]))

def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_p: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    mask = precision >= target_p
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    # ECE for binary probs (positive class).
    y_true = y_true.astype(int)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
