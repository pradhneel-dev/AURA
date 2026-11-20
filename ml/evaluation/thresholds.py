"""Training-only threshold selection and threshold-agnostic ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    total = max(len(y), 1)
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "specificity": specificity, "false_positive_rate": fp/(fp+tn) if fp+tn else 0.0,
            "false_negative_rate": fn/(fn+tp) if fn+tp else 0.0,
            "flagged_rate": (tp+fp)/total, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def select_operational_threshold(y_true, oof_probabilities, capacity_rate: float = 0.20,
                                 thresholds=None) -> tuple[float, pd.DataFrame, str]:
    """Select best F1 among thresholds that fit declared outreach capacity."""
    if not 0 < float(capacity_rate) <= 1: raise ValueError("capacity_rate must be in (0, 1]")
    if thresholds is None: thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = [threshold_metrics(y_true, oof_probabilities, threshold) for threshold in thresholds]
    table = pd.DataFrame(rows)
    eligible = table.loc[table.flagged_rate <= capacity_rate]
    if eligible.empty:
        raise ValueError("No threshold satisfies the configured capacity; increase capacity_rate")
    chosen = eligible.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    rationale = (f"Selected threshold {chosen.threshold:.2f} from training-only out-of-fold predictions: "
                 f"highest F1 among thresholds with flagged rate <= configured {capacity_rate:.1%} outreach capacity; "
                 "ties prefer higher precision, then a higher threshold.")
    return float(chosen.threshold), table, rationale


def ranking_metrics(y_true, probabilities, capacity_rates=(0.10, 0.20, 0.30)) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    base_rate = float(y.mean()) if len(y) else 0.0
    rows = []
    for rate in capacity_rates:
        k = max(1, int(np.ceil(len(y) * rate)))
        chosen = np.argsort(-p)[:k]
        hits = int(y[chosen].sum())
        precision = hits/k
        recall = hits/max(int(y.sum()), 1)
        rows.append({"capacity_rate": float(rate), "k_customers": int(k), "precision_at_k": precision,
                     "recall_at_k": recall, "lift_at_k": precision/base_rate if base_rate else 0.0,
                     "note": "Top-K ranking at stated capacity; not a treatment effect."})
    return rows
