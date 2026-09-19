"""Experiment views computed from the export (no hand-transcribed numbers).

Everything here is recomputed from the frozen detector's real scores on W1
validation, W1 test and W2. Curves are exact: with so few positives, the
precision-recall and ROC step curves are fully described by their corners, one
per positive, so no downsampling can distort them.
"""

from __future__ import annotations

import numpy as np

from backend.services.store import DashboardStore
from ml.baselines.xgboost_baseline import evaluate_at_threshold


def step_curves(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Exact PR and ROC corners, one point per positive in score order."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    negatives = int(ranked.size - positives)
    if positives == 0 or negatives == 0:
        return {"pr": [], "roc": []}
    ranks = np.flatnonzero(ranked)
    tp = np.arange(1, positives + 1)
    predicted = ranks + 1
    fp = predicted - tp
    thresholds = scores[order][ranks]
    pr = [{"recall": 0.0, "precision": 1.0, "threshold": None}] + [
        {"recall": float(r), "precision": float(p), "threshold": float(t)}
        for r, p, t in zip(tp / positives, tp / predicted, thresholds)
    ]
    roc = [{"fpr": 0.0, "tpr": 0.0}] + [
        {"fpr": float(f), "tpr": float(t)} for f, t in zip(fp / negatives, tp / positives)
    ] + [{"fpr": 1.0, "tpr": 1.0}]
    return {"pr": pr, "roc": roc}


def threshold_sweep(labels: np.ndarray, scores: np.ndarray, hours: float, selected: float) -> list[dict]:
    """Precision, recall, F1 and alert load at a fixed grid of thresholds."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    ascending = np.sort(scores)
    order = np.argsort(-scores, kind="stable")
    cumulative_tp = np.cumsum(labels[order])
    positives = int(labels.sum())
    grid = np.unique(np.concatenate([np.geomspace(1e-4, 0.999, 48), [selected]]))
    rows = []
    for threshold in grid:
        predicted = int(scores.size - np.searchsorted(ascending, threshold, side="left"))
        tp = int(cumulative_tp[predicted - 1]) if predicted else 0
        fp = predicted - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive": predicted,
                "tp": tp,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_per_hour": fp / hours if hours else None,
                "selected": bool(threshold == selected),
            }
        )
    return rows


def computed_results(store: DashboardStore) -> dict:
    manifest = store.manifest
    boundaries = manifest["windows"]["w1"]["boundaries"]
    w1_end = float(manifest["windows"]["w1"]["emit_end"])
    w2 = manifest["windows"]["w2"]
    splits = {
        "w1_validation": (
            store.w1["validation_labels"],
            store.w1["validation_scores"],
            (boundaries["validation_end"] - boundaries["train_end"]) / 3600.0,
            "W1 validation (threshold selection)",
        ),
        "w1_test": (
            store.w1["test_labels"],
            store.w1["test_scores"],
            (w1_end - boundaries["validation_end"]) / 3600.0,
            "W1 test (in-window)",
        ),
        "w2": (
            store.labels,
            store.scores,
            (float(w2["emit_end"]) - float(w2["emit_start"])) / 3600.0,
            "W2 full window (frozen, cross-window)",
        ),
    }
    results = {}
    for key, (labels, scores, hours, title) in splits.items():
        metrics = evaluate_at_threshold(labels, scores, store.threshold).as_dict()
        metrics["fp_per_hour"] = metrics["fp"] / hours if hours else None
        results[key] = {
            "title": title,
            "hours": hours,
            "rows": int(np.asarray(labels).size),
            "positives": int(np.asarray(labels).sum()),
            "metrics": metrics,
            "curves": step_curves(labels, scores),
            "sweep": threshold_sweep(labels, scores, hours, store.threshold),
        }
    return {
        "kind": "computed",
        "source": "recomputed from data/processed/dashboard export (frozen M4.0 XGBoost)",
        "model": manifest["model"]["name"],
        "threshold": store.threshold,
        "splits": results,
    }
