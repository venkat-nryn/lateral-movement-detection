"""Supervised XGBoost baseline on the real-LANL labeled window (M4.0).

This is the first supervised model in the project. It consumes the M3.6 bounded
real-LANL window unchanged:

* features  -- the existing 17 temporal/graph features (M3.4), untouched
* labels    -- the existing exact 4-field redteam match (M3.2), untouched
* splits    -- the existing chronological train/validation/test (M3.5), untouched

Research rules enforced here
----------------------------
* **No resampling of any kind.** No SMOTE, no oversampling, no undersampling,
  no synthetic rows. Class imbalance is addressed only through XGBoost's
  ``scale_pos_weight``, which reweights the loss and never invents data.
* **scale_pos_weight is computed from TRAIN only** -- validation and test label
  counts never enter the model configuration.
* **Thresholds are selected on VALIDATION only** and applied unchanged to TEST.
  Nothing in this module reads test labels before test metrics are computed.
* **No shuffling and no resplitting.** Rows arrive in chronological order and
  are consumed in that order. ``subsample`` and ``colsample_bytree`` are pinned
  to 1.0 and ``random_state`` is fixed, so training is deterministic.
* Redteam information reaches the model only as ``y``. It is never a feature.

Nothing here writes a model file, a dataset dump, or any other artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml.preprocessing.ml_dataset import (
    DatasetSplit,
    MLDataset,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
)

#: Reported unconditionally so results are comparable across runs.
DEFAULT_THRESHOLD = 0.50

MODEL_STANDARD = "standard"
MODEL_CLASS_WEIGHTED = "class_weighted"


@dataclass(frozen=True, slots=True)
class XGBoostBaselineConfig:
    """Deterministic XGBoost configuration.

    ``subsample`` and ``colsample_bytree`` stay at 1.0 so no stochastic sampling
    occurs; combined with a fixed ``random_state`` this makes training
    reproducible run to run.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    tree_method: str = "hist"
    random_state: int = 0
    n_jobs: int = 4

    def to_params(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
        }


def split_matrix(split: DatasetSplit) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for one chronological split.

    ``X`` is built directly from the M3.5 flat ``float64`` buffer, so no
    per-row Python objects are created. Row order is the chronological order in
    which the events were emitted.
    """
    width = split.feature_dimension
    rows = len(split)
    buffer = split.feature_buffer()
    if rows == 0:
        return np.empty((0, width), dtype=np.float64), np.empty(0, dtype=np.int8)
    X = np.frombuffer(buffer, dtype=np.float64).reshape(rows, width)
    y = np.frombuffer(split.labels, dtype=np.int8)
    return X, y


def dataset_matrices(
    dataset: MLDataset,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{split_name: (X, y)}`` for all three chronological splits."""
    return {
        name: split_matrix(dataset.get_split(name))
        for name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
    }


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """``negatives / positives`` computed from TRAIN labels only.

    Returns 1.0 when the training split has no positives, which leaves the
    weighted model identical to the standard one rather than dividing by zero.
    """
    positives = int(np.count_nonzero(y_train))
    negatives = int(y_train.shape[0] - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Threshold-dependent counts plus threshold-free ranking metrics."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    predicted_positive: int
    precision: float
    recall: float
    f1: float
    average_precision: float | None
    roc_auc: float | None
    support_positive: int
    support_negative: int

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "predicted_positive": self.predicted_positive,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "support_positive": self.support_positive,
            "support_negative": self.support_negative,
        }


def _ranking_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float | None, float | None]:
    """PR-AUC and ROC-AUC, or ``None`` when only one class is present.

    A split containing a single class makes both metrics undefined. Reporting
    ``None`` is honest; substituting 0.0 or 0.5 would not be.
    """
    if y_true.size == 0:
        return None, None
    positives = int(np.count_nonzero(y_true))
    if positives == 0 or positives == y_true.size:
        return None, None
    return (
        float(average_precision_score(y_true, scores)),
        float(roc_auc_score(y_true, scores)),
    )


def evaluate_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> BinaryMetrics:
    """Confusion counts and derived rates at a fixed decision threshold."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if y_true.shape != scores.shape:
        raise ValueError(
            f"y_true {y_true.shape} and scores {scores.shape} must align"
        )

    predicted = scores >= threshold
    actual = y_true == 1

    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    average_precision, roc_auc = _ranking_metrics(y_true, scores)

    return BinaryMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        predicted_positive=int(np.count_nonzero(predicted)),
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision,
        roc_auc=roc_auc,
        support_positive=int(np.count_nonzero(actual)),
        support_negative=int(y_true.size - np.count_nonzero(actual)),
    )


def select_threshold_on_validation(
    y_validation: np.ndarray, validation_scores: np.ndarray
) -> float:
    """Pick the F1-maximising threshold using VALIDATION data only.

    Test data is never consulted. Ties break toward the higher threshold, which
    is the more conservative choice (fewer alerts). Falls back to
    :data:`DEFAULT_THRESHOLD` when validation has no positives and the curve is
    therefore undefined.
    """
    y_validation = np.asarray(y_validation).astype(np.int64).ravel()
    validation_scores = np.asarray(validation_scores, dtype=np.float64).ravel()

    positives = int(np.count_nonzero(y_validation))
    if positives == 0 or positives == y_validation.size:
        return DEFAULT_THRESHOLD

    precision, recall, thresholds = precision_recall_curve(
        y_validation, validation_scores
    )
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    if f1.size == 0:
        return DEFAULT_THRESHOLD

    best = float(f1.max())
    # Highest threshold among the maximisers.
    candidates = thresholds[f1 >= best]
    return float(candidates.max())


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: XGBoostBaselineConfig | None = None,
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """Fit a deterministic XGBoost classifier on the training split."""
    config = config or XGBoostBaselineConfig()
    params = config.to_params()
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    model = XGBClassifier(**params)
    model.fit(np.ascontiguousarray(X_train), np.asarray(y_train).ravel())
    return model


def predict_scores(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities; empty input yields an empty array."""
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return model.predict_proba(np.ascontiguousarray(X))[:, 1].astype(np.float64)


def feature_importance_ranking(
    model: XGBClassifier, feature_names: Sequence[str]
) -> list[tuple[str, float]]:
    """All features ranked by gain, including those the model never used.

    ``XGBClassifier.feature_importances_`` omits nothing but reports 0.0 for
    unused features, so the ranking always covers the full feature set.
    """
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if importances.shape[0] != len(feature_names):
        raise ValueError(
            f"model reports {importances.shape[0]} importances but "
            f"{len(feature_names)} feature names were supplied"
        )
    ranked = sorted(
        zip(feature_names, (float(value) for value in importances)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked


@dataclass(slots=True)
class ModelReport:
    """Everything measured for one trained model."""

    name: str
    params: dict[str, object]
    scale_pos_weight: float | None
    train_seconds: float
    selected_threshold: float
    metrics: dict[str, dict[str, BinaryMetrics]] = field(default_factory=dict)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)

    def metric(self, split_name: str, threshold_key: str) -> BinaryMetrics:
        return self.metrics[split_name][threshold_key]


def run_model(
    name: str,
    matrices: Mapping[str, tuple[np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    *,
    config: XGBoostBaselineConfig | None = None,
    scale_pos_weight: float | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> ModelReport:
    """Train one model and evaluate it on validation and test.

    The training split is used only for fitting. The threshold is chosen on
    validation and then applied unchanged to test.
    """
    config = config or XGBoostBaselineConfig()
    X_train, y_train = matrices[SPLIT_TRAIN]

    started = time.time()
    model = train_xgboost(
        X_train, y_train, config=config, scale_pos_weight=scale_pos_weight
    )
    train_seconds = time.time() - started

    X_validation, y_validation = matrices[SPLIT_VALIDATION]
    validation_scores = predict_scores(model, X_validation)
    selected_threshold = select_threshold_on_validation(
        y_validation, validation_scores
    )

    report = ModelReport(
        name=name,
        params=config.to_params(),
        scale_pos_weight=scale_pos_weight,
        train_seconds=train_seconds,
        selected_threshold=selected_threshold,
        feature_importance=feature_importance_ranking(model, feature_names),
    )

    for split_name in (SPLIT_VALIDATION, SPLIT_TEST):
        X_split, y_split = matrices[split_name]
        scores = (
            validation_scores
            if split_name == SPLIT_VALIDATION
            else predict_scores(model, X_split)
        )
        report.metrics[split_name] = {
            "default": evaluate_at_threshold(y_split, scores, default_threshold),
            "selected": evaluate_at_threshold(y_split, scores, selected_threshold),
        }

    return report


def run_baseline(
    dataset: MLDataset,
    *,
    config: XGBoostBaselineConfig | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, ModelReport]:
    """Train and evaluate both baselines on one prepared dataset.

    Returns ``{MODEL_STANDARD: report, MODEL_CLASS_WEIGHTED: report}``.
    """
    config = config or XGBoostBaselineConfig()
    matrices = dataset_matrices(dataset)
    feature_names = dataset.feature_names

    _, y_train = matrices[SPLIT_TRAIN]
    scale_pos_weight = compute_scale_pos_weight(y_train)

    return {
        MODEL_STANDARD: run_model(
            MODEL_STANDARD,
            matrices,
            feature_names,
            config=config,
            scale_pos_weight=None,
            default_threshold=default_threshold,
        ),
        MODEL_CLASS_WEIGHTED: run_model(
            MODEL_CLASS_WEIGHTED,
            matrices,
            feature_names,
            config=config,
            scale_pos_weight=scale_pos_weight,
            default_threshold=default_threshold,
        ),
    }
