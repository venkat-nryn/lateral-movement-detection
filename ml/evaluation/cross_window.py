"""Frozen cross-window generalization evaluation (M4.6).

Protocol
--------
RESEARCH_CONSTRAINTS section 9 defines cross-window evaluation as:

1. fit using the development window's TRAIN split only;
2. select the decision threshold using its VALIDATION split only;
3. freeze the model and the threshold;
4. evaluate on a later, temporally disjoint window.

This module turns those steps into code that cannot be quietly bypassed:

* :func:`fit_frozen_detector` is the only function that trains or selects a
  threshold, and it reads nothing but the development window's TRAIN and
  VALIDATION splits.
* :class:`FrozenDetector` is immutable and stores a SHA-256 digest of the
  serialised booster. :func:`evaluate_frozen` re-verifies that digest before and
  after scoring, so a model altered after freezing is detected, not silently
  evaluated.
* :func:`evaluate_frozen` has no code path that fits, refits or re-selects a
  threshold. The evaluation window's labels are used only to compute metrics.
* :func:`assert_generalization_windows` requires the evaluation window's entire
  read region -- context included -- to start after the development window's
  emission ends.

It also closes a reproducibility gap: the M4.1 cross-window result recorded in
PROJECT_STATE.md section 16 was produced ad hoc, with no code in the repository
able to reproduce it.

The detector is the standard M4.0 XGBoost, selected on validation PR-AUC in M4.5
(PROJECT_STATE.md section 30.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ml.baselines.xgboost_baseline import (
    BinaryMetrics,
    XGBoostBaselineConfig,
    evaluate_at_threshold,
    predict_scores,
    select_threshold_on_validation,
    split_matrix,
    train_xgboost,
)
from ml.evaluation.redteam import RedteamRecord
from ml.preprocessing.ml_dataset import (
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    MLDataset,
)
from ml.preprocessing.ml_window import WindowSpec


class CrossWindowError(ValueError):
    """Raised when the cross-window protocol would be violated."""


def model_digest(model) -> str:
    """SHA-256 of the serialised booster; changes if the model changes."""
    get_booster = getattr(model, "get_booster", None)
    booster = get_booster() if callable(get_booster) else model
    return hashlib.sha256(bytes(booster.save_raw(raw_format="ubj"))).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenDetector:
    """A trained model and its validation-selected threshold, frozen together."""

    model: object
    threshold: float
    feature_names: tuple[str, ...]
    digest: str
    train_rows: int
    train_positives: int
    validation_rows: int
    validation_positives: int
    validation_metrics: BinaryMetrics

    def verify_unchanged(self) -> None:
        if model_digest(self.model) != self.digest:
            raise CrossWindowError(
                "the frozen model has been modified since it was fitted"
            )

    def scores(self, X) -> np.ndarray:
        """Positive-class probabilities from the frozen model."""
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise CrossWindowError(
                f"expected a matrix with {len(self.feature_names)} feature "
                f"columns, got shape {matrix.shape}"
            )
        self.verify_unchanged()
        return predict_scores(self.model, matrix)


def fit_frozen_detector(
    dataset: MLDataset, *, config: XGBoostBaselineConfig | None = None
) -> FrozenDetector:
    """Train on TRAIN, select the threshold on VALIDATION, and freeze both.

    This is the standard (unweighted) M4.0 XGBoost. No other split is read.
    """
    X_train, y_train = split_matrix(dataset.get_split(SPLIT_TRAIN))
    X_val, y_val = split_matrix(dataset.get_split(SPLIT_VALIDATION))
    if X_train.shape[0] == 0 or not np.any(y_train):
        raise CrossWindowError(
            "the development TRAIN split must contain at least one positive"
        )
    if X_val.shape[0] == 0:
        raise CrossWindowError("the development VALIDATION split is empty")

    model = train_xgboost(X_train, y_train, config=config or XGBoostBaselineConfig())
    validation_scores = predict_scores(model, X_val)
    threshold = float(select_threshold_on_validation(y_val, validation_scores))

    return FrozenDetector(
        model=model,
        threshold=threshold,
        feature_names=tuple(dataset.feature_names),
        digest=model_digest(model),
        train_rows=int(X_train.shape[0]),
        train_positives=int(np.count_nonzero(y_train)),
        validation_rows=int(X_val.shape[0]),
        validation_positives=int(np.count_nonzero(y_val)),
        validation_metrics=evaluate_at_threshold(y_val, validation_scores, threshold),
    )


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """Every emitted event of a window, in chronological order."""

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    event_ids: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def positives(self) -> int:
        return int(np.count_nonzero(self.y))


def evaluation_window(dataset: MLDataset) -> EvaluationWindow:
    """Concatenate all splits of a window into one chronological evaluation set.

    A frozen evaluation uses the whole emitted region; the window's own
    train/validation/test split is irrelevant because nothing is fitted on it.
    Context events are not emitted and therefore never appear.
    """
    if not dataset.retained_samples:
        raise CrossWindowError(
            "the evaluation window was built without retained samples"
        )
    splits = [dataset.get_split(name) for name in SPLIT_NAMES]
    matrices = [split_matrix(split) for split in splits]
    X = np.vstack([matrix for matrix, _ in matrices]).astype(np.float64, copy=False)
    y = np.concatenate([labels for _, labels in matrices]).astype(np.int8, copy=False)
    timestamps = np.concatenate(
        [np.asarray(split.timestamps, dtype=np.float64) for split in splits]
    )
    event_ids = tuple(event_id for split in splits for event_id in split.event_ids)

    if not (X.shape[0] == y.shape[0] == timestamps.shape[0] == len(event_ids)):
        raise CrossWindowError("evaluation window arrays are misaligned")
    if timestamps.size and np.any(np.diff(timestamps) < 0):
        raise CrossWindowError("evaluation window is not in chronological order")
    return EvaluationWindow(X=X, y=y, timestamps=timestamps, event_ids=event_ids)


def evaluate_frozen(
    detector: FrozenDetector, window: EvaluationWindow
) -> tuple[BinaryMetrics, np.ndarray]:
    """Score a window with the frozen model and threshold.

    Returns ``(metrics, scores)``. The window's labels are used only by
    :func:`evaluate_at_threshold`.
    """
    detector.verify_unchanged()
    scores = detector.scores(window.X)
    metrics = evaluate_at_threshold(window.y, scores, detector.threshold)
    detector.verify_unchanged()
    return metrics, scores


def assert_generalization_windows(
    development: WindowSpec, evaluation: WindowSpec
) -> None:
    """Require the evaluation window to be read entirely after development.

    Comparing ``context_start`` against ``emit_end`` covers both overlap and
    ordering: the evaluation window's history must not reach back into the
    development window.
    """
    if evaluation.context_start <= development.emit_end:
        raise CrossWindowError(
            f"evaluation window read region starts at "
            f"{evaluation.context_start}, not strictly after the development "
            f"window emission end {development.emit_end}"
        )


def quantile_summary(values) -> dict[str, float | int | None]:
    """Distribution summary used for fan-out profile comparisons."""
    array = np.asarray(values, dtype=np.float64).ravel()
    keys = ("min", "p25", "median", "p75", "p99", "max")
    if array.size == 0:
        return {"n": 0, **{key: None for key in keys}}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
    return {"n": int(array.size), **{k: float(q) for k, q in zip(keys, quantiles)}}


def positive_source_hosts(
    positive_timestamps: Sequence[float], records: Sequence[RedteamRecord]
) -> list[str | None]:
    """Source host behind each labeled positive, or ``None`` when ambiguous.

    A positive matched a redteam record exactly, so at least one record shares
    its timestamp. When every record at that timestamp names the same source
    host, the host is unambiguous. When they disagree -- or, defensively, when
    none exists -- ``None`` is returned rather than a guess. Used for reporting
    only; labels are never altered.
    """
    by_timestamp: dict[float, set[str]] = {}
    for record in records:
        by_timestamp.setdefault(float(record.timestamp), set()).add(record.source_host)

    hosts: list[str | None] = []
    for timestamp in positive_timestamps:
        candidates = by_timestamp.get(float(timestamp), set())
        hosts.append(next(iter(candidates)) if len(candidates) == 1 else None)
    return hosts
