"""Exact TreeSHAP attribution for the selected detector (M4.5).

Model selection
---------------
M4.5 explains "the best model from M4.4". The criterion the methodology permits
is VALIDATION PR-AUC -- never test. On W1 the standard M4.0 XGBoost has the
highest validation PR-AUC of the models compared in M4.4, so it is the model
explained here. Its test PR-AUC is also the highest, but that played no part in
the choice.

Method
------
For a gradient-boosted tree ensemble the appropriate attribution is TreeSHAP
(Lundberg et al.): exact Shapley values computed from the tree structure in
polynomial time, with no sampling and no background dataset beyond the training
cover already stored in the trees. XGBoost implements it natively
(``pred_contribs=True``), so no new dependency is needed and there is no
approximation error to validate away.

Properties this module relies on, and which the tests verify:

* **local accuracy** -- for every event the contributions plus the bias equal
  the model's raw log-odds output;
* **dummy** -- a feature the trees never split on receives exactly zero;
* **decision consistency** -- ``sigmoid(margin)`` is the probability the M4.0
  threshold is applied to, so the explained quantity is the decision quantity.

Contributions are additive in log-odds (margin) space, not probability space.

Label separation
----------------
:func:`explain_tree_model` takes a model and a feature matrix. It has no label
argument and cannot see ground truth. Labels appear in exactly one function,
:func:`outcome_indices`, which groups already-computed explanations into
TP/FP/FN/TN for reporting -- the same evaluation-only use the M4.0 metrics make
of them. Nothing produced here is fed back into a model, a threshold or a
feature.

Faithfulness
------------
Additivity proves the numbers are Shapley values of the model. It does not prove
they point at what drives the score. :func:`deletion_check` replaces the features
the attribution ranks highest, lowest and at random with their TRAIN-only median
and measures the resulting drop in margin. A faithful attribution must remove far
more score when its top features are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

OUTCOME_NAMES: tuple[str, ...] = ("tp", "fp", "fn", "tn")


class ExplainabilityError(ValueError):
    """Raised on malformed input to an attribution routine."""


def _booster(model):
    get_booster = getattr(model, "get_booster", None)
    return get_booster() if callable(get_booster) else model


def _as_matrix(X, width: int) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ExplainabilityError(
            f"expected a 2-D feature matrix, got shape {matrix.shape}"
        )
    if matrix.shape[1] != width:
        raise ExplainabilityError(
            f"expected {width} feature columns, got {matrix.shape[1]}"
        )
    return np.ascontiguousarray(matrix)


def model_margins(model, X) -> np.ndarray:
    """Raw log-odds output of a tree model, the quantity TreeSHAP decomposes."""
    from xgboost import DMatrix

    booster = _booster(model)
    matrix = _as_matrix(X, booster.num_features())
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return booster.predict(DMatrix(matrix), output_margin=True).astype(np.float64)


@dataclass(frozen=True, slots=True)
class TreeShapExplanation:
    """Per-event Shapley contributions in log-odds space.

    ``contributions`` has shape ``(events, features)``; ``bias`` and
    ``margins`` have shape ``(events,)``. For every row,
    ``contributions.sum() + bias == margin`` up to float32 rounding inside
    XGBoost.
    """

    feature_names: tuple[str, ...]
    contributions: np.ndarray
    bias: np.ndarray
    margins: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.contributions.shape[0])

    def local_accuracy_error(self) -> float:
        """Largest absolute gap between summed contributions and the margin."""
        if self.num_rows == 0:
            return 0.0
        reconstructed = self.contributions.sum(axis=1) + self.bias
        return float(np.max(np.abs(reconstructed - self.margins)))

    def subset(self, rows) -> "TreeShapExplanation":
        rows = np.asarray(rows, dtype=np.int64)
        return TreeShapExplanation(
            feature_names=self.feature_names,
            contributions=self.contributions[rows],
            bias=self.bias[rows],
            margins=self.margins[rows],
        )


def explain_tree_model(
    model, X, feature_names: Sequence[str]
) -> TreeShapExplanation:
    """Exact TreeSHAP contributions of ``model`` for every row of ``X``.

    Deliberately label-free: the only inputs are the fitted model and the
    feature matrix.
    """
    from xgboost import DMatrix

    names = tuple(feature_names)
    booster = _booster(model)
    if booster.num_features() != len(names):
        raise ExplainabilityError(
            f"model expects {booster.num_features()} features but "
            f"{len(names)} names were supplied"
        )
    matrix = _as_matrix(X, len(names))
    rows, width = matrix.shape
    if rows == 0:
        return TreeShapExplanation(
            feature_names=names,
            contributions=np.empty((0, width), dtype=np.float64),
            bias=np.empty(0, dtype=np.float64),
            margins=np.empty(0, dtype=np.float64),
        )

    dmatrix = DMatrix(matrix)
    raw = booster.predict(dmatrix, pred_contribs=True)
    if raw.shape != (rows, width + 1):
        raise ExplainabilityError(
            f"unexpected contribution shape {raw.shape}; a binary model should "
            f"return {(rows, width + 1)}"
        )
    margins = booster.predict(dmatrix, output_margin=True)
    return TreeShapExplanation(
        feature_names=names,
        contributions=raw[:, :width].astype(np.float64),
        bias=raw[:, width].astype(np.float64),
        margins=margins.astype(np.float64),
    )


# ----------------------------------------------------------------------
# Global and local summaries
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    """One feature's attribution summarised over a set of events."""

    feature: str
    mean_abs: float
    mean_signed: float
    share: float
    nonzero_fraction: float


def global_importance(explanation: TreeShapExplanation) -> list[FeatureAttribution]:
    """Features ranked by mean absolute contribution; ties break by name."""
    if explanation.num_rows == 0:
        raise ExplainabilityError("cannot rank features over zero events")
    phi = explanation.contributions
    mean_abs = np.abs(phi).mean(axis=0)
    mean_signed = phi.mean(axis=0)
    nonzero = (phi != 0.0).mean(axis=0)
    total = float(mean_abs.sum())
    ranked = [
        FeatureAttribution(
            feature=name,
            mean_abs=float(mean_abs[i]),
            mean_signed=float(mean_signed[i]),
            share=float(mean_abs[i] / total) if total > 0 else 0.0,
            nonzero_fraction=float(nonzero[i]),
        )
        for i, name in enumerate(explanation.feature_names)
    ]
    return sorted(ranked, key=lambda item: (-item.mean_abs, item.feature))


@dataclass(frozen=True, slots=True)
class LocalContribution:
    """One feature's value and contribution for a single event."""

    feature: str
    value: float
    contribution: float


def top_contributions(
    explanation: TreeShapExplanation, X, row: int, k: int = 5
) -> list[LocalContribution]:
    """The ``k`` largest-magnitude contributions for one event."""
    matrix = _as_matrix(X, len(explanation.feature_names))
    if matrix.shape[0] != explanation.num_rows:
        raise ExplainabilityError("X and the explanation have different row counts")
    if not 0 <= row < explanation.num_rows:
        raise ExplainabilityError(f"row {row} is out of range")
    if k <= 0:
        raise ExplainabilityError("k must be positive")
    phi = explanation.contributions[row]
    names = explanation.feature_names
    order = sorted(range(len(names)), key=lambda i: (-abs(phi[i]), names[i]))[:k]
    return [
        LocalContribution(
            feature=names[i], value=float(matrix[row, i]), contribution=float(phi[i])
        )
        for i in order
    ]


def rank_correlation(
    first: Mapping[str, float], second: Mapping[str, float]
) -> float | None:
    """Spearman correlation of two per-feature importance maps.

    Ties receive average ranks. Returns ``None`` when either map is constant,
    because the correlation is then undefined; reporting 0.0 would not be honest.
    """
    if set(first) != set(second):
        raise ExplainabilityError("importance maps cover different features")
    names = sorted(first)
    a = np.asarray([first[name] for name in names], dtype=np.float64)
    b = np.asarray([second[name] for name in names], dtype=np.float64)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)


# ----------------------------------------------------------------------
# Faithfulness
# ----------------------------------------------------------------------
def train_reference_values(X_train) -> np.ndarray:
    """Per-feature TRAIN medians: the neutral values for the deletion check.

    Computed from training data only (RESEARCH_CONSTRAINTS section 13), so no
    validation or test statistic can shape the check.
    """
    matrix = np.asarray(X_train, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ExplainabilityError("reference values need a non-empty TRAIN matrix")
    return np.median(matrix, axis=0)


@dataclass(frozen=True, slots=True)
class DeletionResult:
    """Mean margin drop after neutralising k features chosen three ways."""

    k: int
    rows: int
    mean_drop_top: float
    mean_drop_random: float
    mean_drop_bottom: float
    top_exceeds_random_fraction: float


def deletion_check(
    model,
    X,
    explanation: TreeShapExplanation,
    reference: np.ndarray,
    *,
    k: int,
    seed: int = 0,
) -> DeletionResult:
    """Replace k features per event with ``reference`` and measure margin drops.

    * ``top``    -- the k features with the largest positive contribution, i.e.
      the ones the attribution says push the event toward "attack";
    * ``bottom`` -- the k features with the smallest absolute contribution;
    * ``random`` -- k features drawn with a fixed seed, so the check is
      reproducible.

    A faithful attribution gives ``top`` >> ``random`` >= ``bottom``. Removing
    features jointly is not additive, so this is a sanity check of direction and
    magnitude, not a proof.
    """
    width = len(explanation.feature_names)
    matrix = _as_matrix(X, width)
    rows = matrix.shape[0]
    if rows != explanation.num_rows:
        raise ExplainabilityError("X and the explanation have different row counts")
    if rows == 0:
        raise ExplainabilityError("the deletion check needs at least one event")
    if not 1 <= k <= width:
        raise ExplainabilityError(f"k must be in [1, {width}]")
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (width,):
        raise ExplainabilityError(
            f"reference must have shape {(width,)}, got {reference.shape}"
        )

    phi = explanation.contributions
    rng = np.random.default_rng(seed)
    choices = {
        "top": np.argsort(-phi, axis=1, kind="stable")[:, :k],
        "random": np.stack(
            [rng.choice(width, size=k, replace=False) for _ in range(rows)]
        ),
        "bottom": np.argsort(np.abs(phi), axis=1, kind="stable")[:, :k],
    }

    baseline = model_margins(model, matrix)
    row_index = np.arange(rows)[:, None]
    drops: dict[str, np.ndarray] = {}
    for name, columns in choices.items():
        perturbed = matrix.copy()
        perturbed[row_index, columns] = reference[columns]
        drops[name] = baseline - model_margins(model, perturbed)

    return DeletionResult(
        k=k,
        rows=rows,
        mean_drop_top=float(drops["top"].mean()),
        mean_drop_random=float(drops["random"].mean()),
        mean_drop_bottom=float(drops["bottom"].mean()),
        top_exceeds_random_fraction=float(np.mean(drops["top"] > drops["random"])),
    )


# ----------------------------------------------------------------------
# Post-hoc grouping (the only place labels are used)
# ----------------------------------------------------------------------
def outcome_indices(y_true, scores, threshold: float) -> dict[str, np.ndarray]:
    """Row indices of TP/FP/FN/TN at a frozen threshold.

    Uses the same ``score >= threshold`` rule as
    :func:`ml.baselines.xgboost_baseline.evaluate_at_threshold`. This groups
    explanations for reporting only; it feeds nothing back into the model.
    """
    labels = np.asarray(y_true).astype(np.int64).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != values.shape:
        raise ExplainabilityError(
            f"labels {labels.shape} and scores {values.shape} must align"
        )
    predicted = values >= threshold
    actual = labels == 1
    return {
        "tp": np.flatnonzero(predicted & actual),
        "fp": np.flatnonzero(predicted & ~actual),
        "fn": np.flatnonzero(~predicted & actual),
        "tn": np.flatnonzero(~predicted & ~actual),
    }
