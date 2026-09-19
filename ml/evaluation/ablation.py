"""Information-source ablation of the neural event classifiers (M4.7).

Design
------
M4.4 compared three variants on W1 with one seed each. Later modules showed
that this is not enough to value information sources:

1. **The input design was incomplete.** Every M4.4 variant also feeds the
   classifier head the 7 target-event attributes, three of which are
   graph-derived "seen before" flags, so ``behavioral_only`` was never purely
   behavioural. With the two extra variants in
   :data:`ml.models.gnn.ABLATION_VARIANTS`, each source's marginal value is the
   difference between two variants that differ in exactly that source
   (:data:`MARGINAL_CONTRIBUTIONS`):

   ================================  =========================================
   contribution                      comparison
   ================================  =========================================
   graph, given target               graph_only - target_only
   graph, given behavioural+target   hybrid - behavioral_only
   behavioural, given target         behavioral_only - target_only
   behavioural, given graph+target   hybrid - graph_only
   target, given behavioural         behavioral_only - behavioral_no_target
   ================================  =========================================

2. **One seed is one draw.** PROJECT_STATE section 29 showed that a single run
   of a graph-bearing variant is a sample from a wide distribution, so each
   variant is trained with several seeds. Seeds estimate variance; they are not
   searched over.

3. **W1 does not predict W2.** Section 31 showed the selected model collapses on
   W2, so every trained model is also evaluated under the frozen W1 -> W2
   protocol of M4.6.

Nothing is tuned. Every variant uses the M4.4 configuration unchanged.

Protocol
--------
* :func:`train_variant` fits on the development TRAIN split, selects the epoch
  and threshold on its VALIDATION split, and returns an immutable
  :class:`TrainedVariant` holding a detached CPU copy of the weights.
* :func:`evaluate_trained_variant` rebuilds the model from those weights and
  scores batches at the frozen threshold. It never trains, never re-selects a
  threshold, and never refits the behavioural scaler.
* :func:`as_evaluation_batches` relabels a later window's batches as test data,
  so :class:`~ml.models.gnn.BehavioralScaler` refuses to fit on them even by
  accident.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ml.baselines.xgboost_baseline import (
    BinaryMetrics,
    evaluate_at_threshold,
    select_threshold_on_validation,
)
from ml.models.gnn import (
    MODEL_BEHAVIORAL_NO_TARGET,
    MODEL_BEHAVIORAL_ONLY,
    MODEL_GRAPH_ONLY,
    MODEL_HYBRID,
    MODEL_TARGET_ONLY,
    VARIANT_INPUTS,
    BatchedEventDataset,
    BehavioralScaler,
    EventGraphBatch,
    GNNExperimentConfig,
    build_model,
    count_parameters,
    predict_scores,
    train_model,
)
from ml.preprocessing.ml_dataset import SPLIT_TEST, SPLIT_VALIDATION

#: ``(label, variant with the source, variant without it)``. Each pair differs in
#: exactly one input block of :data:`ml.models.gnn.VARIANT_INPUTS`.
MARGINAL_CONTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    ("graph | target", MODEL_GRAPH_ONLY, MODEL_TARGET_ONLY),
    ("graph | behavioural+target", MODEL_HYBRID, MODEL_BEHAVIORAL_ONLY),
    ("behavioural | target", MODEL_BEHAVIORAL_ONLY, MODEL_TARGET_ONLY),
    ("behavioural | graph+target", MODEL_HYBRID, MODEL_GRAPH_ONLY),
    ("target | behavioural", MODEL_BEHAVIORAL_ONLY, MODEL_BEHAVIORAL_NO_TARGET),
)


class AblationError(ValueError):
    """Raised when the ablation protocol would be violated."""


@dataclass(frozen=True, slots=True)
class TrainedVariant:
    """One trained variant and seed, frozen with its validation threshold."""

    variant: str
    seed: int
    parameters: int
    state_dict: dict
    threshold: float
    best_epoch: int | None
    best_validation_average_precision: float | None
    training_seconds: float
    validation_metrics: BinaryMetrics
    test_metrics: BinaryMetrics


def _require_fitted(scaler: BehavioralScaler) -> None:
    if not scaler.is_fitted:
        raise AblationError(
            "the behavioural scaler must already be fitted on TRAIN batches"
        )


def train_variant(
    dataset: BatchedEventDataset,
    scaler: BehavioralScaler,
    *,
    variant: str,
    config: GNNExperimentConfig,
    device: str = "cpu",
) -> TrainedVariant:
    """Train one variant on TRAIN; select epoch and threshold on VALIDATION.

    The seeding order -- seed, build, train -- is the one the M4.4 experiment
    used, so a deterministic run with the same seed reproduces its results.
    """
    import torch

    if variant not in VARIANT_INPUTS:
        raise AblationError(f"unknown variant {variant!r}")
    _require_fitted(scaler)

    torch.manual_seed(config.seed)
    model = build_model(config, variant=variant).to(device)
    history = train_model(model, dataset, config=config, device=device, scaler=scaler)

    validation_labels = dataset.labels(SPLIT_VALIDATION)
    validation_scores = predict_scores(
        model, dataset.split(SPLIT_VALIDATION), device=device, scaler=scaler
    )
    threshold = float(select_threshold_on_validation(validation_labels, validation_scores))
    test_scores = predict_scores(
        model, dataset.split(SPLIT_TEST), device=device, scaler=scaler
    )
    state = {
        key: value.detach().to("cpu").clone()
        for key, value in model.state_dict().items()
    }

    return TrainedVariant(
        variant=variant,
        seed=int(config.seed),
        parameters=count_parameters(model),
        state_dict=state,
        threshold=threshold,
        best_epoch=history.best_epoch,
        best_validation_average_precision=history.best_validation_average_precision,
        training_seconds=float(history.seconds),
        validation_metrics=evaluate_at_threshold(
            validation_labels, validation_scores, threshold
        ),
        test_metrics=evaluate_at_threshold(
            dataset.labels(SPLIT_TEST), test_scores, threshold
        ),
    )


def restore_model(
    trained: TrainedVariant, *, config: GNNExperimentConfig, device: str = "cpu"
):
    """Rebuild a trained variant in evaluation mode from its frozen weights."""
    model = build_model(config, variant=trained.variant)
    model.load_state_dict(trained.state_dict)
    return model.to(device).eval()


def evaluate_trained_variant(
    trained: TrainedVariant,
    batches: Sequence[EventGraphBatch],
    scaler: BehavioralScaler,
    *,
    config: GNNExperimentConfig,
    device: str = "cpu",
) -> tuple[BinaryMetrics, np.ndarray]:
    """Score ``batches`` at the frozen threshold. Nothing is fitted or selected."""
    _require_fitted(scaler)
    model = restore_model(trained, config=config, device=device)
    scores = predict_scores(model, batches, device=device, scaler=scaler)
    labels = (
        np.concatenate([batch.labels for batch in batches])
        if batches
        else np.empty(0, dtype=np.int8)
    )
    return evaluate_at_threshold(labels, scores, trained.threshold), scores


def as_evaluation_batches(
    batches: Sequence[EventGraphBatch],
) -> list[EventGraphBatch]:
    """Relabel a later window's batches as test data. Arrays are shared.

    A later window is built with its own chronological split, so some of its
    batches would otherwise carry the TRAIN label -- and the behavioural scaler
    accepts TRAIN batches. Relabelling makes an accidental refit impossible.
    """
    return [dataclasses.replace(batch, split=SPLIT_TEST) for batch in batches]


def seed_summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean, minimum and maximum over seeds, skipping undefined values."""
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(defined),
        "mean": float(np.mean(defined)),
        "min": min(defined),
        "max": max(defined),
    }


def marginal_contributions(
    means: Mapping[str, float | None],
) -> list[tuple[str, float | None]]:
    """Difference in a metric between each pair of :data:`MARGINAL_CONTRIBUTIONS`.

    Returns ``None`` for a pair when either variant's value is missing.
    """
    result: list[tuple[str, float | None]] = []
    for label, with_source, without_source in MARGINAL_CONTRIBUTIONS:
        a, b = means.get(with_source), means.get(without_source)
        result.append((label, None if a is None or b is None else a - b))
    return result
