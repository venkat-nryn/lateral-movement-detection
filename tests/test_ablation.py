"""Unit tests for the M4.7 information-source ablation.

Fixtures are tiny, deterministic and in-memory, and every model trains on the
CPU. No ablation conclusion is derived from anything here; the real W1 and W2
runs live in ``scripts/run_ablation.py``.
"""

from __future__ import annotations

import inspect
import io
import os
import tokenize

import numpy as np
import pytest

from ml.evaluation import ablation
from ml.evaluation.ablation import (
    MARGINAL_CONTRIBUTIONS,
    AblationError,
    as_evaluation_batches,
    evaluate_trained_variant,
    marginal_contributions,
    restore_model,
    seed_summary,
    train_variant,
)
from ml.models import gnn
from ml.models.gnn import (
    ABLATION_VARIANTS,
    MODEL_BEHAVIORAL_NO_TARGET,
    MODEL_BEHAVIORAL_ONLY,
    MODEL_GRAPH_ONLY,
    MODEL_HYBRID,
    MODEL_TARGET_ONLY,
    MODEL_VARIANTS,
    TARGET_FEATURE_NAMES,
    BEHAVIORAL_FEATURE_NAMES,
    VARIANT_INPUTS,
    BatchedEventDataset,
    BehavioralScaler,
    GNNError,
    GNNExperimentConfig,
    build_model,
    count_parameters,
    forward_batch,
    iter_event_batches,
)
from ml.models.graph_data import pyg_available, torch_available
from ml.preprocessing.ml_dataset import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, TimestampRange
from ml.preprocessing.schema import CanonicalEvent

requires_pyg = pytest.mark.skipif(
    not (torch_available() and pyg_available()),
    reason="PyTorch and PyTorch Geometric are required",
)

CONFIG = GNNExperimentConfig(block_size=10, max_history_events=25, epochs=2)


def linear_events(count: int = 60, start: float = 100.0) -> list[CanonicalEvent]:
    return [
        CanonicalEvent(
            event_id=f"e{i:03d}",
            timestamp=start + i,
            user=f"U{i % 5}",
            source_host=f"H{i % 7}",
            destination_host=f"H{(i + 3) % 7}",
            event_type="authentication",
            success=(i % 4 != 0),
        )
        for i in range(count)
    ]


def build_dataset() -> BatchedEventDataset:
    events = linear_events()
    labels = frozenset(
        (e.timestamp, e.user, e.source_host, e.destination_host)
        for e in (events[10], events[30], events[50])
    )
    return BatchedEventDataset.from_batches(
        iter_event_batches(
            events,
            labels,
            timestamp_range=TimestampRange(start=100.0, end=159.0),
            emit_from_timestamp=100.0,
            config=CONFIG,
        )
    )


def fitted_scaler(dataset: BatchedEventDataset) -> BehavioralScaler:
    return BehavioralScaler().fit(dataset.split(SPLIT_TRAIN))


@pytest.fixture(scope="module")
def dataset() -> BatchedEventDataset:
    return build_dataset()


@pytest.fixture(scope="module")
def scaler(dataset) -> BehavioralScaler:
    return fitted_scaler(dataset)


# ----------------------------------------------------------------------
# 1. The input design
# ----------------------------------------------------------------------


def test_input_design_is_complete_and_leaves_m44_untouched():
    assert set(VARIANT_INPUTS) == set(ABLATION_VARIANTS)
    assert len(set(VARIANT_INPUTS.values())) == len(VARIANT_INPUTS)
    assert MODEL_VARIANTS == (MODEL_GRAPH_ONLY, MODEL_BEHAVIORAL_ONLY, MODEL_HYBRID)
    assert set(MODEL_VARIANTS) < set(ABLATION_VARIANTS)


def test_each_marginal_pair_differs_in_exactly_one_input_block():
    for label, with_source, without_source in MARGINAL_CONTRIBUTIONS:
        present, absent = VARIANT_INPUTS[with_source], VARIANT_INPUTS[without_source]
        differing = [i for i, (a, b) in enumerate(zip(present, absent)) if a != b]
        assert len(differing) == 1, label
        assert present[differing[0]] and not absent[differing[0]], label


@requires_pyg
def test_m44_variant_parameter_counts_are_unchanged():
    config = GNNExperimentConfig()
    assert count_parameters(build_model(config, variant=MODEL_GRAPH_ONLY)) == 10_285
    assert count_parameters(build_model(config, variant=MODEL_BEHAVIORAL_ONLY)) == 881
    assert count_parameters(build_model(config, variant=MODEL_HYBRID)) == 10_863


@requires_pyg
def test_new_variant_parameter_counts_match_their_inputs():
    hidden = GNNExperimentConfig().hidden_dim

    def head_only(width: int) -> int:
        # LayerNorm(width) + Linear(width, hidden) + Linear(hidden, 1)
        return 2 * width + (width * hidden + hidden) + (hidden + 1)

    config = GNNExperimentConfig()
    assert count_parameters(build_model(config, variant=MODEL_TARGET_ONLY)) == head_only(
        len(TARGET_FEATURE_NAMES)
    )
    assert count_parameters(
        build_model(config, variant=MODEL_BEHAVIORAL_NO_TARGET)
    ) == head_only(len(BEHAVIORAL_FEATURE_NAMES))


@requires_pyg
def test_a_model_with_no_input_block_is_rejected():
    with pytest.raises(GNNError, match="at least one"):
        gnn._model_class()(
            node_dim=8,
            edge_dim=7,
            target_dim=7,
            behavioral_dim=17,
            hidden_dim=4,
            heads=1,
            dropout=0.0,
            use_graph=False,
            use_behavioral=False,
            use_target=False,
        )


@requires_pyg
def test_target_only_ignores_graph_and_behavioural_inputs():
    import torch

    data = build_dataset()
    scaler = fitted_scaler(data)
    torch.manual_seed(1)
    model = build_model(CONFIG, variant=MODEL_TARGET_ONLY)
    batch = data.batches[2]

    before = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.x[:] = 999.0
    batch.edge_attr[:] = -5.0
    batch.behavioral[:] = batch.behavioral + 50.0
    after = forward_batch(model, batch, scaler=scaler).detach()

    assert torch.equal(before, after)


@requires_pyg
def test_behavioral_no_target_ignores_target_attributes_only():
    import torch

    data = build_dataset()
    scaler = fitted_scaler(data)
    torch.manual_seed(2)
    model = build_model(CONFIG, variant=MODEL_BEHAVIORAL_NO_TARGET)
    batch = data.batches[2]

    before = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.target_attr[:] = batch.target_attr + 9.0
    after_target = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.behavioral[:] = batch.behavioral + 50.0
    after_behavioral = forward_batch(model, batch, scaler=scaler).detach()

    assert torch.equal(before, after_target)
    assert not torch.equal(before, after_behavioral)


# ----------------------------------------------------------------------
# 2. Frozen evaluation protocol
# ----------------------------------------------------------------------


def test_evaluation_batches_are_relabelled_and_share_arrays(dataset):
    relabelled = as_evaluation_batches(dataset.batches)

    assert len(relabelled) == len(dataset.batches)
    assert {batch.split for batch in relabelled} == {SPLIT_TEST}
    for original, copy in zip(dataset.batches, relabelled):
        assert copy.x is original.x
        assert copy.labels is original.labels


def test_scaler_refuses_to_fit_on_evaluation_batches(dataset):
    with pytest.raises(GNNError, match="train"):
        BehavioralScaler().fit(as_evaluation_batches(dataset.split(SPLIT_TRAIN)))


@requires_pyg
def test_train_variant_selects_the_threshold_on_validation(dataset, scaler, monkeypatch):
    seen = {}
    real = ablation.select_threshold_on_validation

    def spy(labels, scores):
        seen["rows"] = len(labels)
        seen["threshold"] = real(labels, scores)
        return seen["threshold"]

    monkeypatch.setattr(ablation, "select_threshold_on_validation", spy)
    trained = train_variant(dataset, scaler, variant=MODEL_BEHAVIORAL_ONLY, config=CONFIG)

    assert seen["rows"] == dataset.counts()[SPLIT_VALIDATION]["events"]
    assert trained.threshold == pytest.approx(seen["threshold"])
    assert trained.validation_metrics.threshold == trained.threshold


@requires_pyg
def test_restored_model_reproduces_the_recorded_test_metrics(dataset, scaler):
    trained = train_variant(dataset, scaler, variant=MODEL_HYBRID, config=CONFIG)
    metrics, _ = evaluate_trained_variant(
        trained, dataset.split(SPLIT_TEST), scaler, config=CONFIG
    )
    assert metrics.as_dict() == trained.test_metrics.as_dict()


@requires_pyg
def test_evaluation_never_trains_or_reselects(dataset, scaler, monkeypatch):
    trained = train_variant(dataset, scaler, variant=MODEL_TARGET_ONLY, config=CONFIG)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not train or select a threshold")

    monkeypatch.setattr(ablation, "train_model", forbidden)
    monkeypatch.setattr(ablation, "select_threshold_on_validation", forbidden)
    metrics, scores = evaluate_trained_variant(
        trained, as_evaluation_batches(dataset.batches), scaler, config=CONFIG
    )

    assert metrics.threshold == trained.threshold
    assert scores.shape == (sum(b.num_targets for b in dataset.batches),)


@requires_pyg
def test_evaluation_leaves_weights_and_scaler_untouched(dataset, scaler):
    import torch

    trained = train_variant(dataset, scaler, variant=MODEL_GRAPH_ONLY, config=CONFIG)
    weights = {key: value.clone() for key, value in trained.state_dict.items()}
    mean, scale = scaler.mean.copy(), scaler.scale.copy()

    model = restore_model(trained, config=CONFIG)
    for parameter in model.parameters():
        parameter.data.add_(1.0)
    evaluate_trained_variant(trained, dataset.batches, scaler, config=CONFIG)

    for key, value in weights.items():
        assert torch.equal(trained.state_dict[key], value)
    assert np.array_equal(scaler.mean, mean)
    assert np.array_equal(scaler.scale, scale)


@requires_pyg
def test_training_and_evaluation_require_a_fitted_scaler(dataset):
    with pytest.raises(AblationError, match="fitted"):
        train_variant(dataset, BehavioralScaler(), variant=MODEL_HYBRID, config=CONFIG)


@requires_pyg
def test_training_is_reproducible_and_seeds_change_initialisation(dataset, scaler):
    import torch
    from dataclasses import replace

    first = train_variant(dataset, scaler, variant=MODEL_BEHAVIORAL_ONLY, config=CONFIG)
    again = train_variant(dataset, scaler, variant=MODEL_BEHAVIORAL_ONLY, config=CONFIG)
    other = train_variant(
        dataset, scaler, variant=MODEL_BEHAVIORAL_ONLY, config=replace(CONFIG, seed=1)
    )

    assert all(torch.equal(first.state_dict[k], again.state_dict[k]) for k in first.state_dict)
    assert not all(torch.equal(first.state_dict[k], other.state_dict[k]) for k in first.state_dict)
    assert (first.seed, other.seed) == (0, 1)


# ----------------------------------------------------------------------
# 3. Summaries
# ----------------------------------------------------------------------


def test_seed_summary_skips_undefined_values():
    summary = seed_summary([0.2, None, 0.4])
    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(0.3)
    assert (summary["min"], summary["max"]) == (0.2, 0.4)
    assert seed_summary([None]) == {"n": 0, "mean": None, "min": None, "max": None}


def test_marginal_contributions_are_pairwise_differences():
    means = {
        MODEL_TARGET_ONLY: 0.10,
        MODEL_BEHAVIORAL_NO_TARGET: 0.80,
        MODEL_BEHAVIORAL_ONLY: 0.90,
        MODEL_GRAPH_ONLY: 0.20,
        MODEL_HYBRID: None,
    }
    result = dict(marginal_contributions(means))

    assert result["graph | target"] == pytest.approx(0.10)
    assert result["behavioural | target"] == pytest.approx(0.80)
    assert result["target | behavioural"] == pytest.approx(0.10)
    assert result["graph | behavioural+target"] is None
    assert result["behavioural | graph+target"] is None


# ----------------------------------------------------------------------
# 4. Hygiene
# ----------------------------------------------------------------------


def test_module_never_references_redteam():
    raw = inspect.getsource(ablation)
    code = " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(raw).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    ).lower()
    assert "redteam" not in code
    assert "ground_truth" not in code


@requires_pyg
def test_enable_deterministic_cuda_sets_the_workspace_and_flag(monkeypatch):
    import torch

    # setenv then delenv makes monkeypatch restore the variable's original
    # absence after the test, even though the function under test sets it.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "placeholder")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)

    gnn.enable_deterministic_cuda()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [True]
