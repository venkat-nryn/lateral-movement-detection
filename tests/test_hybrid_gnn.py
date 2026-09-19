"""Unit tests for the M4.4 hybrid graph + behavioural classifier.

Every fixture is a tiny deterministic in-memory object. Nothing here reads
``auth.txt.gz`` or ``redteam.txt.gz``, and no number produced by this file is a
research result. The real W1 comparison lives in
``scripts/run_gnn_experiment.py``.

The M4.3 batch/temporal/leakage properties are already covered by
``tests/test_gnn.py``; this file tests only what M4.4 adds -- the behavioural
block, the train-only scaler and the three model variants.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest

from ml.models import gnn
from ml.models.gnn import (
    BEHAVIORAL_FEATURE_NAMES,
    MODEL_BEHAVIORAL_ONLY,
    MODEL_GRAPH_ONLY,
    MODEL_HYBRID,
    MODEL_VARIANTS,
    TARGET_FEATURE_NAMES,
    BatchedEventDataset,
    BehavioralScaler,
    GNNError,
    GNNExperimentConfig,
    build_model,
    count_parameters,
    forward_batch,
    iter_event_batches,
    predict_scores,
    signed_log1p,
    train_model,
)
from ml.models.graph_data import pyg_available, torch_available
from ml.preprocessing.features import (
    NO_PRIOR_TIME_DELTA,
    EventFeatures,
    TemporalGraphFeatureExtractor,
)
from ml.preprocessing.ml_dataset import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TimestampRange,
)
from ml.preprocessing.schema import CanonicalEvent

requires_pyg = pytest.mark.skipif(
    not (torch_available() and pyg_available()),
    reason="PyTorch and PyTorch Geometric are required",
)


def make_event(
    event_id: str,
    timestamp: float,
    user: str = "U1",
    source_host: str = "H1",
    destination_host: str = "H2",
    success: bool = True,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp=timestamp,
        user=user,
        source_host=source_host,
        destination_host=destination_host,
        event_type="authentication",
        success=success,
    )


def linear_events(count: int = 60, start: float = 100.0) -> list[CanonicalEvent]:
    return [
        make_event(
            f"e{i:03d}",
            start + i,
            user=f"U{i % 5}",
            source_host=f"H{i % 7}",
            destination_host=f"H{(i + 3) % 7}",
            success=(i % 4 != 0),
        )
        for i in range(count)
    ]


def label_keys(events, positions) -> frozenset:
    return frozenset(
        (
            events[p].timestamp,
            events[p].user,
            events[p].source_host,
            events[p].destination_host,
        )
        for p in positions
    )


def build_dataset(
    events=None, *, labels=(10, 30, 50), emit_from=None, config=None
) -> BatchedEventDataset:
    events = events if events is not None else linear_events()
    emit_from = emit_from if emit_from is not None else events[0].timestamp
    emitted = [e for e in events if e.timestamp >= emit_from]
    return BatchedEventDataset.from_batches(
        iter_event_batches(
            events,
            label_keys(events, labels),
            timestamp_range=TimestampRange(
                start=emitted[0].timestamp, end=emitted[-1].timestamp
            ),
            emit_from_timestamp=emit_from,
            config=config
            or GNNExperimentConfig(block_size=10, max_history_events=25),
        )
    )


def fitted_scaler(dataset: BatchedEventDataset) -> BehavioralScaler:
    return BehavioralScaler().fit(dataset.split(SPLIT_TRAIN))


# ----------------------------------------------------------------------
# 1. All 17 features reach the classifier, unchanged
# ----------------------------------------------------------------------


def test_behavioral_block_is_exactly_the_17_m34_features():
    assert BEHAVIORAL_FEATURE_NAMES == EventFeatures.feature_names()
    assert len(BEHAVIORAL_FEATURE_NAMES) == 17


def test_behavioral_array_shape_and_dtype():
    dataset = build_dataset()
    for batch in dataset.batches:
        assert batch.behavioral.shape == (batch.num_targets, 17)
        assert batch.behavioral.dtype == np.float32
        assert np.isfinite(batch.behavioral).all()
        batch.validate()


def test_behavioral_values_match_the_standalone_m34_extractor():
    """The batched features must equal a plain M3.4 streaming pass."""
    events = linear_events(60)
    dataset = build_dataset(events)

    reference = TemporalGraphFeatureExtractor()
    expected = np.asarray(
        [reference.process_event(event).to_vector() for event in events],
        dtype=np.float32,
    )
    produced = np.concatenate([batch.behavioral for batch in dataset.batches])

    assert produced.shape == expected.shape
    assert np.array_equal(produced, expected)


def test_context_events_feed_the_extractor_without_becoming_targets():
    """History built in the context region must reach the emitted features."""
    events = linear_events(30)
    emit_from = events[10].timestamp

    with_context = build_dataset(events, labels=(), emit_from=emit_from)
    produced = np.concatenate([b.behavioral for b in with_context.batches])

    reference = TemporalGraphFeatureExtractor()
    expected = np.asarray(
        [reference.process_event(event).to_vector() for event in events],
        dtype=np.float32,
    )[10:]

    assert produced.shape == (20, 17)
    assert np.array_equal(produced, expected)

    # Direct proof the context mattered: a stream that starts at the emission
    # boundary has no prior activity for the same first target event.
    without_context = build_dataset(events[10:], labels=())
    cold = np.concatenate([b.behavioral for b in without_context.batches])
    column = BEHAVIORAL_FEATURE_NAMES.index("user_recent_event_count")

    assert produced[0, column] > 0.0
    assert cold[0, column] == 0.0


def test_every_feature_column_is_wired_through_to_the_head():
    """No column may be silently dropped between batch and model input."""
    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    batch = dataset.batches[0]

    scaled = scaler.transform(batch.behavioral)
    assert scaled.shape == (batch.num_targets, 17)

    model = build_model(GNNExperimentConfig(), variant=MODEL_HYBRID)
    expected_in = 3 * 32 + len(TARGET_FEATURE_NAMES) + 17
    assert model.head[0].in_features == expected_in


# ----------------------------------------------------------------------
# 2. Train-only preprocessing statistics
# ----------------------------------------------------------------------


def test_scaler_is_fitted_on_train_batches_only():
    dataset = build_dataset()
    scaler = BehavioralScaler().fit(dataset.split(SPLIT_TRAIN))

    train_rows = sum(b.num_targets for b in dataset.split(SPLIT_TRAIN))
    assert scaler.fitted_on_events == train_rows
    assert scaler.is_fitted


def test_scaler_refuses_validation_or_test_batches():
    dataset = build_dataset()
    for split in (SPLIT_VALIDATION, SPLIT_TEST):
        with pytest.raises(GNNError, match="train"):
            BehavioralScaler().fit(dataset.split(split))
    with pytest.raises(GNNError, match="train"):
        BehavioralScaler().fit(dataset.batches)


def test_scaler_statistics_ignore_validation_and_test_data():
    """Adding later events must not move the frozen statistics."""
    events = linear_events(60)
    span = TimestampRange(start=events[0].timestamp, end=events[-1].timestamp)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)

    def build(source):
        return BatchedEventDataset.from_batches(
            iter_event_batches(
                source,
                frozenset(),
                timestamp_range=span,
                emit_from_timestamp=events[0].timestamp,
                config=config,
            )
        )

    full = BehavioralScaler().fit(build(events).split(SPLIT_TRAIN))
    train_region_only = BehavioralScaler().fit(build(events[:42]).split(SPLIT_TRAIN))

    assert np.array_equal(full.mean, train_region_only.mean)
    assert np.array_equal(full.scale, train_region_only.scale)


def test_scaler_cannot_be_refitted():
    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    with pytest.raises(GNNError, match="already fitted"):
        scaler.fit(dataset.split(SPLIT_TRAIN))


def test_scaler_must_be_fitted_before_use():
    with pytest.raises(GNNError, match="must be fitted"):
        BehavioralScaler().transform(np.zeros((2, 17), dtype=np.float32))


def test_scaler_rejects_an_empty_fit():
    with pytest.raises(GNNError, match="zero batches"):
        BehavioralScaler().fit([])


def test_scaler_standardises_the_training_distribution():
    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    scaled = np.concatenate(
        [scaler.transform(b.behavioral) for b in dataset.split(SPLIT_TRAIN)]
    )

    varying = scaler.scale > 1.0 + 1e-9
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(scaled[:, varying].std(axis=0), 1.0, atol=1e-3)


def test_constant_training_columns_become_zero_not_infinite():
    """A feature that never varies in training must not divide by ~0."""
    events = [make_event(f"e{i}", 100.0 + i) for i in range(20)]
    dataset = build_dataset(events, labels=())
    scaler = fitted_scaler(dataset)
    scaled = scaler.transform(dataset.split(SPLIT_TRAIN)[0].behavioral)

    assert np.isfinite(scaled).all()
    # source_equals_destination is constant here (H1 -> H2 always).
    column = BEHAVIORAL_FEATURE_NAMES.index("source_equals_destination")
    assert scaler.scale[column] == 1.0


def test_signed_log1p_keeps_the_no_prior_sentinel_distinct_from_zero():
    """log1p(clamp(min=0)) would collapse -1.0 onto a real zero-second gap."""
    values = np.array([[NO_PRIOR_TIME_DELTA, 0.0, 3600.0]], dtype=np.float32)
    compressed = signed_log1p(values)

    assert compressed[0, 0] < 0.0
    assert compressed[0, 1] == 0.0
    assert compressed[0, 0] != compressed[0, 1]
    assert np.isfinite(compressed).all()


def test_scaler_transform_is_deterministic():
    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    batch = dataset.batches[0]
    assert np.array_equal(
        scaler.transform(batch.behavioral), scaler.transform(batch.behavioral)
    )


# ----------------------------------------------------------------------
# 3. No leakage introduced by the behavioural branch
# ----------------------------------------------------------------------


def executable_source(module) -> str:
    raw = inspect.getsource(module)
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(raw).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept).lower()


def test_module_still_never_references_redteam():
    code = executable_source(gnn)
    assert "redteam" not in code
    assert "ml.evaluation" not in code


def test_labels_do_not_influence_the_behavioral_block():
    events = linear_events(60)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)
    without = build_dataset(events, labels=(), config=config)
    with_labels = build_dataset(events, labels=(10, 30, 50), config=config)

    for plain, labeled in zip(without.batches, with_labels.batches):
        assert np.array_equal(plain.behavioral, labeled.behavioral)
    assert sum(b.positive_count() for b in with_labels.batches) == 3


def test_behavioral_features_are_causal_for_the_first_event():
    """The very first event has no history, so counts are 0 and gaps are -1."""
    dataset = build_dataset(linear_events(20), labels=())
    first = dataset.batches[0].behavioral[0]

    for name in (
        "user_destination_count",
        "source_destination_count",
        "user_unique_destination_count",
        "user_recent_event_count",
    ):
        assert first[BEHAVIORAL_FEATURE_NAMES.index(name)] == 0.0
    for name in (
        "time_since_user_previous_event",
        "time_since_user_destination_event",
    ):
        assert first[BEHAVIORAL_FEATURE_NAMES.index(name)] == NO_PRIOR_TIME_DELTA


def test_future_events_cannot_change_earlier_behavioral_features():
    events = linear_events(30)
    span = TimestampRange(start=events[0].timestamp, end=events[-1].timestamp)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)

    def build(source):
        return BatchedEventDataset.from_batches(
            iter_event_batches(
                source,
                frozenset(),
                timestamp_range=span,
                emit_from_timestamp=events[0].timestamp,
                config=config,
            )
        )

    full = build(events)
    truncated = build(events[:20])
    for early, late in zip(truncated.batches, full.batches):
        assert np.array_equal(early.behavioral, late.behavioral)


# ----------------------------------------------------------------------
# 4. Deterministic preprocessing
# ----------------------------------------------------------------------


def test_behavioral_preprocessing_is_byte_identical_across_builds():
    first, second = build_dataset(), build_dataset()
    for a, b in zip(first.batches, second.batches):
        assert np.array_equal(a.behavioral, b.behavioral)

    scaler_a, scaler_b = fitted_scaler(first), fitted_scaler(second)
    assert np.array_equal(scaler_a.mean, scaler_b.mean)
    assert np.array_equal(scaler_a.scale, scaler_b.scale)


# ----------------------------------------------------------------------
# 5. Model variants: dimensions, forward and backward
# ----------------------------------------------------------------------


@requires_pyg
def test_unknown_variant_is_rejected():
    with pytest.raises(GNNError, match="unknown variant"):
        build_model(GNNExperimentConfig(), variant="graph_and_vibes")


@requires_pyg
def test_variant_parameter_counts_and_branches():
    config = GNNExperimentConfig()
    graph = build_model(config, variant=MODEL_GRAPH_ONLY)
    behavioral = build_model(config, variant=MODEL_BEHAVIORAL_ONLY)
    hybrid = build_model(config, variant=MODEL_HYBRID)

    assert (graph.use_graph, graph.use_behavioral) == (True, False)
    assert (behavioral.use_graph, behavioral.use_behavioral) == (False, True)
    assert (hybrid.use_graph, hybrid.use_behavioral) == (True, True)
    # The hybrid is the graph model plus the behavioural block only.
    assert count_parameters(hybrid) > count_parameters(graph)
    assert count_parameters(behavioral) < count_parameters(graph)


@requires_pyg
def test_graph_only_variant_is_unchanged_from_m43():
    """M4.3's recorded architecture must survive the M4.4 extension."""
    model = build_model(GNNExperimentConfig(), variant=MODEL_GRAPH_ONLY)
    assert count_parameters(model) == 10_285
    assert not hasattr(model, "behavioral_norm")


@requires_pyg
def test_every_variant_produces_one_logit_per_event():
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    for variant in MODEL_VARIANTS:
        model = build_model(GNNExperimentConfig(), variant=variant)
        for batch in dataset.batches:
            logits = forward_batch(model, batch, scaler=scaler)
            assert logits.shape == (batch.num_targets,)
            assert torch.isfinite(logits).all()


@requires_pyg
def test_behavioral_variant_ignores_the_graph_tensors():
    """Corrupting the graph must not change a graph-free model's output."""
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    torch.manual_seed(3)
    model = build_model(GNNExperimentConfig(), variant=MODEL_BEHAVIORAL_ONLY)

    batch = dataset.batches[2]
    before = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.x[:] = 999.0
    batch.edge_attr[:] = -5.0
    after = forward_batch(model, batch, scaler=scaler).detach()

    assert torch.equal(before, after)


@requires_pyg
def test_hybrid_variant_actually_uses_the_behavioral_block():
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    torch.manual_seed(5)
    model = build_model(GNNExperimentConfig(), variant=MODEL_HYBRID)

    batch = dataset.batches[2]
    before = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.behavioral[:] = batch.behavioral + 10.0
    after = forward_batch(model, batch, scaler=scaler).detach()

    assert not torch.equal(before, after)


@requires_pyg
def test_hybrid_variant_also_uses_the_graph_branch():
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    torch.manual_seed(5)
    model = build_model(GNNExperimentConfig(), variant=MODEL_HYBRID)

    batch = dataset.batches[2]
    before = forward_batch(model, batch, scaler=scaler).detach().clone()
    batch.x[:] = batch.x + 7.0
    after = forward_batch(model, batch, scaler=scaler).detach()

    assert not torch.equal(before, after)


@requires_pyg
def test_backward_pass_reaches_every_branch_of_the_hybrid():
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)
    torch.manual_seed(0)
    model = build_model(GNNExperimentConfig(), variant=MODEL_HYBRID)

    logits = forward_batch(model, dataset.batches[3], scaler=scaler)
    logits.sum().backward()

    named = dict(model.named_parameters())
    for name in ("conv1.lin.weight", "behavioral_norm.weight", "head.0.weight"):
        gradient = named[name].grad
        assert gradient is not None, f"{name} received no gradient"
        assert torch.isfinite(gradient).all()
    assert named["head.0.weight"].grad.abs().sum() > 0


@requires_pyg
def test_a_behavioral_variant_without_its_block_is_an_error():
    dataset = build_dataset()
    model = build_model(GNNExperimentConfig(), variant=MODEL_BEHAVIORAL_ONLY)
    tensors = dataset.batches[0].to_device("cpu")
    with pytest.raises(GNNError, match="behavioural feature block"):
        model(
            tensors["x"],
            tensors["edge_index"],
            tensors["edge_attr"],
            tensors["user_index"],
            tensors["source_index"],
            tensors["dest_index"],
            tensors["target_attr"],
            None,
        )


@requires_pyg
def test_training_runs_for_every_variant():
    dataset = build_dataset(linear_events(60))
    scaler = fitted_scaler(dataset)
    for variant in MODEL_VARIANTS:
        model = build_model(GNNExperimentConfig(), variant=variant)
        history = train_model(
            model,
            dataset,
            config=GNNExperimentConfig(epochs=2),
            device="cpu",
            scaler=scaler,
        )
        assert history.epochs == [1, 2]
        assert all(np.isfinite(history.train_loss))


@requires_pyg
def test_cpu_training_is_reproducible_for_every_variant():
    """CPU training must be bitwise reproducible run to run.

    The GPU path is *not*: the GAT scatter reduction accumulates in a
    non-deterministic order on CUDA unless ``torch.use_deterministic_algorithms``
    is enabled, which is why ``run_gnn_experiment.py`` exposes ``--deterministic``
    and why the M4.3/M4.4 records note two differing graph_only runs. Pinning
    the CPU guarantee here keeps that distinction an explicit, tested property
    rather than a surprise.
    """
    dataset = build_dataset(linear_events(60))
    scaler = fitted_scaler(dataset)

    def run(variant: str) -> list[float]:
        import torch

        torch.manual_seed(0)
        model = build_model(GNNExperimentConfig(), variant=variant)
        history = train_model(
            model,
            dataset,
            config=GNNExperimentConfig(epochs=3),
            device="cpu",
            scaler=scaler,
        )
        return list(history.train_loss)

    for variant in MODEL_VARIANTS:
        assert run(variant) == run(variant), f"{variant} is not reproducible on CPU"


@requires_pyg
def test_scores_are_reproducible_for_a_fixed_seed():
    import torch

    dataset = build_dataset()
    scaler = fitted_scaler(dataset)

    def run() -> np.ndarray:
        torch.manual_seed(11)
        model = build_model(GNNExperimentConfig(), variant=MODEL_HYBRID)
        return predict_scores(model, dataset.batches, scaler=scaler)

    first, second = run(), run()
    assert np.array_equal(first, second)
    assert (first >= 0.0).all() and (first <= 1.0).all()


# ----------------------------------------------------------------------
# 6. Memory safety
# ----------------------------------------------------------------------


def test_behavioral_block_is_bounded_by_the_block_size():
    dataset = build_dataset(linear_events(60))
    for batch in dataset.batches:
        assert batch.behavioral.shape[0] <= 10
        assert batch.behavioral.nbytes <= 10 * 17 * 4


def test_dataset_bytes_account_for_the_behavioral_block():
    dataset = build_dataset()
    accounted = sum(
        b.behavioral.nbytes for b in dataset.batches
    )
    assert accounted > 0
    assert dataset.nbytes() > accounted
