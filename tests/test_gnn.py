"""Unit tests for the M4.3 temporal event GNN.

Every fixture is a tiny deterministic in-memory object. Nothing here reads
``auth.txt.gz`` or ``redteam.txt.gz``, no model trained here is evaluated for
accuracy, and no number produced by this file is a research result.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest

from ml.graph.temporal_graph import host_node_id, user_node_id
from ml.models import gnn
from ml.models.gnn import (
    MEMORY_SAFE_MAX_BLOCK_SIZE,
    MEMORY_SAFE_MAX_HISTORY_EVENTS,
    TARGET_FEATURE_NAMES,
    BatchedEventDataset,
    GNNError,
    GNNExperimentConfig,
    build_model,
    compute_pos_weight,
    count_parameters,
    estimate_device_bytes,
    forward_batch,
    iter_event_batches,
    predict_scores,
    resolve_device,
    train_model,
)
from ml.models.graph_data import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    pyg_available,
    torch_available,
)
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TimestampRange,
    build_label_index,
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
    """A deterministic stream: one event per second, rotating hosts and users."""
    events = []
    for i in range(count):
        events.append(
            make_event(
                f"e{i:03d}",
                start + i,
                user=f"U{i % 5}",
                source_host=f"H{i % 7}",
                destination_host=f"H{(i + 3) % 7}",
                success=(i % 4 != 0),
            )
        )
    return events


def label_keys(events, positions) -> frozenset:
    """Exact 4-field label keys for the events at ``positions``."""
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
    events=None,
    *,
    labels=(10, 30, 50),
    emit_from=None,
    config=None,
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
            config=config or GNNExperimentConfig(block_size=10, max_history_events=25),
        )
    )


# ----------------------------------------------------------------------
# 1. Configuration guards
# ----------------------------------------------------------------------


def test_memory_guards_reject_oversized_configurations():
    with pytest.raises(GNNError, match="block_size"):
        GNNExperimentConfig(block_size=MEMORY_SAFE_MAX_BLOCK_SIZE + 1)
    with pytest.raises(GNNError, match="max_history_events"):
        GNNExperimentConfig(
            max_history_events=MEMORY_SAFE_MAX_HISTORY_EVENTS + 1
        )
    with pytest.raises(GNNError, match="max_batches"):
        GNNExperimentConfig(max_batches=0)
    with pytest.raises(GNNError, match="epochs"):
        GNNExperimentConfig(epochs=0)


def test_default_configuration_is_bounded():
    config = GNNExperimentConfig()
    assert config.block_size <= MEMORY_SAFE_MAX_BLOCK_SIZE
    assert config.max_history_events <= MEMORY_SAFE_MAX_HISTORY_EVENTS
    assert config.device == "cpu"


def test_batch_guard_stops_an_unbounded_run():
    events = linear_events(60)
    with pytest.raises(GNNError, match="batch guard"):
        list(
            iter_event_batches(
                events,
                frozenset(),
                timestamp_range=TimestampRange(start=100.0, end=159.0),
                emit_from_timestamp=100.0,
                config=GNNExperimentConfig(block_size=5, max_batches=3),
            )
        )


def test_resolve_device_falls_back_to_cpu():
    assert resolve_device("cpu") == "cpu"
    resolved = resolve_device("cuda")
    assert resolved in ("cpu", "cuda")


# ----------------------------------------------------------------------
# 2. Tensor shapes
# ----------------------------------------------------------------------


def test_batch_tensor_shapes_are_internally_consistent():
    dataset = build_dataset()
    assert dataset.batches

    for batch in dataset.batches:
        n, e, b = batch.num_nodes, batch.num_edges, batch.num_targets
        assert batch.x.shape == (n, len(NODE_FEATURE_NAMES))
        assert batch.x.dtype == np.float32
        assert batch.edge_index.shape == (2, e)
        assert batch.edge_attr.shape == (e, len(EDGE_FEATURE_NAMES))
        assert batch.target_attr.shape == (b, len(TARGET_FEATURE_NAMES))
        assert batch.labels.shape == (b,)
        assert batch.timestamps.shape == (b,)
        assert batch.target_user_index.shape == (b,)
        assert e == 2 * batch.history_events
        batch.validate()


def test_every_target_index_points_at_a_real_node():
    dataset = build_dataset()
    for batch in dataset.batches:
        for index in (
            batch.target_user_index,
            batch.target_source_index,
            batch.target_dest_index,
        ):
            assert index.min() >= 0
            assert index.max() < batch.num_nodes


def test_target_endpoints_absent_from_history_get_zero_history_rows():
    events = linear_events(12)
    dataset = build_dataset(
        events, labels=(), config=GNNExperimentConfig(block_size=12, max_history_events=25)
    )
    first = dataset.batches[0]

    # The very first block has no history at all, so every endpoint is new.
    assert first.history_events == 0
    assert first.num_edges == 0
    counts = first.x[:, NODE_FEATURE_NAMES.index("event_count")]
    assert counts.tolist() == [0.0] * first.num_nodes
    types = first.x[:, NODE_FEATURE_NAMES.index("is_user")] + first.x[
        :, NODE_FEATURE_NAMES.index("is_host")
    ]
    assert types.tolist() == [1.0] * first.num_nodes


def test_all_events_are_covered_exactly_once():
    events = linear_events(60)
    dataset = build_dataset(events)
    total = sum(batch.num_targets for batch in dataset.batches)
    assert total == len(events)

    timestamps = np.concatenate([batch.timestamps for batch in dataset.batches])
    assert timestamps.tolist() == [event.timestamp for event in events]


def test_no_nan_or_inf_anywhere_in_a_batch():
    for batch in build_dataset().batches:
        for array in (batch.x, batch.edge_attr, batch.target_attr):
            assert np.isfinite(array).all()


# ----------------------------------------------------------------------
# 3. Temporal cutoff
# ----------------------------------------------------------------------


def test_history_graph_never_reaches_the_cutoff():
    dataset = build_dataset()
    for batch in dataset.batches:
        assert batch.timestamps.min() == batch.cutoff_timestamp
        # Every target is at or after the cutoff; no history edge may be.
        assert (batch.timestamps >= batch.cutoff_timestamp).all()


def test_history_grows_monotonically_with_time():
    dataset = build_dataset()
    history_sizes = [batch.history_events for batch in dataset.batches]
    assert history_sizes[0] == 0
    assert history_sizes == sorted(history_sizes)


def test_future_events_cannot_change_an_earlier_batch():
    events = linear_events(30)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)
    # The same timestamp range for both runs, so the split boundaries -- and
    # therefore the batch boundaries -- are identical and only the presence of
    # the later events differs.
    span = TimestampRange(start=events[0].timestamp, end=events[-1].timestamp)

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

    # The truncated stream stops inside the training region; the full stream
    # continues through validation and test.
    assert len(truncated.batches) == 2
    assert len(full.batches) > len(truncated.batches)
    for early, late in zip(truncated.batches, full.batches):
        assert early.cutoff_timestamp == late.cutoff_timestamp
        assert np.array_equal(early.x, late.x)
        assert np.array_equal(early.edge_index, late.edge_index)
        assert np.array_equal(early.edge_attr, late.edge_attr)
        assert np.array_equal(early.target_attr, late.target_attr)


def test_context_events_build_history_but_are_never_targets():
    events = linear_events(30)
    emit_from = events[10].timestamp
    dataset = build_dataset(events, labels=(), emit_from=emit_from)

    targets = sum(batch.num_targets for batch in dataset.batches)
    assert targets == 20
    assert dataset.batches[0].history_events == 10
    assert dataset.batches[0].cutoff_timestamp == emit_from


def test_seen_before_flags_reflect_history_only():
    events = [
        make_event("a", 10.0, "Uold", "Hold", "Hother"),
        make_event("b", 20.0, "Unew", "Hnew", "Hold"),
    ]
    dataset = build_dataset(
        events,
        labels=(),
        emit_from=20.0,
        config=GNNExperimentConfig(block_size=5, max_history_events=25),
    )
    batch = dataset.batches[0]
    attrs = batch.target_attr[0]

    assert attrs[TARGET_FEATURE_NAMES.index("user_seen_before")] == 0.0
    assert attrs[TARGET_FEATURE_NAMES.index("source_seen_before")] == 0.0
    assert attrs[TARGET_FEATURE_NAMES.index("destination_seen_before")] == 1.0


def test_out_of_order_events_are_rejected():
    events = [make_event("a", 200.0), make_event("b", 100.0)]
    with pytest.raises(GNNError, match="non-decreasing"):
        list(
            iter_event_batches(
                events,
                frozenset(),
                timestamp_range=TimestampRange(start=100.0, end=200.0),
                emit_from_timestamp=0.0,
            )
        )


# ----------------------------------------------------------------------
# 4. Chronological splits
# ----------------------------------------------------------------------


def test_no_batch_straddles_a_split_boundary():
    events = linear_events(60)
    dataset = build_dataset(events)
    splits = ChronologicalSplitConfig()
    start, end = events[0].timestamp, events[-1].timestamp

    for batch in dataset.batches:
        expected = {
            splits.split_for_timestamp(float(t), start, end)
            for t in batch.timestamps
        }
        assert expected == {batch.split}


def test_splits_are_chronological_and_ordered():
    dataset = build_dataset(linear_events(60))
    order = [batch.split for batch in dataset.batches]
    first_seen = {name: order.index(name) for name in set(order)}

    assert order == sorted(order, key=lambda name: first_seen[name])
    assert order[0] == SPLIT_TRAIN
    assert order[-1] == SPLIT_TEST
    assert SPLIT_VALIDATION in order


def test_counts_add_up_across_splits():
    dataset = build_dataset(linear_events(60))
    counts = dataset.counts()
    total = sum(counts[name]["events"] for name in counts)
    positives = sum(counts[name]["positive"] for name in counts)

    assert total == 60
    assert positives == 3
    for name in counts:
        assert (
            counts[name]["positive"] + counts[name]["negative"]
            == counts[name]["events"]
        )


def test_labels_helper_returns_split_labels_in_order():
    dataset = build_dataset(linear_events(60))
    train_labels = dataset.labels(SPLIT_TRAIN)
    assert train_labels.shape[0] == dataset.counts()[SPLIT_TRAIN]["events"]
    assert dataset.labels("train").sum() == dataset.counts()[SPLIT_TRAIN][
        "positive"
    ]


# ----------------------------------------------------------------------
# 5. Label separation
# ----------------------------------------------------------------------


def executable_source(module) -> str:
    raw = inspect.getsource(module)
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(raw).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept).lower()


def test_module_never_imports_redteam_ground_truth():
    code = executable_source(gnn)
    assert "redteam" not in code
    assert "ml.evaluation" not in code


def test_changing_labels_changes_nothing_but_the_label_vector():
    events = linear_events(60)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)
    without = build_dataset(events, labels=(), config=config)
    with_labels = build_dataset(events, labels=(10, 30, 50), config=config)

    assert without.labels(SPLIT_TRAIN).sum() == 0
    assert sum(b.positive_count() for b in with_labels.batches) == 3

    for plain, labeled in zip(without.batches, with_labels.batches):
        assert np.array_equal(plain.x, labeled.x)
        assert np.array_equal(plain.edge_index, labeled.edge_index)
        assert np.array_equal(plain.edge_attr, labeled.edge_attr)
        assert np.array_equal(plain.target_attr, labeled.target_attr)


def test_no_label_bearing_name_appears_in_any_feature_layout():
    forbidden = ("label", "redteam", "malicious", "attack", "y", "target")
    for name in NODE_FEATURE_NAMES + EDGE_FEATURE_NAMES + TARGET_FEATURE_NAMES:
        assert name not in forbidden


def test_label_index_uses_the_existing_exact_match_rule():
    """Labels come from the shared M3.2/M3.5 index, not a local rule."""
    source = executable_source(gnn)
    assert "label_for_event" in source
    assert build_label_index.__module__ == "ml.preprocessing.ml_dataset"


# ----------------------------------------------------------------------
# 6. Deterministic preprocessing
# ----------------------------------------------------------------------


def test_preprocessing_is_byte_identical_across_repeated_builds():
    first = build_dataset()
    second = build_dataset()

    assert len(first.batches) == len(second.batches)
    for a, b in zip(first.batches, second.batches):
        assert a.split == b.split
        assert a.cutoff_timestamp == b.cutoff_timestamp
        assert np.array_equal(a.x, b.x)
        assert np.array_equal(a.edge_index, b.edge_index)
        assert np.array_equal(a.edge_attr, b.edge_attr)
        assert np.array_equal(a.target_attr, b.target_attr)
        assert np.array_equal(a.labels, b.labels)


def test_generator_and_list_sources_agree():
    events = linear_events(30)
    config = GNNExperimentConfig(block_size=10, max_history_events=25)
    from_list = build_dataset(events, labels=(), config=config)
    from_generator = BatchedEventDataset.from_batches(
        iter_event_batches(
            (event for event in events),
            frozenset(),
            timestamp_range=TimestampRange(start=100.0, end=129.0),
            emit_from_timestamp=100.0,
            config=config,
        )
    )
    for a, b in zip(from_list.batches, from_generator.batches):
        assert np.array_equal(a.x, b.x)
        assert np.array_equal(a.target_attr, b.target_attr)


# ----------------------------------------------------------------------
# 7. Loss weighting and reporting helpers
# ----------------------------------------------------------------------


def test_pos_weight_matches_the_m40_convention():
    labels = np.array([0, 0, 0, 1], dtype=np.int8)
    assert compute_pos_weight(labels) == 3.0


def test_pos_weight_is_neutral_without_positives():
    assert compute_pos_weight(np.zeros(10, dtype=np.int8)) == 1.0


def test_dataset_summary_reports_memory_and_peaks():
    dataset = build_dataset()
    summary = dataset.summary()

    assert summary["batches"] == len(dataset.batches)
    assert summary["cpu_bytes"] > 0
    assert summary["peak_batch_edges"] >= 0
    assert set(summary["counts"]) == {SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST}


def test_device_byte_estimate_is_positive_and_finite():
    dataset = build_dataset()
    estimate = estimate_device_bytes(dataset, hidden_dim=32, heads=4)
    assert estimate >= 0


# ----------------------------------------------------------------------
# 8. Model: forward pass, shapes, parameter count
# ----------------------------------------------------------------------


@requires_pyg
def test_model_builds_with_a_small_parameter_count():
    model = build_model(GNNExperimentConfig())
    total = count_parameters(model)
    assert 0 < total < 100_000


@requires_pyg
def test_forward_pass_returns_one_logit_per_target_event():
    import torch

    dataset = build_dataset()
    model = build_model(GNNExperimentConfig())
    for batch in dataset.batches:
        logits = forward_batch(model, batch)
        assert logits.shape == (batch.num_targets,)
        assert torch.isfinite(logits).all()


@requires_pyg
def test_forward_pass_survives_an_empty_history_graph():
    import torch

    dataset = build_dataset(linear_events(8), labels=())
    first = dataset.batches[0]
    assert first.num_edges == 0

    logits = forward_batch(build_model(GNNExperimentConfig()), first)
    assert logits.shape == (first.num_targets,)
    assert torch.isfinite(logits).all()


@requires_pyg
def test_predict_scores_are_probabilities_in_chronological_order():
    dataset = build_dataset()
    model = build_model(GNNExperimentConfig())
    batches = dataset.split(SPLIT_TRAIN)
    scores = predict_scores(model, batches)

    assert scores.shape[0] == sum(b.num_targets for b in batches)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


@requires_pyg
def test_predict_scores_of_no_batches_is_empty():
    model = build_model(GNNExperimentConfig())
    assert predict_scores(model, []).shape == (0,)


@requires_pyg
def test_forward_pass_is_reproducible_for_a_fixed_seed():
    import torch

    dataset = build_dataset()
    torch.manual_seed(7)
    first = predict_scores(build_model(GNNExperimentConfig()), dataset.batches)
    torch.manual_seed(7)
    second = predict_scores(build_model(GNNExperimentConfig()), dataset.batches)

    assert np.allclose(first, second)


@requires_pyg
def test_training_runs_and_never_touches_the_test_split():
    dataset = build_dataset(linear_events(60))
    model = build_model(GNNExperimentConfig())
    source = inspect.getsource(train_model)

    history = train_model(
        model, dataset, config=GNNExperimentConfig(epochs=2), device="cpu"
    )

    assert history.epochs == [1, 2]
    assert len(history.train_loss) == 2
    assert all(np.isfinite(history.train_loss))
    assert history.seconds >= 0.0
    assert "SPLIT_TEST" not in source


@requires_pyg
def test_training_refuses_an_empty_training_split():
    empty = BatchedEventDataset(batches=[])
    with pytest.raises(GNNError, match="training split is empty"):
        train_model(build_model(), empty, config=GNNExperimentConfig(epochs=1))
