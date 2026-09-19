"""Unit tests for the M4.2 PyTorch Geometric graph representation.

Every fixture here is a tiny deterministic in-memory object. Nothing in this
file reads ``auth.txt.gz``, ``redteam.txt.gz`` or any other real dataset, and
no accuracy claim is derived from anything built here.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest

from ml.graph.temporal_graph import TemporalGraph, host_node_id, user_node_id
from ml.models import graph_data
from ml.models.graph_data import (
    DEFAULT_MAX_SNAPSHOT_EVENTS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    NODE_TYPE_HOST,
    NODE_TYPE_USER,
    RELATION_IDENTITY,
    RELATION_MOVEMENT,
    GraphDataError,
    SnapshotBoundError,
    TemporalLeakageError,
    build_graph_snapshot,
    pyg_available,
    snapshot_from_temporal_graph,
    torch_available,
)
from ml.preprocessing.features import DEFAULT_RECENT_WINDOW_SECONDS
from ml.preprocessing.schema import CanonicalEvent

requires_torch = pytest.mark.skipif(
    not torch_available(), reason="PyTorch is not installed"
)
requires_pyg = pytest.mark.skipif(
    not pyg_available(), reason="PyTorch Geometric is not installed"
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


#: Three events exercising a new user, a re-used host and a host self-loop.
def reference_events() -> list[CanonicalEvent]:
    return [
        make_event("e1", 100.0, "U1", "H1", "H2", success=True),
        make_event("e2", 200.0, "U1", "H2", "H3", success=False),
        make_event("e3", 300.0, "U2", "H3", "H3", success=True),
    ]


def feature_column(snapshot, name: str) -> np.ndarray:
    return snapshot.x[:, NODE_FEATURE_NAMES.index(name)]


def edge_column(snapshot, name: str) -> np.ndarray:
    return snapshot.edge_attr[:, EDGE_FEATURE_NAMES.index(name)]


# ----------------------------------------------------------------------
# 1. Empty graph handling
# ----------------------------------------------------------------------


def test_empty_event_sequence_produces_well_formed_empty_tensors():
    snapshot = build_graph_snapshot([])

    assert snapshot.num_nodes == 0
    assert snapshot.num_edges == 0
    assert snapshot.num_events == 0
    assert snapshot.x.shape == (0, len(NODE_FEATURE_NAMES))
    assert snapshot.edge_index.shape == (2, 0)
    assert snapshot.edge_attr.shape == (0, len(EDGE_FEATURE_NAMES))
    assert snapshot.edge_time.shape == (0,)
    assert snapshot.window_start is None
    assert snapshot.window_end is None
    snapshot.validate()


def test_empty_snapshot_from_cutoff_before_every_event():
    snapshot = build_graph_snapshot(reference_events(), cutoff_timestamp=50.0)

    assert snapshot.num_events == 0
    assert snapshot.num_edges == 0
    assert snapshot.excluded_future_events == 3


# ----------------------------------------------------------------------
# 2. USER / HOST node encoding
# ----------------------------------------------------------------------


def test_user_and_host_nodes_are_namespaced_and_typed():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.node_ids == (
        user_node_id("U1"),
        host_node_id("H1"),
        host_node_id("H2"),
        host_node_id("H3"),
        user_node_id("U2"),
    )
    assert snapshot.node_type.tolist() == [
        NODE_TYPE_USER,
        NODE_TYPE_HOST,
        NODE_TYPE_HOST,
        NODE_TYPE_HOST,
        NODE_TYPE_USER,
    ]
    assert feature_column(snapshot, "is_user").tolist() == [1, 0, 0, 0, 1]
    assert feature_column(snapshot, "is_host").tolist() == [0, 1, 1, 1, 0]


def test_user_and_host_identifiers_never_collide():
    events = [make_event("e1", 10.0, user="X", source_host="X", destination_host="X")]
    snapshot = build_graph_snapshot(events)

    assert snapshot.num_nodes == 2
    assert snapshot.user_index("X") != snapshot.host_index("X")
    assert snapshot.node_type[snapshot.user_index("X")] == NODE_TYPE_USER
    assert snapshot.node_type[snapshot.host_index("X")] == NODE_TYPE_HOST


def test_unknown_node_lookup_raises_key_error():
    snapshot = build_graph_snapshot(reference_events())
    with pytest.raises(KeyError):
        snapshot.node_index("host:absent")


# ----------------------------------------------------------------------
# 3. Edge structure: USER -> HOST identity and HOST -> HOST movement
# ----------------------------------------------------------------------


def test_each_event_produces_one_identity_and_one_movement_edge():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.num_edges == 2 * snapshot.num_events
    assert int((snapshot.edge_type == RELATION_IDENTITY).sum()) == 3
    assert int((snapshot.edge_type == RELATION_MOVEMENT).sum()) == 3


def test_identity_edges_run_from_user_to_destination_host():
    snapshot = build_graph_snapshot(reference_events())
    identity = snapshot.edge_type == RELATION_IDENTITY
    sources = snapshot.edge_index[0][identity]
    destinations = snapshot.edge_index[1][identity]

    assert (snapshot.node_type[sources] == NODE_TYPE_USER).all()
    assert (snapshot.node_type[destinations] == NODE_TYPE_HOST).all()
    assert sources.tolist() == [
        snapshot.user_index("U1"),
        snapshot.user_index("U1"),
        snapshot.user_index("U2"),
    ]
    assert destinations.tolist() == [
        snapshot.host_index("H2"),
        snapshot.host_index("H3"),
        snapshot.host_index("H3"),
    ]


def test_movement_edges_run_from_source_host_to_destination_host():
    snapshot = build_graph_snapshot(reference_events())
    movement = snapshot.edge_type == RELATION_MOVEMENT
    sources = snapshot.edge_index[0][movement]
    destinations = snapshot.edge_index[1][movement]

    assert (snapshot.node_type[sources] == NODE_TYPE_HOST).all()
    assert (snapshot.node_type[destinations] == NODE_TYPE_HOST).all()
    assert sources.tolist() == [
        snapshot.host_index("H1"),
        snapshot.host_index("H2"),
        snapshot.host_index("H3"),
    ]
    assert destinations.tolist() == [
        snapshot.host_index("H2"),
        snapshot.host_index("H3"),
        snapshot.host_index("H3"),
    ]


def test_source_only_host_is_kept_as_a_node():
    """H1 is never a destination; it must still exist with in-degree 0."""
    snapshot = build_graph_snapshot(reference_events())
    index = snapshot.host_index("H1")

    assert feature_column(snapshot, "in_degree")[index] == 0
    assert feature_column(snapshot, "out_degree")[index] == 1


def test_edge_structure_matches_the_existing_temporal_graph_engine():
    events = reference_events()
    graph = TemporalGraph()
    graph.add_events(events)
    snapshot = build_graph_snapshot(events)

    assert snapshot.num_nodes == graph.number_of_nodes()
    assert snapshot.num_edges == graph.number_of_edges()
    assert set(snapshot.node_ids) == {node for node, _ in graph.iter_nodes()}


# ----------------------------------------------------------------------
# 4. Node features
# ----------------------------------------------------------------------


def test_node_activity_degree_and_outcome_counts():
    snapshot = build_graph_snapshot(reference_events())

    # Node order: user:U1, host:H1, host:H2, host:H3, user:U2
    assert feature_column(snapshot, "event_count").tolist() == [2, 1, 2, 2, 1]
    assert feature_column(snapshot, "out_degree").tolist() == [2, 1, 1, 1, 1]
    assert feature_column(snapshot, "in_degree").tolist() == [0, 0, 2, 4, 0]
    assert feature_column(snapshot, "success_count").tolist() == [1, 1, 1, 1, 1]
    assert feature_column(snapshot, "failure_count").tolist() == [1, 0, 1, 1, 0]


def test_self_referencing_event_counts_once_per_distinct_node():
    """H3 is both source and destination of e3; activity must not double count."""
    snapshot = build_graph_snapshot(reference_events())
    index = snapshot.host_index("H3")

    assert feature_column(snapshot, "event_count")[index] == 2
    assert (
        feature_column(snapshot, "success_count")[index]
        + feature_column(snapshot, "failure_count")[index]
        == feature_column(snapshot, "event_count")[index]
    )


def test_recent_activity_respects_the_recent_window():
    snapshot = build_graph_snapshot(
        reference_events(), recent_window_seconds=150.0
    )

    # Reference time is the last admitted timestamp (300), so only events with
    # timestamp > 150 are recent: e2 (200) and e3 (300).
    assert feature_column(snapshot, "recent_event_count").tolist() == [
        1,
        0,
        1,
        2,
        1,
    ]


def test_recent_activity_is_measured_back_from_the_cutoff():
    events = reference_events()
    snapshot = build_graph_snapshot(
        events, cutoff_timestamp=1000.0, recent_window_seconds=150.0
    )

    # Cutoff 1000 with a 150 s window leaves nothing recent at all.
    assert feature_column(snapshot, "recent_event_count").tolist() == [0] * 5
    assert feature_column(snapshot, "event_count").tolist() == [2, 1, 2, 2, 1]


def test_default_recent_window_matches_the_existing_feature_extractor():
    snapshot = build_graph_snapshot(reference_events())
    assert snapshot.recent_window_seconds == DEFAULT_RECENT_WINDOW_SECONDS


def test_degrees_agree_with_the_materialized_edge_index():
    snapshot = build_graph_snapshot(reference_events())
    out_degree = np.bincount(snapshot.edge_index[0], minlength=snapshot.num_nodes)
    in_degree = np.bincount(snapshot.edge_index[1], minlength=snapshot.num_nodes)

    assert feature_column(snapshot, "out_degree").tolist() == out_degree.tolist()
    assert feature_column(snapshot, "in_degree").tolist() == in_degree.tolist()


# ----------------------------------------------------------------------
# 5. Edge features
# ----------------------------------------------------------------------


def test_edge_features_carry_success_relation_and_event_type():
    snapshot = build_graph_snapshot(reference_events())

    # Two edges per event, in (identity, movement) order.
    assert edge_column(snapshot, "success").tolist() == [1, 1, 0, 0, 1, 1]
    assert edge_column(snapshot, "is_identity").tolist() == [1, 0, 1, 0, 1, 0]
    assert edge_column(snapshot, "is_movement").tolist() == [0, 1, 0, 1, 0, 1]
    assert edge_column(snapshot, "event_type_authentication").tolist() == [1] * 6


def test_self_loop_flag_marks_source_equal_to_destination():
    snapshot = build_graph_snapshot(reference_events())
    assert edge_column(snapshot, "self_loop").tolist() == [0, 0, 0, 0, 1, 1]


def test_edge_time_delta_is_seconds_since_the_window_start():
    snapshot = build_graph_snapshot(reference_events())
    assert edge_column(snapshot, "time_delta").tolist() == [
        0.0,
        0.0,
        100.0,
        100.0,
        200.0,
        200.0,
    ]


def test_relation_flags_are_mutually_exclusive():
    snapshot = build_graph_snapshot(reference_events())
    total = edge_column(snapshot, "is_identity") + edge_column(
        snapshot, "is_movement"
    )
    assert total.tolist() == [1.0] * snapshot.num_edges


def test_edges_reference_their_event_instead_of_duplicating_it():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.event_ids == ("e1", "e2", "e3")
    assert snapshot.edge_event_index.tolist() == [0, 0, 1, 1, 2, 2]
    assert np.array_equal(
        snapshot.event_timestamps[snapshot.edge_event_index], snapshot.edge_time
    )


# ----------------------------------------------------------------------
# 6. Timestamp preservation and temporal ordering
# ----------------------------------------------------------------------


def test_raw_timestamps_are_preserved_exactly():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.event_timestamps.dtype == np.float64
    assert snapshot.event_timestamps.tolist() == [100.0, 200.0, 300.0]
    assert snapshot.edge_time.tolist() == [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]
    assert snapshot.window_start == 100.0
    assert snapshot.window_end == 300.0


def test_large_lanl_scale_timestamps_survive_the_round_trip():
    timestamp = 5011199.0  # near the end of the real LANL timeline
    snapshot = build_graph_snapshot([make_event("e1", timestamp)])

    assert snapshot.event_timestamps[0] == timestamp
    assert snapshot.edge_time.tolist() == [timestamp, timestamp]


def test_edges_are_emitted_in_non_decreasing_time_order():
    events = [make_event(f"e{i}", float(i)) for i in range(20)]
    snapshot = build_graph_snapshot(events)

    assert (np.diff(snapshot.edge_time) >= 0).all()


def test_decreasing_input_timestamps_are_rejected():
    events = [make_event("e1", 200.0), make_event("e2", 100.0)]
    with pytest.raises(GraphDataError, match="non-decreasing"):
        build_graph_snapshot(events)


def test_same_timestamp_events_are_ordered_by_event_id():
    events = [
        make_event("b", 50.0, user="Ub"),
        make_event("a", 50.0, user="Ua"),
    ]
    snapshot = build_graph_snapshot(events)

    assert snapshot.event_ids == ("a", "b")


# ----------------------------------------------------------------------
# 7. Temporal rule: no future edges
# ----------------------------------------------------------------------


def test_cutoff_excludes_events_at_or_after_the_target_timestamp():
    snapshot = build_graph_snapshot(reference_events(), cutoff_timestamp=200.0)

    assert snapshot.event_ids == ("e1",)
    assert snapshot.excluded_future_events == 2
    assert (snapshot.edge_time < 200.0).all()


def test_future_events_change_nothing_before_the_cutoff():
    past = reference_events()
    future = past + [
        make_event("e4", 400.0, "U9", "H9", "H2"),
        make_event("e5", 500.0, "U1", "H2", "H9"),
    ]

    bounded = build_graph_snapshot(future, cutoff_timestamp=400.0)
    expected = build_graph_snapshot(past, cutoff_timestamp=400.0)

    assert bounded.node_ids == expected.node_ids
    assert np.array_equal(bounded.x, expected.x)
    assert np.array_equal(bounded.edge_index, expected.edge_index)
    assert np.array_equal(bounded.edge_attr, expected.edge_attr)
    assert bounded.excluded_future_events == 2


def test_no_future_node_can_enter_the_snapshot():
    events = reference_events() + [make_event("e4", 400.0, "UFUTURE", "H1", "H2")]
    snapshot = build_graph_snapshot(events, cutoff_timestamp=400.0)

    assert user_node_id("UFUTURE") not in snapshot.node_ids


def test_validate_rejects_a_hand_tampered_future_edge():
    snapshot = build_graph_snapshot(reference_events(), cutoff_timestamp=400.0)
    # Move the last event, and both of its edges, past the cutoff.
    snapshot.edge_time[-2:] = 500.0
    snapshot.event_timestamps[-1] = 500.0

    with pytest.raises(TemporalLeakageError, match="future"):
        snapshot.validate()


def test_validate_rejects_out_of_order_edge_times():
    snapshot = build_graph_snapshot(reference_events())
    snapshot.edge_time[0] = 999.0
    snapshot.event_timestamps[0] = 999.0

    with pytest.raises(TemporalLeakageError, match="non-decreasing"):
        snapshot.validate()


def test_snapshot_from_temporal_graph_applies_the_same_cutoff():
    graph = TemporalGraph()
    graph.add_events(reference_events())

    snapshot = snapshot_from_temporal_graph(graph, cutoff_timestamp=300.0)

    assert snapshot.event_ids == ("e1", "e2")
    assert (snapshot.edge_time < 300.0).all()


def test_snapshot_from_temporal_graph_without_cutoff_uses_every_event():
    graph = TemporalGraph()
    graph.add_events(reference_events())

    snapshot = snapshot_from_temporal_graph(graph)
    direct = build_graph_snapshot(reference_events())

    assert snapshot.event_ids == direct.event_ids
    assert np.array_equal(snapshot.x, direct.x)


def test_snapshot_from_temporal_graph_rejects_a_non_graph():
    with pytest.raises(GraphDataError, match="TemporalGraph"):
        snapshot_from_temporal_graph(reference_events())  # type: ignore[arg-type]


def executable_source(module) -> str:
    """Module source with comments and string literals removed."""
    raw = inspect.getsource(module)
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(raw).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept).lower()


def test_no_redteam_or_label_information_reaches_this_module():
    """Labels must never become graph features."""
    code = executable_source(graph_data)

    assert "redteam" not in code
    assert "label" not in code
    assert "ml.evaluation" not in code
    assert "ground_truth" not in code


def test_no_label_bearing_field_appears_in_the_feature_layouts():
    forbidden = ("label", "redteam", "malicious", "attack", "y")
    for name in NODE_FEATURE_NAMES + EDGE_FEATURE_NAMES:
        assert name not in forbidden


# ----------------------------------------------------------------------
# 8. Deterministic conversion
# ----------------------------------------------------------------------


def test_conversion_is_byte_identical_across_repeated_builds():
    first = build_graph_snapshot(reference_events())
    second = build_graph_snapshot(reference_events())

    assert first.node_ids == second.node_ids
    assert first.event_ids == second.event_ids
    for name in ("x", "edge_index", "edge_attr", "edge_type", "edge_time"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_tie_order_of_equal_timestamps_does_not_change_the_tensors():
    events = [
        make_event("a", 10.0, "Ua", "Ha", "Hb"),
        make_event("b", 10.0, "Ub", "Hb", "Hc"),
        make_event("c", 10.0, "Uc", "Hc", "Ha"),
    ]
    forward = build_graph_snapshot(events)
    reversed_ties = build_graph_snapshot(list(reversed(events)))

    assert forward.node_ids == reversed_ties.node_ids
    assert forward.event_ids == reversed_ties.event_ids
    assert np.array_equal(forward.edge_index, reversed_ties.edge_index)
    assert np.array_equal(forward.x, reversed_ties.x)


def test_generator_and_list_inputs_agree():
    from_list = build_graph_snapshot(reference_events())
    from_generator = build_graph_snapshot(event for event in reference_events())

    assert from_list.node_ids == from_generator.node_ids
    assert np.array_equal(from_list.edge_attr, from_generator.edge_attr)


# ----------------------------------------------------------------------
# 9. Tensor shapes, dtypes and numeric health
# ----------------------------------------------------------------------


def test_array_shapes_and_dtypes_follow_the_documented_contract():
    snapshot = build_graph_snapshot(reference_events())
    n, e, m = snapshot.num_nodes, snapshot.num_edges, snapshot.num_events

    assert (n, e, m) == (5, 6, 3)
    assert snapshot.x.shape == (n, 8) and snapshot.x.dtype == np.float32
    assert snapshot.node_type.shape == (n,) and snapshot.node_type.dtype == np.int8
    assert snapshot.edge_index.shape == (2, e)
    assert snapshot.edge_index.dtype == np.int64
    assert snapshot.edge_attr.shape == (e, 6)
    assert snapshot.edge_attr.dtype == np.float32
    assert snapshot.edge_type.shape == (e,) and snapshot.edge_type.dtype == np.int8
    assert snapshot.edge_time.shape == (e,) and snapshot.edge_time.dtype == np.float64
    assert snapshot.edge_event_index.shape == (e,)
    assert snapshot.event_timestamps.shape == (m,)


def test_feature_dimensions_match_the_declared_names():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.node_feature_dimension == len(NODE_FEATURE_NAMES) == 8
    assert snapshot.edge_feature_dimension == len(EDGE_FEATURE_NAMES) == 6


def test_no_nan_or_inf_in_any_numeric_array():
    snapshot = build_graph_snapshot(reference_events())

    for array in (
        snapshot.x,
        snapshot.edge_attr,
        snapshot.edge_time,
        snapshot.event_timestamps,
    ):
        assert np.isfinite(array).all()


def test_edge_index_stays_inside_the_node_range():
    snapshot = build_graph_snapshot(reference_events())

    assert snapshot.edge_index.min() >= 0
    assert snapshot.edge_index.max() < snapshot.num_nodes


def test_summary_reports_consistent_counts():
    summary = build_graph_snapshot(reference_events()).summary()

    assert summary["num_nodes"] == 5
    assert summary["num_user_nodes"] + summary["num_host_nodes"] == 5
    assert summary["num_identity_edges"] == summary["num_movement_edges"] == 3
    assert summary["num_edges"] == 6


def test_nbytes_is_positive_and_bounded_for_a_tiny_graph():
    snapshot = build_graph_snapshot(reference_events())
    assert 0 < snapshot.nbytes() < 10_000


# ----------------------------------------------------------------------
# 10. Input validation and the bounded-snapshot guard
# ----------------------------------------------------------------------


def test_duplicate_event_ids_are_rejected():
    events = [make_event("dup", 10.0), make_event("dup", 20.0)]
    with pytest.raises(GraphDataError, match="more than once"):
        build_graph_snapshot(events)


def test_non_canonical_input_is_rejected():
    with pytest.raises(GraphDataError, match="CanonicalEvent"):
        build_graph_snapshot([{"event_id": "e1"}])  # type: ignore[list-item]


def test_max_events_guard_stops_an_oversized_build():
    events = [make_event(f"e{i}", float(i)) for i in range(5)]
    with pytest.raises(SnapshotBoundError, match="exceed the bound"):
        build_graph_snapshot(events, max_events=3)


def test_default_guard_is_bounded_well_below_the_full_dataset():
    assert 0 < DEFAULT_MAX_SNAPSHOT_EVENTS <= 1_000_000


def test_invalid_configuration_is_rejected():
    with pytest.raises(GraphDataError):
        build_graph_snapshot([], recent_window_seconds=-1.0)
    with pytest.raises(GraphDataError):
        build_graph_snapshot([], max_events=0)


# ----------------------------------------------------------------------
# 11. Tensor conversion (skipped when the libraries are absent)
# ----------------------------------------------------------------------


def test_availability_probes_do_not_raise():
    assert isinstance(torch_available(), bool)
    assert isinstance(pyg_available(), bool)


def test_conversion_raises_a_clear_error_when_torch_is_missing():
    if torch_available():
        pytest.skip("PyTorch is installed")
    snapshot = build_graph_snapshot(reference_events())
    with pytest.raises(ImportError, match="PyTorch is not installed"):
        snapshot.to_torch()


@requires_torch
def test_cpu_torch_conversion_preserves_shapes_and_values():
    import torch

    snapshot = build_graph_snapshot(reference_events())
    tensors = snapshot.to_torch()

    assert tensors["x"].shape == (snapshot.num_nodes, 8)
    assert tensors["edge_index"].shape == (2, snapshot.num_edges)
    assert tensors["edge_index"].dtype == torch.int64
    assert tensors["edge_attr"].shape == (snapshot.num_edges, 6)
    assert tensors["edge_time"].dtype == torch.float64
    assert not torch.isnan(tensors["x"]).any()
    assert not torch.isinf(tensors["edge_attr"]).any()
    assert np.array_equal(tensors["x"].numpy(), snapshot.x)


@requires_pyg
def test_cpu_pyg_data_conversion():
    snapshot = build_graph_snapshot(reference_events())
    data = snapshot.to_pyg_data()

    assert data.num_nodes == snapshot.num_nodes
    assert data.num_edges == snapshot.num_edges
    assert data.x.shape == (snapshot.num_nodes, 8)
    assert data.edge_attr.shape == (snapshot.num_edges, 6)
    assert data.edge_time.shape == (snapshot.num_edges,)


@requires_pyg
def test_empty_snapshot_survives_pyg_conversion():
    data = build_graph_snapshot([]).to_pyg_data()
    assert data.num_nodes == 0
    assert data.edge_index.shape == (2, 0)
