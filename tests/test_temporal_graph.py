"""Unit tests for the M2.1 temporal graph core.

All objects are tiny deterministic in-memory software fixtures; they
are not research data and do not depend on any real dataset.
"""

import math

import pytest

from ml.graph.temporal_graph import (
    DuplicateEventError,
    NodeEventRecord,
    TemporalGraph,
    UnknownEventError,
    UnknownNodeError,
    host_node_id,
    user_node_id,
)
from ml.preprocessing.schema import CanonicalEvent


def make_event(
    event_id="evt-1",
    timestamp=3600.0,
    user="UserA",
    source_host="HostA",
    destination_host="HostB",
    success=True,
):
    return CanonicalEvent(
        event_id=event_id,
        timestamp=timestamp,
        user=user,
        source_host=source_host,
        destination_host=destination_host,
        event_type="authentication",
        success=success,
    )


# 1. Empty graph


def test_empty_graph_has_no_nodes_edges_or_events():
    tg = TemporalGraph()
    assert tg.number_of_nodes() == 0
    assert tg.number_of_edges() == 0
    assert tg.number_of_events() == 0
    assert tg.get_events_between(0.0, math.inf) == []


# 2. Add one event


def test_add_single_event_populates_counts_and_retrieval():
    tg = TemporalGraph()
    tg.add_event(make_event())
    assert tg.number_of_events() == 1
    assert tg.number_of_edges() == 2  # user→dest + source→dest
    assert tg.number_of_nodes() == 3  # one user + two hosts
    event = tg.get_event("evt-1")
    assert event.destination_host == "HostB"


# 3. Add multiple events


def test_add_multiple_events_via_batch_returns_count():
    tg = TemporalGraph()
    count = tg.add_events(
        [make_event(event_id=f"e{i}", timestamp=100.0 + i) for i in range(5)]
    )
    assert count == 5
    assert tg.number_of_events() == 5
    assert tg.number_of_edges() == 10  # 2 edges per event (user→dest + source→dest)


# 4. Correct node creation


def test_correct_node_creation_for_user_and_hosts():
    tg = TemporalGraph()
    tg.add_event(make_event())
    assert tg.get_node(user_node_id("UserA")) == {
        "entity_type": "user",
        "identifier": "UserA",
    }
    assert tg.get_node(host_node_id("HostA"))["entity_type"] == "host"
    assert tg.get_node(host_node_id("HostB"))["entity_type"] == "host"
    assert tg.number_of_nodes() == 3


# 5. User/host namespace separation


def test_user_and_host_namespaces_do_not_collide():
    tg = TemporalGraph()
    tg.add_event(make_event(user="shared-id", source_host="shared-id"))
    tg.add_event(
        make_event(event_id="evt-2", timestamp=3660.0, destination_host="shared-id")
    )
    user_node = user_node_id("shared-id")
    host_node = host_node_id("shared-id")
    assert user_node != host_node
    assert tg.get_node(user_node)["entity_type"] == "user"
    assert tg.get_node(host_node)["entity_type"] == "host"
    # distinct namespaces: shared-id exists once as user and once as host
    node_ids = {
        user_node,
        host_node,
        host_node_id("HostB"),
        user_node_id("UserA"),
        host_node_id("HostA"),
    }
    assert tg.number_of_nodes() == len(node_ids) == 5


# 6. Correct edge creation


def test_edge_creation_with_complete_context():
    tg = TemporalGraph()
    event = make_event(success=False)
    tg.add_event(event)
    edges = list(tg.iter_edges())
    assert len(edges) == 2  # user→dest + source→dest
    
    # Check identity edge: user → destination
    identity_edges = [e for e in edges if e[0] == user_node_id("UserA")]
    assert len(identity_edges) == 1
    u, v, key, attrs = identity_edges[0]
    assert u == user_node_id("UserA")
    assert v == host_node_id("HostB")
    assert key == "evt-1"
    assert attrs == {
        "event_id": "evt-1",
        "timestamp": 3600.0,
        "event_type": "authentication",
        "user": "UserA",
        "source_host": "HostA",
        "destination_host": "HostB",
        "success": False,
    }
    
    # Check movement edge: source → destination
    movement_edges = [e for e in edges if e[0] == host_node_id("HostA")]
    assert len(movement_edges) == 1
    u, v, key, attrs = movement_edges[0]
    assert u == host_node_id("HostA")
    assert v == host_node_id("HostB")
    assert key == "evt-1"
    assert attrs == {
        "event_id": "evt-1",
        "timestamp": 3600.0,
        "event_type": "authentication",
        "user": "UserA",
        "source_host": "HostA",
        "destination_host": "HostB",
        "success": False,
    }


# 7. Event retrieval by event_id


def test_get_unknown_event_raises_explicitly():
    tg = TemporalGraph()
    tg.add_event(make_event())
    with pytest.raises(UnknownEventError):
        tg.get_event("does-not-exist")


# 8. Node event history


def test_get_node_events_across_roles():
    tg = TemporalGraph()
    tg.add_event(make_event())  # HostA as source
    tg.add_event(
        make_event(
            event_id="evt-2", timestamp=3700.0, source_host="HostZ",
            destination_host="HostA",
        )
    )  # HostA as destination
    host_a = host_node_id("HostA")
    events = tg.get_node_events(host_a)
    assert [e.event_id for e in events] == ["evt-1", "evt-2"]
    records = tg.get_event_history(host_a)
    assert all(isinstance(r, NodeEventRecord) for r in records)
    assert records[0].roles == frozenset({"source_host"})
    assert records[1].roles == frozenset({"destination_host"})


def test_history_roles_for_multi_role_node():
    tg = TemporalGraph()
    tg.add_event(
        make_event(source_host="HostZ", destination_host="HostA")
    )  # HostA only as destination
    tg.add_event(
        make_event(event_id="evt-2", timestamp=3800.0, source_host="HostA")
    )  # HostA only as source
    record_by_id = {
        r.event.event_id: r for r in tg.get_event_history(host_node_id("HostA"))
    }
    assert record_by_id["evt-1"].roles == frozenset({"destination_host"})
    assert record_by_id["evt-2"].roles == frozenset({"source_host"})


# 9. Chronological ordering


def test_chronological_ordering_regardless_of_insertion_order():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="e3", timestamp=300.0))
    tg.add_event(make_event(event_id="e1", timestamp=100.0))
    tg.add_event(make_event(event_id="e2", timestamp=200.0))
    assert [e.event_id for e in tg.get_events_between(0, math.inf)] == [
        "e1",
        "e2",
        "e3",
    ]
    history = tg.get_event_history(user_node_id("UserA"))
    assert [r.event.timestamp for r in history] == [100.0, 200.0, 300.0]


# 10. Same timestamp deterministic ordering


def test_same_timestamp_ordered_by_event_id():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="b-second", timestamp=500.0))
    tg.add_event(make_event(event_id="a-first", timestamp=500.0))
    tg.add_event(make_event(event_id="c-third", timestamp=500.0))
    assert [e.event_id for e in tg.get_events_between(0, 1000)] == [
        "a-first",
        "b-second",
        "c-third",
    ]


# 11. Multiple events between same entities


def test_repeated_events_between_same_entities_preserved():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="auth-1", timestamp=3600.0))  # 10:00
    tg.add_event(make_event(event_id="auth-2", timestamp=3660.0))  # 10:05
    assert tg.number_of_edges() == 4  # 2 edges per event
    assert tg.number_of_events() == 2
    keys = {key for _, _, key, _ in tg.iter_edges()}
    assert keys == {"auth-1", "auth-2"}
    assert len(tg.get_node_events(user_node_id("UserA"))) == 2


# 12. Time-window query


def test_time_window_bounds_are_inclusive():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="low", timestamp=100.0))
    tg.add_event(make_event(event_id="mid", timestamp=150.0))
    tg.add_event(make_event(event_id="high", timestamp=200.0))
    assert [e.event_id for e in tg.get_events_between(100.0, 200.0)] == [
        "low",
        "mid",
        "high",
    ]
    assert [e.event_id for e in tg.get_events_between(100.001, 199.999)] == [
        "mid"
    ]
    assert [e.event_id for e in tg.get_events_between(150.0, 150.0)] == ["mid"]


# 13. Temporal neighbor query


def test_temporal_neighbors_within_window():
    tg = TemporalGraph()
    tg.add_event(make_event(timestamp=100.0))  # UserA: HostA -> HostB
    tg.add_event(make_event(event_id="e2", timestamp=200.0))
    tg.add_event(
        make_event(event_id="old", timestamp=50.0, destination_host="HostC")
    )
    user_node = user_node_id("UserA")
    # window covering only the first two events
    assert set(tg.get_temporal_neighbors(user_node, 90.0, 210.0)) == {
        host_node_id("HostA"),
        host_node_id("HostB"),
    }
    # narrow window excludes everything except e2's endpoints
    assert set(tg.get_temporal_neighbors(user_node, 190.0, 210.0)) == {
        host_node_id("HostA"),
        host_node_id("HostB"),
    }
    # source host co-occurs with users/destinations despite no edges;
    # event "old" also links HostA to HostC
    src = host_node_id("HostA")
    assert set(tg.get_temporal_neighbors(src, 0.0, 1000.0)) == {
        user_node_id("UserA"),
        host_node_id("HostB"),
        host_node_id("HostC"),
    }


def test_self_loop_event_indexed_once_per_node():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="loop", source_host="HostA",
                            destination_host="HostA"))
    assert tg.number_of_events() == 1
    assert tg.number_of_edges() == 2  # user→HostA + HostA→HostA (self-loop)
    history = tg.get_event_history(host_node_id("HostA"))
    assert [r.event.event_id for r in history] == ["loop"]
    assert history[0].roles == frozenset({"source_host", "destination_host"})


def test_get_neighbors_reflects_edge_structure_only():
    tg = TemporalGraph()
    tg.add_event(make_event())
    user_node = user_node_id("UserA")
    assert tg.get_neighbors(user_node) == [host_node_id("HostB")]
    # source host now has destination host as neighbour (via movement edge)
    assert tg.get_neighbors(host_node_id("HostA")) == [host_node_id("HostB")]


# 14. Temporal subgraph


def test_subgraph_is_independent_windowed_copy():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="e1", timestamp=100.0))
    tg.add_event(make_event(event_id="e2", timestamp=200.0))
    sub = tg.get_subgraph(50.0, 150.0)
    assert sub.number_of_events() == 1
    assert sub.number_of_edges() == 2  # 2 edges per event (user→dest + source→dest)
    assert sub.has_event("e1")
    assert not sub.has_event("e2")
    # independence: later mutation of original leaves subgraph intact
    tg.add_event(make_event(event_id="e3", timestamp=120.0))
    assert sub.number_of_events() == 1
    assert tg.number_of_events() == 3


# 15. Invalid event rejection


class FakeEvent:
    event_id = "fake"


@pytest.mark.parametrize(
    "bad_input",
    [None, {"event_id": "x"}, FakeEvent(), "evt-1", 42],
)
def test_invalid_event_types_rejected(bad_input):
    tg = TemporalGraph()
    with pytest.raises(TypeError, match="CanonicalEvent"):
        tg.add_event(bad_input)


def test_batch_rejects_invalid_member_atomically():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="existing"))
    with pytest.raises(TypeError):
        tg.add_events([make_event(event_id="new"), FakeEvent()])
    assert not tg.has_event("new")
    assert tg.number_of_events() == 1


# 16. Duplicate event_id behavior


def test_duplicate_event_id_rejected_and_original_preserved():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="dup"))
    with pytest.raises(DuplicateEventError, match="dup"):
        tg.add_event(make_event(event_id="dup", timestamp=9999.0))
    assert tg.get_event("dup").timestamp == 3600.0
    assert tg.number_of_events() == 1


def test_duplicate_within_batch_is_atomic():
    tg = TemporalGraph()
    with pytest.raises(DuplicateEventError):
        tg.add_events([make_event(event_id="x"), make_event(event_id="x")])
    assert tg.number_of_events() == 0


# 17. Successful and failed authentication preserved


def test_failed_authentication_preserved_distinct_from_success():
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="ok", timestamp=10.0, success=True))
    tg.add_event(make_event(event_id="fail", timestamp=20.0, success=False))
    ok = tg.get_event("ok")
    failed = tg.get_event("fail")
    assert ok.success is True and failed.success is False
    attrs_by_key = {key: attrs for _, _, key, attrs in tg.iter_edges()}
    assert attrs_by_key["ok"]["success"] is True
    assert attrs_by_key["fail"]["success"] is False


# 18. Consistency after multiple insertions


def test_graph_consistent_after_many_insertions():
    tg = TemporalGraph()
    events = [
        make_event(
            event_id=f"e{i:03d}",
            timestamp=float(i),
            user=f"U{i % 4}",
            source_host=f"H{i % 3}",
            destination_host=f"H{(i % 3) + 1}",
        )
        for i in range(60)
    ]
    assert tg.add_events(events) == 60
    assert tg.number_of_events() == 60
    assert tg.number_of_edges() == 120  # 2 edges per event (user→dest + source→dest)
    expected_hosts = {f"H{i}" for i in range(4)}
    expected_users = {f"U{i}" for i in range(4)}
    assert tg.number_of_nodes() == len(expected_users) + len(expected_hosts)

    ordered = tg.get_events_between(-math.inf, math.inf)
    assert [e.event_id for e in ordered] == [
        e.event_id for e in sorted(events, key=lambda e: (e.timestamp, e.event_id))
    ]

    stamps = [e.timestamp for e in ordered]
    assert stamps == sorted(stamps)
    for node in [user_node_id(u) for u in expected_users] + [
        host_node_id(h) for h in expected_hosts
    ]:
        history = tg.get_node_events(node)
        node_stamps = [e.timestamp for e in history]
        assert node_stamps == sorted(node_stamps)
        assert len(history) > 0


def test_unknown_node_queries_raise_explicitly():
    tg = TemporalGraph()
    tg.add_event(make_event())
    with pytest.raises(UnknownNodeError):
        tg.get_node("user:nobody")
    with pytest.raises(UnknownNodeError):
        tg.get_node_events("host:nobody")
    with pytest.raises(UnknownNodeError):
        tg.get_event_history("host:nobody")
    with pytest.raises(UnknownNodeError):
        tg.get_neighbors("host:nobody")


# 19. Host-to-host movement edges (M3.1.1)


def test_host_to_host_movement_edge_created():
    """Verify that each event creates a source → destination host edge."""
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="evt-1"))
    edges = list(tg.iter_edges())
    assert len(edges) == 2
    
    # Find movement edge (source → destination hosts)
    movement_edges = [e for e in edges if e[0] == host_node_id("HostA")]
    assert len(movement_edges) == 1
    u, v, key, attrs = movement_edges[0]
    assert u == host_node_id("HostA")
    assert v == host_node_id("HostB")
    assert key == "evt-1"
    assert attrs["source_host"] == "HostA"
    assert attrs["destination_host"] == "HostB"


def test_source_host_exists_as_node():
    """Verify source_host is a first-class graph node."""
    tg = TemporalGraph()
    tg.add_event(make_event(source_host="SourceX"))
    source_node = host_node_id("SourceX")
    assert tg.has_node(source_node)
    node_attrs = tg.get_node(source_node)
    assert node_attrs["entity_type"] == "host"
    assert node_attrs["identifier"] == "SourceX"


def test_destination_host_exists_as_node():
    """Verify destination_host is a first-class graph node."""
    tg = TemporalGraph()
    tg.add_event(make_event(destination_host="DestY"))
    dest_node = host_node_id("DestY")
    assert tg.has_node(dest_node)
    node_attrs = tg.get_node(dest_node)
    assert node_attrs["entity_type"] == "host"
    assert node_attrs["identifier"] == "DestY"


def test_identity_relationship_preserved():
    """Verify user → destination_host edges still exist."""
    tg = TemporalGraph()
    tg.add_event(make_event(event_id="evt-1"))
    edges = list(tg.iter_edges())
    
    identity_edges = [e for e in edges if e[0] == user_node_id("UserA")]
    assert len(identity_edges) == 1
    u, v, key, attrs = identity_edges[0]
    assert u == user_node_id("UserA")
    assert v == host_node_id("HostB")
    assert key == "evt-1"


def test_event_metadata_on_host_movement_edge():
    """Verify all event metadata is preserved on host-to-host edges."""
    tg = TemporalGraph()
    event = make_event(
        event_id="evt-123",
        timestamp=999.5,
        user="Alice",
        source_host="HostX",
        destination_host="HostY",
        success=False,
    )
    tg.add_event(event)
    
    edges_by_key = {key: attrs for _, _, key, attrs in tg.iter_edges()}
    attrs = edges_by_key["evt-123"]
    
    assert attrs["event_id"] == "evt-123"
    assert attrs["timestamp"] == 999.5
    assert attrs["event_type"] == "authentication"
    assert attrs["user"] == "Alice"
    assert attrs["source_host"] == "HostX"
    assert attrs["destination_host"] == "HostY"
    assert attrs["success"] is False


def test_self_loop_host_movement_no_duplicate_indexing():
    """Verify self-loop (source == destination) creates one movement edge."""
    tg = TemporalGraph()
    tg.add_event(make_event(
        event_id="loop",
        source_host="HostA",
        destination_host="HostA",
    ))
    
    # Should have 2 edges: user→HostA and HostA→HostA (self-loop)
    assert tg.number_of_edges() == 2
    
    # Host should be indexed once despite being both source and destination
    host_events = tg.get_node_events(host_node_id("HostA"))
    assert len(host_events) == 1
    assert host_events[0].event_id == "loop"


def test_multiple_host_to_host_edges_remain_distinct():
    """Verify multiple events between same hosts create distinct edges."""
    tg = TemporalGraph()
    tg.add_event(make_event(
        event_id="evt-1",
        timestamp=100.0,
        source_host="HostX",
        destination_host="HostY",
    ))
    tg.add_event(make_event(
        event_id="evt-2",
        timestamp=101.0,
        source_host="HostX",
        destination_host="HostY",
    ))
    
    # Should have 4 edges total (2 per event: user→dest + source→dest)
    assert tg.number_of_edges() == 4
    
    # Both movement edges should exist
    movement_edges = [
        e for e in tg.iter_edges() if e[0] == host_node_id("HostX")
    ]
    assert len(movement_edges) == 2
    assert {e[2] for e in movement_edges} == {"evt-1", "evt-2"}


def test_graph_queries_after_host_movement_edges():
    """Verify graph queries work correctly with host-to-host edges."""
    tg = TemporalGraph()
    tg.add_event(make_event(
        event_id="evt-1",
        timestamp=100.0,
        source_host="HostA",
        destination_host="HostB",
    ))
    tg.add_event(make_event(
        event_id="evt-2",
        timestamp=101.0,
        source_host="HostB",
        destination_host="HostC",
    ))
    
    # HostA → HostB → HostC chain (plus UserA → HostB, UserA → HostC)
    assert tg.get_neighbors(host_node_id("HostA")) == [host_node_id("HostB")]
    # HostB has predecessors (HostA, UserA) and successors (HostC)
    assert set(tg.get_neighbors(host_node_id("HostB"))) == {
        host_node_id("HostA"),
        user_node_id("UserA"),
        host_node_id("HostC"),
    }
    # HostC has predecessors (HostB, UserA) - note both events use UserA
    assert set(tg.get_neighbors(host_node_id("HostC"))) == {
        host_node_id("HostB"),
        user_node_id("UserA"),
    }
    
    # Time window query should find both events
    events = tg.get_events_between(0.0, 200.0)
    assert len(events) == 2
    assert {e.event_id for e in events} == {"evt-1", "evt-2"}
