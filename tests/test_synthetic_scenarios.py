"""Unit tests for M2.2 deterministic synthetic scenario generators.

These tests use only the tiny deterministic software fixtures produced
by ``ml.graph.synthetic_scenarios``; they are not research data.
"""

import math

import pytest

from ml.graph.synthetic_scenarios import (
    create_benign_scenario,
    create_mixed_scenario,
    create_multihop_scenario,
)
from ml.graph.temporal_graph import TemporalGraph
from ml.preprocessing.schema import CanonicalEvent

GENERATORS = [
    ("benign", create_benign_scenario),
    ("multihop", create_multihop_scenario),
    ("mixed", create_mixed_scenario),
]

EXPECTED_SIZES = {"benign": 3, "multihop": 4, "mixed": 9}


# 1. Scenario generation succeeds


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_scenario_generation_succeeds(name, generator):
    events = generator()
    assert len(events) == EXPECTED_SIZES[name]
    assert all(isinstance(e, CanonicalEvent) for e in events)


# 2. Returned objects are CanonicalEvent


def test_all_events_are_canonical_event_instances():
    for _, generator in GENERATORS:
        assert all(
            isinstance(e, CanonicalEvent) and type(e) is CanonicalEvent
            for e in generator()
        )


# 3. Deterministic repeated generation


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_repeated_generation_is_identical(name, generator):
    first = generator()
    second = generator()
    assert first == second
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


# 4. Event IDs are unique


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_event_ids_are_unique_within_scenario(name, generator):
    events = generator()
    ids = [e.event_id for e in events]
    assert len(ids) == len(set(ids))


def test_event_ids_unique_across_scenarios():
    all_ids = [
        e.event_id
        for _, generator in GENERATORS
        for e in generator()
    ]
    assert len(all_ids) == len(set(all_ids))


# 5. Timestamps are valid


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_timestamps_are_valid(name, generator):
    for event in generator():
        assert isinstance(event.timestamp, (int, float))
        assert not isinstance(event.timestamp, bool)
        assert math.isfinite(event.timestamp)
        assert event.timestamp >= 0


# 6. Multi-hop timestamps are ordered


def test_multihop_timestamps_strictly_increasing():
    stamps = [e.timestamp for e in create_multihop_scenario()]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))


def test_multihop_chain_preserved_in_source_context():
    events = create_multihop_scenario()
    assert {e.user for e in events} == {"User_A"}
    destinations = [e.destination_host for e in events]
    sources = [e.source_host for e in events]
    # each hop departs from the previous destination
    assert sources[1:] == destinations[:-1]


# 7. Multiple hosts exist


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_multiple_hosts_exist(name, generator):
    hosts = {e.source_host for e in generator()} | {
        e.destination_host for e in generator()
    }
    assert len(hosts) >= 2


# 8. Multiple users exist where appropriate


def test_benign_and_mixed_have_multiple_users():
    assert len({e.user for e in create_benign_scenario()}) >= 3
    assert len({e.user for e in create_mixed_scenario()}) >= 3


# 9. Successful and failed events preserved


def test_mixed_scenario_contains_successes_and_failures():
    successes = {e.success for e in create_mixed_scenario()}
    assert successes == {True, False}


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_success_values_are_strict_booleans(name, generator):
    assert all(isinstance(e.success, bool) for e in generator())


# 10 + 11. Insertion into TemporalGraph and count agreement


@pytest.mark.parametrize("name,generator", GENERATORS)
def test_scenario_inserts_into_temporal_graph_with_matching_count(name, generator):
    events = generator()
    tg = TemporalGraph()
    tg.add_events(events)
    assert tg.number_of_events() == EXPECTED_SIZES[name] == len(events)


def test_mixed_scenario_graph_structure():
    tg = TemporalGraph()
    tg.add_events(create_mixed_scenario())
    # repeated authentication must survive as separate edges
    # 9 events * 2 edges per event (user→host + host→host) = 18 edges
    assert tg.number_of_events() == 9
    assert tg.number_of_edges() == 18
    window = tg.get_events_between(1_000_040.0, 1_000_080.0)
    assert [e.event_id for e in window] == [
        "mix-0000040.0",
        "mix-0000080.0",
    ]


def test_mixed_scenario_failed_event_queryable():
    tg = TemporalGraph()
    tg.add_events(create_mixed_scenario())
    failed = [
        e for e in tg.get_events_between(0, math.inf) if e.success is False
    ]
    assert len(failed) == 1
    assert failed[0].user == "User_D"
