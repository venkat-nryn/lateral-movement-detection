"""Unit tests for the M2.0 canonical data contract.

All objects here are small in-memory software-test fixtures; they are
not research data and do not depend on any real dataset.
"""

import math

import pytest

from ml.graph.schema import (
    NODE_ENTITY_TYPES,
    EntityType,
    TemporalRelationship,
)
from ml.preprocessing.schema import (
    CANONICAL_EVENT_TYPES,
    CanonicalEvent,
    SchemaValidationError,
)


def make_event(**overrides):
    base = {
        "event_id": "evt-0001",
        "timestamp": 1_700_000_000.0,
        "user": "U1@test.local",
        "source_host": "H1@test.local",
        "destination_host": "H2@test.local",
        "event_type": "authentication",
        "success": True,
    }
    base.update(overrides)
    return CanonicalEvent(**base)


def make_relationship(**overrides):
    base = {
        "source": "H1",
        "destination": "H2",
        "timestamp": 1_700_000_000.0,
        "event_id": "evt-0001",
        "event_type": "authentication",
        "user": "U1",
        "success": True,
    }
    base.update(overrides)
    return TemporalRelationship(**base)


# 1. Valid event


def test_valid_canonical_event_is_accepted():
    event = make_event()
    assert event.event_id == "evt-0001"
    assert event.source_host == "H1@test.local"
    assert event.destination_host == "H2@test.local"
    assert event.event_type in CANONICAL_EVENT_TYPES
    assert event.success is True


def test_valid_event_round_trips_through_dict():
    event = make_event()
    restored = CanonicalEvent.from_dict(event.to_dict())
    assert restored == event


# 2. Missing event_id


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_missing_or_empty_event_id_rejected(bad_id):
    with pytest.raises(SchemaValidationError, match="event_id"):
        make_event(event_id=bad_id)


def test_from_dict_with_absent_event_id_key_rejected():
    record = make_event().to_dict()
    del record["event_id"]
    with pytest.raises(SchemaValidationError, match="missing required fields"):
        CanonicalEvent.from_dict(record)


# 3. Invalid timestamp


@pytest.mark.parametrize(
    "bad_ts", ["1700000000", None, -1.0, float("nan"), float("inf"), True]
)
def test_invalid_timestamp_rejected(bad_ts):
    with pytest.raises(SchemaValidationError, match="timestamp"):
        make_event(timestamp=bad_ts)


# 4. Missing source host


@pytest.mark.parametrize("bad_host", [None, "", "  "])
def test_missing_source_host_rejected(bad_host):
    with pytest.raises(SchemaValidationError, match="source_host"):
        make_event(source_host=bad_host)


# 5. Missing destination host


@pytest.mark.parametrize("bad_host", [None, "", "  "])
def test_missing_destination_host_rejected(bad_host):
    with pytest.raises(SchemaValidationError, match="destination_host"):
        make_event(destination_host=bad_host)


# 6. Invalid success value


@pytest.mark.parametrize("bad_success", ["true", 1, 0, None, 1.0])
def test_invalid_success_value_rejected(bad_success):
    with pytest.raises(SchemaValidationError, match="success"):
        make_event(success=bad_success)


def test_invalid_event_type_rejected():
    with pytest.raises(SchemaValidationError, match="event_type"):
        make_event(event_type="kerberos_ticket")


def test_malformed_record_unknown_field_rejected():
    record = make_event().to_dict()
    record["raw_column"] = "something"
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        CanonicalEvent.from_dict(record)


def test_events_are_immutable():
    event = make_event()
    with pytest.raises(AttributeError):
        event.user = "other"


# 7. Valid temporal relationship


def test_valid_temporal_relationship_is_accepted():
    rel = make_relationship()
    assert rel.source == "H1"
    assert rel.destination == "H2"
    assert rel.identity_key() == ("H1", "H2", 1_700_000_000.0, "evt-0001")


def test_relationship_from_canonical_event():
    event = make_event()
    rel = TemporalRelationship.from_event(event)
    assert rel.source == event.source_host
    assert rel.destination == event.destination_host
    assert rel.timestamp == event.timestamp
    assert rel.event_id == event.event_id
    assert rel.user == event.user
    assert rel.success is event.success


def test_invalid_temporal_relationship_rejected():
    with pytest.raises(SchemaValidationError, match="temporal relationship"):
        make_relationship(destination="")


# 8. Multiple events between same hosts at different timestamps


def test_multiple_events_between_same_hosts_at_different_timestamps():
    rels = [
        TemporalRelationship.from_event(make_event(timestamp=1_700_000_000.0)),
        TemporalRelationship.from_event(make_event(timestamp=1_700_000_600.0)),
    ]
    assert len(rels) == 2
    assert rels[0].endpoint_key() == rels[1].endpoint_key()
    assert len({r.identity_key() for r in rels}) == 2


def test_entity_types_are_limited_to_user_and_host():
    assert {et.name for et in EntityType} == {"USER", "HOST"}
    assert NODE_ENTITY_TYPES == frozenset({EntityType.USER, EntityType.HOST})


def test_nan_timestamp_message_is_explicit():
    with pytest.raises(SchemaValidationError) as excinfo:
        make_event(timestamp=math.nan)
    assert "timestamp" in str(excinfo.value)
