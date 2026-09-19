"""Unit tests for streaming temporal/graph feature extraction (M3.4)."""

from __future__ import annotations

from ml.preprocessing.features import (
    DEFAULT_RECENT_WINDOW_SECONDS,
    NO_PRIOR_TIME_DELTA,
    EventFeatures,
    TemporalGraphFeatureExtractor,
)
from ml.preprocessing.schema import CanonicalEvent


def make_event(
    event_id: str = "evt-1",
    timestamp: float = 1000.0,
    user: str = "UserA",
    source_host: str = "HostA",
    destination_host: str = "HostB",
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


def test_first_event_uses_zero_initial_history() -> None:
    extractor = TemporalGraphFeatureExtractor()

    features = extractor.process_event(
        make_event(source_host="HostA", destination_host="HostA", success=False)
    )

    assert isinstance(features, EventFeatures)
    assert features.source_equals_destination == 1
    assert features.user_destination_seen == 0
    assert features.source_destination_seen == 0
    assert features.user_destination_count == 0
    assert features.source_destination_count == 0
    assert features.user_unique_destination_count == 0
    assert features.source_unique_destination_count == 0
    assert features.destination_unique_source_count == 0
    assert features.user_recent_event_count == 0
    assert features.user_recent_unique_destination_count == 0
    assert features.time_since_user_previous_event == NO_PRIOR_TIME_DELTA
    assert features.time_since_user_destination_event == NO_PRIOR_TIME_DELTA
    assert features.time_since_source_destination_event == NO_PRIOR_TIME_DELTA
    assert features.failed_authentication == 1
    assert features.user_recent_failure_count == 0
    assert features.destination_recent_event_count == 0
    assert features.movement_edge == 0
    assert len(features.to_vector()) == extractor.feature_dimension() == 17


def test_repeated_user_destination_interaction() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(make_event(event_id="evt-1", timestamp=1000.0))

    features = extractor.process_event(make_event(event_id="evt-2", timestamp=1001.0))

    assert features.user_destination_seen == 1
    assert features.user_destination_count == 1
    assert features.user_unique_destination_count == 1


def test_new_destination_for_existing_user_updates_unique_counts() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, destination_host="HostB")
    )

    features = extractor.process_event(
        make_event(event_id="evt-2", timestamp=1001.0, destination_host="HostC")
    )

    assert features.user_destination_seen == 0
    assert features.user_unique_destination_count == 1
    assert features.user_recent_event_count == 1
    assert features.user_recent_unique_destination_count == 1


def test_new_source_destination_pair() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(
            event_id="evt-1",
            timestamp=1000.0,
            source_host="HostA",
            destination_host="HostB",
        )
    )

    features = extractor.process_event(
        make_event(
            event_id="evt-2",
            timestamp=1001.0,
            source_host="HostA",
            destination_host="HostC",
        )
    )

    assert features.source_destination_seen == 0
    assert features.source_destination_count == 0
    assert features.source_unique_destination_count == 1


def test_repeated_source_destination_pair() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(
            event_id="evt-1",
            timestamp=1000.0,
            source_host="HostA",
            destination_host="HostB",
        )
    )

    features = extractor.process_event(
        make_event(
            event_id="evt-2",
            timestamp=1001.0,
            source_host="HostA",
            destination_host="HostB",
        )
    )

    assert features.source_destination_seen == 1
    assert features.source_destination_count == 1


def test_unique_destination_counting() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, destination_host="HostB")
    )
    extractor.process_event(
        make_event(event_id="evt-2", timestamp=1001.0, destination_host="HostC")
    )

    features = extractor.process_event(
        make_event(event_id="evt-3", timestamp=1002.0, destination_host="HostD")
    )

    assert features.user_unique_destination_count == 2


def test_unique_source_counting_for_destination() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(
            event_id="evt-1",
            timestamp=1000.0,
            source_host="HostA",
            destination_host="HostZ",
        )
    )
    extractor.process_event(
        make_event(
            event_id="evt-2",
            timestamp=1001.0,
            source_host="HostB",
            destination_host="HostZ",
            user="UserB",
        )
    )

    features = extractor.process_event(
        make_event(
            event_id="evt-3",
            timestamp=1002.0,
            source_host="HostC",
            destination_host="HostZ",
            user="UserC",
        )
    )

    assert features.destination_unique_source_count == 2


def test_recent_window_counting() -> None:
    extractor = TemporalGraphFeatureExtractor(recent_window_seconds=60.0)
    extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, destination_host="HostB")
    )
    extractor.process_event(
        make_event(event_id="evt-2", timestamp=1030.0, destination_host="HostC")
    )

    features = extractor.process_event(
        make_event(event_id="evt-3", timestamp=1059.0, destination_host="HostC")
    )

    assert features.user_recent_event_count == 2
    assert features.user_recent_unique_destination_count == 2
    assert features.destination_recent_event_count == 1


def test_expired_events_are_not_counted() -> None:
    extractor = TemporalGraphFeatureExtractor(recent_window_seconds=60.0)
    extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, destination_host="HostB")
    )
    extractor.process_event(
        make_event(event_id="evt-2", timestamp=1030.0, destination_host="HostC")
    )

    features = extractor.process_event(
        make_event(event_id="evt-3", timestamp=1060.0, destination_host="HostD")
    )

    assert features.user_recent_event_count == 1
    assert features.user_recent_unique_destination_count == 1
    assert features.destination_recent_event_count == 0


def test_failed_authentication_and_recent_failure_count() -> None:
    extractor = TemporalGraphFeatureExtractor(recent_window_seconds=120.0)
    extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, success=False)
    )
    extractor.process_event(
        make_event(event_id="evt-2", timestamp=1010.0, success=True)
    )

    features = extractor.process_event(
        make_event(event_id="evt-3", timestamp=1020.0, success=False)
    )

    assert features.failed_authentication == 1
    assert features.user_recent_failure_count == 1


def test_time_since_features() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(
        make_event(
            event_id="evt-1",
            timestamp=1000.0,
            user="UserA",
            source_host="HostA",
            destination_host="HostB",
        )
    )
    extractor.process_event(
        make_event(
            event_id="evt-2",
            timestamp=1035.0,
            user="UserA",
            source_host="HostA",
            destination_host="HostC",
        )
    )

    features = extractor.process_event(
        make_event(
            event_id="evt-3",
            timestamp=1080.0,
            user="UserA",
            source_host="HostA",
            destination_host="HostB",
        )
    )

    assert features.time_since_user_previous_event == 45.0
    assert features.time_since_user_destination_event == 80.0
    assert features.time_since_source_destination_event == 80.0


def test_strict_temporal_ordering_rejects_older_event() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(make_event(event_id="evt-1", timestamp=1000.0))

    try:
        extractor.process_event(make_event(event_id="evt-2", timestamp=999.0))
    except ValueError as exc:
        assert "non-decreasing timestamp order" in str(exc)
    else:
        raise AssertionError("older timestamp should be rejected")


def test_same_timestamp_events_do_not_use_each_other() -> None:
    extractor = TemporalGraphFeatureExtractor()

    first = extractor.process_event(
        make_event(event_id="evt-1", timestamp=1000.0, destination_host="HostB")
    )
    second = extractor.process_event(
        make_event(event_id="evt-2", timestamp=1000.0, destination_host="HostB")
    )
    third = extractor.process_event(
        make_event(event_id="evt-3", timestamp=1001.0, destination_host="HostB")
    )

    assert first.user_destination_seen == 0
    assert second.user_destination_seen == 0
    assert second.user_destination_count == 0
    assert third.user_destination_seen == 1
    assert third.user_destination_count == 2


def test_reset_clears_state() -> None:
    extractor = TemporalGraphFeatureExtractor()
    extractor.process_event(make_event(event_id="evt-1", timestamp=1000.0))
    extractor.reset()

    features = extractor.process_event(make_event(event_id="evt-2", timestamp=2000.0))

    assert extractor.number_of_events_processed() == 1
    assert features.user_destination_seen == 0
    assert features.time_since_user_previous_event == NO_PRIOR_TIME_DELTA


def test_deterministic_output() -> None:
    events = [
        make_event(event_id="evt-1", timestamp=1000.0),
        make_event(event_id="evt-2", timestamp=1005.0, destination_host="HostC"),
        make_event(event_id="evt-3", timestamp=1010.0, success=False),
    ]
    extractor_one = TemporalGraphFeatureExtractor()
    extractor_two = TemporalGraphFeatureExtractor()

    result_one = [extractor_one.process_event(event).to_dict() for event in events]
    result_two = [extractor_two.process_event(event).to_dict() for event in events]

    assert result_one == result_two


def test_process_events_matches_sequential_calls() -> None:
    events = [
        make_event(event_id="evt-1", timestamp=1000.0),
        make_event(event_id="evt-2", timestamp=1001.0, destination_host="HostC"),
        make_event(event_id="evt-3", timestamp=1002.0, user="UserB"),
    ]

    sequential = TemporalGraphFeatureExtractor()
    sequential_results = [sequential.process_event(event).to_dict() for event in events]

    batched = TemporalGraphFeatureExtractor()
    batched_results = [result.to_dict() for result in batched.process_events(events)]

    assert batched_results == sequential_results


def test_extractor_does_not_retain_complete_event_history() -> None:
    extractor = TemporalGraphFeatureExtractor(recent_window_seconds=10.0)
    events = [
        make_event(
            event_id=f"evt-{idx}",
            timestamp=1000.0 + idx * 20.0,
            destination_host=f"Host{idx % 2}",
        )
        for idx in range(6)
    ]

    for event in events:
        extractor.process_event(event)

    state = extractor.get_state_size_estimate()
    assert extractor.number_of_events_processed() == 6
    assert state["user_recent_event_items"] <= 1
    assert state["destination_recent_event_items"] <= 1
    assert state["pending_events"] == 1


def test_feature_names_and_default_window_are_stable() -> None:
    extractor = TemporalGraphFeatureExtractor()

    assert extractor.recent_window_seconds == DEFAULT_RECENT_WINDOW_SECONDS
    assert EventFeatures.feature_names() == (
        "source_equals_destination",
        "user_destination_seen",
        "source_destination_seen",
        "user_destination_count",
        "source_destination_count",
        "user_unique_destination_count",
        "source_unique_destination_count",
        "destination_unique_source_count",
        "user_recent_event_count",
        "user_recent_unique_destination_count",
        "time_since_user_previous_event",
        "time_since_user_destination_event",
        "time_since_source_destination_event",
        "failed_authentication",
        "user_recent_failure_count",
        "destination_recent_event_count",
        "movement_edge",
    )
