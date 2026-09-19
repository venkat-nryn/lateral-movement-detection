"""Unit tests for chronological ML dataset preparation (M3.5)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.features import EventFeatures, TemporalGraphFeatureExtractor
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    DEVELOPMENT_MAX_EVENTS,
    DatasetSplit,
    FullDatasetGuardError,
    MLDataset,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TimestampRange,
    build_label_index,
    iter_chronological_samples,
    label_for_event,
)
from ml.preprocessing.schema import CanonicalEvent


def make_event(
    event_id: str,
    timestamp: float,
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


def make_ground_truth(*events: CanonicalEvent) -> RedteamGroundTruth:
    records = [
        RedteamRecord(
            redteam_id=f"redteam-{index}",
            timestamp=event.timestamp,
            user=event.user,
            source_host=event.source_host,
            destination_host=event.destination_host,
        )
        for index, event in enumerate(events)
    ]
    return RedteamGroundTruth(iter(records))


def test_chronological_splitting_is_timestamp_based() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 10.0),
        make_event("evt-3", 20.0),
        make_event("evt-4", 30.0),
    ]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(0.5, 0.25, 0.25),
    )

    assert dataset.get_split(SPLIT_TRAIN).event_ids == ["evt-1", "evt-2"]
    assert dataset.get_split(SPLIT_VALIDATION).event_ids == ["evt-3"]
    assert dataset.get_split(SPLIT_TEST).event_ids == ["evt-4"]


def test_splitting_is_deterministic_and_not_shuffled() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 10.0),
        make_event("evt-3", 20.0),
        make_event("evt-4", 30.0),
    ]
    ground_truth = make_ground_truth(events[2])

    first = MLDataset.from_events(events, ground_truth)
    second = MLDataset.from_events(events, ground_truth)

    for split_name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        assert first.get_split(split_name).event_ids == second.get_split(
            split_name
        ).event_ids
        assert list(first.iter_split(split_name)) == list(
            second.iter_split(split_name)
        )


def test_train_validation_test_ordering() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 50.0),
        make_event("evt-3", 80.0),
        make_event("evt-4", 100.0),
    ]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(0.5, 0.3, 0.2),
    )

    train_times = dataset.get_split(SPLIT_TRAIN).timestamps
    validation_times = dataset.get_split(SPLIT_VALIDATION).timestamps
    test_times = dataset.get_split(SPLIT_TEST).timestamps

    assert max(train_times) <= min(validation_times)
    assert max(validation_times) <= min(test_times)


def test_feature_dimension_is_always_17() -> None:
    dataset = MLDataset.from_events(
        [make_event("evt-1", 0.0), make_event("evt-2", 1.0)],
        make_ground_truth(),
    )

    assert dataset.feature_dimension() == 17
    for split_name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        for sample in dataset.iter_split(split_name):
            assert len(sample.features) == 17


def test_labels_are_binary() -> None:
    positive = make_event("evt-1", 0.0)
    negative = make_event("evt-2", 10.0)
    dataset = MLDataset.from_events(
        [positive, negative],
        make_ground_truth(positive),
        split_config=ChronologicalSplitConfig(0.5, 0.25, 0.25),
    )

    labels = []
    for split_name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        labels.extend(dataset.get_split(split_name).labels)
    assert set(labels) <= {0, 1}
    assert 1 in labels
    assert 0 in labels


def test_redteam_matching_is_separate_from_feature_extraction() -> None:
    positive_first_event = make_event("evt-1", 0.0)
    dataset = MLDataset.from_events(
        [positive_first_event],
        make_ground_truth(positive_first_event),
    )

    train_samples = list(dataset.iter_split(SPLIT_TRAIN))
    assert len(train_samples) == 1
    sample = train_samples[0]
    assert sample.label == 1
    assert sample.features[1] == 0
    assert sample.features[2] == 0


def test_no_future_event_is_used_in_features() -> None:
    events = [
        make_event("evt-1", 0.0, destination_host="HostB"),
        make_event("evt-2", 10.0, destination_host="HostB"),
    ]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    samples = list(dataset.iter_split(SPLIT_TRAIN))
    assert samples[0].features[1] == 0
    assert samples[0].features[3] == 0
    assert samples[1].features[1] == 1
    assert samples[1].features[3] == 1


def test_empty_split_handling_is_deterministic() -> None:
    events = [
        make_event("evt-1", 5.0),
        make_event("evt-2", 5.0),
        make_event("evt-3", 5.0),
    ]
    dataset = MLDataset.from_events(events, make_ground_truth())

    assert dataset.get_split(SPLIT_TRAIN).event_ids == ["evt-1", "evt-2", "evt-3"]
    assert len(dataset.get_split(SPLIT_VALIDATION)) == 0
    assert len(dataset.get_split(SPLIT_TEST)) == 0


def test_reset_reproducibility_by_rebuilding_dataset() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 1.0, destination_host="HostC"),
        make_event("evt-3", 2.0, success=False),
    ]
    ground_truth = make_ground_truth(events[2])

    first = MLDataset.from_events(events, ground_truth)
    second = MLDataset.from_events(events, ground_truth)

    assert first.split_sample_counts() == second.split_sample_counts()
    assert first.split_label_counts() == second.split_label_counts()


def test_bounded_processing_respects_max_events() -> None:
    events = [make_event(f"evt-{index}", float(index)) for index in range(6)]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(events[4]),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
        max_events=3,
    )

    assert dataset.events_processed == 3
    assert dataset.get_split(SPLIT_TRAIN).event_ids == ["evt-0", "evt-1", "evt-2"]
    assert dataset.split_label_counts()[SPLIT_TRAIN]["positive"] == 0


def test_empty_event_source_produces_empty_dataset() -> None:
    dataset = MLDataset.from_events([], make_ground_truth())

    assert dataset.events_processed == 0
    assert dataset.timestamp_range is None
    assert dataset.split_sample_counts() == {
        SPLIT_TRAIN: 0,
        SPLIT_VALIDATION: 0,
        SPLIT_TEST: 0,
    }


def test_dataset_output_is_deterministic() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 10.0, destination_host="HostC"),
        make_event("evt-3", 20.0, user="UserB"),
    ]
    ground_truth = make_ground_truth(events[1])

    first = MLDataset.from_events(events, ground_truth)
    second = MLDataset.from_events(events, ground_truth)

    first_records = [
        sample.to_dict(first.feature_names)
        for split_name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
        for sample in first.iter_split(split_name)
    ]
    second_records = [
        sample.to_dict(second.feature_names)
        for split_name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
        for sample in second.iter_split(split_name)
    ]

    assert first_records == second_records


def test_non_decreasing_timestamp_order_is_required() -> None:
    events = [make_event("evt-1", 5.0), make_event("evt-2", 4.0)]

    try:
        MLDataset.from_events(events, make_ground_truth())
    except ValueError as exc:
        assert "non-decreasing timestamp order" in str(exc)
    else:
        raise AssertionError("out-of-order events should be rejected")


# ---------------------------------------------------------------------------
# M3.5 completion coverage
# ---------------------------------------------------------------------------


def test_no_random_shuffling_anywhere_in_module() -> None:
    """The pipeline must contain no RNG at all, not merely a fixed seed."""
    import ml.preprocessing.ml_dataset as ml_dataset

    source = inspect.getsource(ml_dataset)
    for forbidden in ("import random", "random.", "shuffle", "np.random", "seed("):
        assert forbidden not in source


def test_split_assignment_is_a_pure_function_of_timestamp() -> None:
    config = ChronologicalSplitConfig()
    for timestamp in (0.0, 12.5, 70.0, 70.0000001, 85.0, 100.0):
        assert config.split_for_timestamp(timestamp, 0.0, 100.0) == (
            config.split_for_timestamp(timestamp, 0.0, 100.0)
        )
    assert config.split_for_timestamp(70.0, 0.0, 100.0) == SPLIT_TRAIN
    assert config.split_for_timestamp(80.0, 0.0, 100.0) == SPLIT_VALIDATION
    assert config.split_for_timestamp(100.0, 0.0, 100.0) == SPLIT_TEST


def test_same_timestamp_events_never_straddle_a_split_boundary() -> None:
    """Every event sharing a timestamp must land in exactly one split."""
    events = []
    index = 0
    for timestamp in (0.0, 70.0, 70.0, 70.0, 85.0, 85.0, 100.0):
        index += 1
        events.append(
            make_event(f"evt-{index}", timestamp, destination_host=f"Host{index}")
        )

    dataset = MLDataset.from_events(events, make_ground_truth())

    timestamp_to_split: dict[float, str] = {}
    for split_name in SPLIT_NAMES:
        for sample in dataset.iter_split(split_name):
            previous = timestamp_to_split.setdefault(sample.timestamp, split_name)
            assert previous == split_name


def test_same_timestamp_events_cannot_see_each_other() -> None:
    """Leakage guard: features(E) may only use timestamps strictly < T."""
    events = [
        make_event("evt-1", 10.0, destination_host="HostZ"),
        make_event("evt-2", 10.0, destination_host="HostZ"),
        make_event("evt-3", 11.0, destination_host="HostZ"),
    ]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    names = dataset.feature_names
    seen_index = names.index("user_destination_seen")
    count_index = names.index("user_destination_count")
    samples = list(dataset.iter_split(SPLIT_TRAIN))

    # Both same-timestamp events see zero prior history for (user, HostZ).
    assert samples[0].features[seen_index] == 0
    assert samples[0].features[count_index] == 0
    assert samples[1].features[seen_index] == 0
    assert samples[1].features[count_index] == 0
    # The later event sees both of them.
    assert samples[2].features[seen_index] == 1
    assert samples[2].features[count_index] == 2


def test_feature_vector_consistency_matches_extractor_contract() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 5.0, user="UserB", destination_host="HostC"),
        make_event("evt-3", 9.0, success=False),
    ]
    dataset = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    assert dataset.feature_names == EventFeatures.feature_names()
    assert "event_id" not in dataset.feature_names
    assert "timestamp" not in dataset.feature_names

    split = dataset.get_split(SPLIT_TRAIN)
    assert split.feature_dimension == 17
    for row in split.iter_feature_rows():
        assert len(row) == len(dataset.feature_names)
        assert all(isinstance(value, (int, float)) for value in row)

    extractor = TemporalGraphFeatureExtractor()
    expected = [
        tuple(float(value) for value in extractor.process_event(event).to_vector())
        for event in events
    ]
    assert list(split.iter_feature_rows()) == expected


def test_exact_redteam_matching_rejects_near_misses() -> None:
    event = make_event(
        "evt-1", 100.0, user="UserA", source_host="HostA", destination_host="HostB"
    )
    near_misses = [
        RedteamRecord("r1", 101.0, "UserA", "HostA", "HostB"),  # timestamp differs
        RedteamRecord("r2", 100.0, "UserB", "HostA", "HostB"),  # user differs
        RedteamRecord("r3", 100.0, "UserA", "HostX", "HostB"),  # source differs
        RedteamRecord("r4", 100.0, "UserA", "HostA", "HostY"),  # destination differs
    ]
    label_index = build_label_index(RedteamGroundTruth(iter(near_misses)))
    assert label_for_event(event, label_index) == 0

    exact = RedteamRecord("r5", 100.0, "UserA", "HostA", "HostB")
    exact_index = build_label_index(
        RedteamGroundTruth(iter(near_misses + [exact]))
    )
    assert label_for_event(event, exact_index) == 1


def test_redteam_records_are_not_injected_as_events() -> None:
    """Redteam records must only label; they must never add history."""
    events = [make_event("evt-1", 0.0), make_event("evt-2", 1.0)]
    unrelated = RedteamGroundTruth(
        iter(
            [
                RedteamRecord("r1", 0.5, "UserA", "HostA", "HostB"),
                RedteamRecord("r2", 0.7, "UserA", "HostA", "HostB"),
            ]
        )
    )

    with_labels = MLDataset.from_events(
        events, unrelated, split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0)
    )
    without_labels = MLDataset.from_events(
        events,
        make_ground_truth(),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    assert [
        sample.features for sample in with_labels.iter_split(SPLIT_TRAIN)
    ] == [sample.features for sample in without_labels.iter_split(SPLIT_TRAIN)]
    assert with_labels.total_positive_count() == 0


def test_split_counts_sum_to_events_processed() -> None:
    events = [
        make_event(f"evt-{index}", float(index), destination_host=f"Host{index}")
        for index in range(20)
    ]
    dataset = MLDataset.from_events(events, make_ground_truth())

    counts = dataset.split_sample_counts()
    assert sorted(counts) == sorted(SPLIT_NAMES)
    assert sum(counts.values()) == dataset.events_processed == 20
    assert counts[SPLIT_TRAIN] > 0


def test_label_counts_are_consistent_with_sample_counts() -> None:
    events = [make_event(f"evt-{index}", float(index)) for index in range(10)]
    dataset = MLDataset.from_events(
        events, make_ground_truth(events[0], events[5], events[9])
    )

    counts = dataset.split_sample_counts()
    label_counts = dataset.split_label_counts()
    for split_name in SPLIT_NAMES:
        positive = label_counts[split_name]["positive"]
        negative = label_counts[split_name]["negative"]
        assert positive + negative == counts[split_name]
        assert positive >= 0 and negative >= 0
    assert dataset.total_positive_count() == 3


def test_x_y_interface_is_ml_ready() -> None:
    events = [
        make_event(f"evt-{index}", float(index), destination_host=f"Host{index}")
        for index in range(8)
    ]
    dataset = MLDataset.from_events(events, make_ground_truth(events[1]))
    split = dataset.get_split(SPLIT_TRAIN)

    X, y = split.X_y()
    assert len(X) == len(y) == len(split)
    assert all(len(row) == dataset.feature_dimension() for row in X)
    assert all(label in (0, 1) for label in y)

    buffer = split.feature_buffer()
    assert buffer.typecode == "d"
    assert len(buffer) == len(split) * dataset.feature_dimension()
    assert tuple(buffer[0 : dataset.feature_dimension()]) == X[0]


def test_full_dataset_guard_blocks_unbounded_lanl_runs() -> None:
    missing = Path("data") / "raw" / "lanl" / "does-not-exist.txt.gz"

    with pytest.raises(FullDatasetGuardError):
        MLDataset.from_lanl_files(missing, missing, max_events=None)

    with pytest.raises(FullDatasetGuardError):
        MLDataset.from_lanl_files(
            missing, missing, max_events=DEVELOPMENT_MAX_EVENTS + 1
        )


def test_from_lanl_files_defaults_to_the_development_bound() -> None:
    default = inspect.signature(MLDataset.from_lanl_files).parameters[
        "max_events"
    ].default
    assert default == DEVELOPMENT_MAX_EVENTS == 1_000_000
    assert (
        inspect.signature(MLDataset.from_lanl_files)
        .parameters["allow_full_dataset"]
        .default
        is False
    )


def test_counts_only_mode_retains_no_feature_rows() -> None:
    events = [
        make_event(f"evt-{index}", float(index), destination_host=f"Host{index}")
        for index in range(12)
    ]
    ground_truth = make_ground_truth(events[3], events[11])

    retained = MLDataset.from_events(events, ground_truth)
    counts_only = MLDataset.from_events(
        events, ground_truth, retain_samples=False
    )

    assert counts_only.split_sample_counts() == retained.split_sample_counts()
    assert counts_only.split_label_counts() == retained.split_label_counts()
    for split_name in SPLIT_NAMES:
        split = counts_only.get_split(split_name)
        assert split.event_ids == []
        assert len(split.feature_buffer()) == 0
        assert list(split.iter_samples()) == []


def test_streaming_generator_matches_materialized_dataset() -> None:
    events = [
        make_event(f"evt-{index}", float(index), destination_host=f"Host{index % 3}")
        for index in range(15)
    ]
    ground_truth = make_ground_truth(events[2], events[14])
    dataset = MLDataset.from_events(events, ground_truth)

    streamed = list(
        iter_chronological_samples(
            iter(events),
            build_label_index(ground_truth),
            timestamp_range=TimestampRange(start=0.0, end=14.0),
        )
    )

    assert len(streamed) == len(events)
    materialized = {
        split_name: [
            (sample.event_id, sample.features, sample.label)
            for sample in dataset.iter_split(split_name)
        ]
        for split_name in SPLIT_NAMES
    }
    regrouped: dict[str, list[tuple[str, tuple[float, ...], int]]] = {
        name: [] for name in SPLIT_NAMES
    }
    for split_name, event_id, _timestamp, features, label in streamed:
        regrouped[split_name].append(
            (event_id, tuple(float(v) for v in features.to_vector()), label)
        )
    assert regrouped == materialized


def test_extractor_reset_reproduces_identical_features() -> None:
    events = [
        make_event("evt-1", 0.0),
        make_event("evt-2", 1.0, destination_host="HostC"),
        make_event("evt-3", 2.0, success=False),
    ]
    extractor = TemporalGraphFeatureExtractor()

    first = [extractor.process_event(event).to_vector() for event in events]
    extractor.reset()
    second = [extractor.process_event(event).to_vector() for event in events]

    assert first == second


def test_summary_reports_timestamp_range_and_redteam_overlap_honestly() -> None:
    events = [make_event(f"evt-{index}", float(index)) for index in range(5)]
    late_redteam = RedteamGroundTruth(
        iter([RedteamRecord("r1", 150885.0, "U620@DOM1", "C17693", "C1003")])
    )
    dataset = MLDataset.from_events(events, late_redteam)

    summary = dataset.summary()
    assert summary["events_processed"] == 5
    assert summary["feature_dimension"] == 17
    assert summary["timestamp_min"] == 0.0
    assert summary["timestamp_max"] == 4.0

    overlap = summary["redteam_overlap"]
    assert overlap["redteam_record_count"] == 1
    assert overlap["redteam_timestamp_min"] == 150885.0
    assert overlap["timestamp_ranges_overlap"] is False
    assert overlap["matched_positive_count"] == 0
    assert dataset.total_positive_count() == 0


def test_split_config_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        ChronologicalSplitConfig(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        ChronologicalSplitConfig(-0.1, 0.6, 0.5)


def test_dataset_split_rejects_non_binary_labels() -> None:
    split = DatasetSplit(name=SPLIT_TRAIN, feature_names=EventFeatures.feature_names())
    features = TemporalGraphFeatureExtractor().process_event(
        make_event("evt-1", 0.0)
    )

    with pytest.raises(ValueError):
        split.add_sample("evt-1", 0.0, features, 2)
