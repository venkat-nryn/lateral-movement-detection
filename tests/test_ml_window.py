"""Unit tests for the bounded real-LANL labeled window (M3.6).

All fixtures are tiny and in-memory. The 7.6 GB auth.txt.gz is never opened
here; the one real-data run for this module is performed separately.
"""

from __future__ import annotations

import gzip
import inspect

import pytest

from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.features import EventFeatures
from ml.preprocessing.lanl_adapter import iter_lanl_events
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    MLDataset,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
)
from ml.preprocessing.ml_window import (
    DEFAULT_CONTEXT_SECONDS,
    DEFAULT_MAX_WINDOW_EVENTS,
    WindowTruncationError,
    DEFAULT_WINDOW_SPAN_SECONDS,
    WindowSelectionError,
    WindowSpec,
    WindowStatistics,
    build_window_dataset,
    identity_overlap,
    iter_window_events,
    redteam_identity_profile,
    redteam_records_in_window,
    select_densest_redteam_window,
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


def ground_truth_from(*records: tuple[float, str, str, str]) -> RedteamGroundTruth:
    return RedteamGroundTruth(
        iter(
            [
                RedteamRecord(
                    redteam_id=f"redteam-{index}",
                    timestamp=timestamp,
                    user=user,
                    source_host=source_host,
                    destination_host=destination_host,
                )
                for index, (timestamp, user, source_host, destination_host) in
                enumerate(records)
            ]
        )
    )


def write_auth_gz(path, rows: list[tuple]) -> None:
    """Write a tiny gzip auth file in the real 9-column LANL layout."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for timestamp, user, source, destination, status in rows:
            handle.write(
                f"{timestamp},{user},{user},{source},{destination},"
                f"Kerberos,Network,LogOn,{status}\n"
            )


# ---------------------------------------------------------------------------
# 1. window includes required timestamps
# ---------------------------------------------------------------------------


def test_selected_window_includes_the_redteam_activity() -> None:
    ground_truth = ground_truth_from(
        (1000.0, "U1", "C1", "C2"),
        (1100.0, "U1", "C1", "C3"),
        (1200.0, "U2", "C1", "C4"),
        (9000.0, "U3", "C1", "C5"),
    )
    spec = select_densest_redteam_window(
        ground_truth, span_seconds=500.0, context_seconds=200.0
    )

    assert spec.emit_start == 1000.0
    assert spec.emit_end == 1500.0
    assert spec.context_start == 800.0
    assert spec.redteam_records_in_window == 3
    for timestamp in (1000.0, 1100.0, 1200.0):
        assert spec.contains_emitted(timestamp)
    assert not spec.contains_emitted(9000.0)


def test_densest_window_is_chosen_and_ties_break_to_earliest() -> None:
    ground_truth = ground_truth_from(
        (100.0, "U1", "C1", "C2"),
        (150.0, "U1", "C1", "C3"),
        (5000.0, "U2", "C1", "C4"),
        (5050.0, "U2", "C1", "C5"),
    )
    spec = select_densest_redteam_window(
        ground_truth, span_seconds=100.0, context_seconds=0.0
    )

    assert spec.emit_start == 100.0
    assert spec.redteam_records_in_window == 2


def test_window_selection_rejects_empty_ground_truth_and_bad_spans() -> None:
    with pytest.raises(WindowSelectionError):
        select_densest_redteam_window(RedteamGroundTruth(iter([])))
    with pytest.raises(WindowSelectionError):
        select_densest_redteam_window(
            ground_truth_from((1.0, "U1", "C1", "C2")), span_seconds=0.0
        )
    with pytest.raises(WindowSelectionError):
        WindowSpec(
            context_start=10.0,
            emit_start=5.0,
            emit_end=20.0,
            redteam_records_in_window=0,
        )


def test_window_defaults_are_bounded_and_real() -> None:
    assert DEFAULT_WINDOW_SPAN_SECONDS == 7200.0
    assert DEFAULT_CONTEXT_SECONDS == 3600.0
    assert DEFAULT_MAX_WINDOW_EVENTS == 4_000_000
    signature = inspect.signature(build_window_dataset)
    assert signature.parameters["max_events"].default == DEFAULT_MAX_WINDOW_EVENTS


# ---------------------------------------------------------------------------
# 2. + 3. context events build history but are excluded from output
# ---------------------------------------------------------------------------


def _window_dataset_from_events(events, ground_truth, spec, **kwargs):
    return MLDataset.from_event_source(
        event_source_factory=lambda: iter(events),
        ground_truth=ground_truth,
        timestamp_range=spec.timestamp_range(),
        emit_from_timestamp=spec.emit_start,
        **kwargs,
    )


def test_events_before_the_window_are_context_not_output() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [
        make_event("ctx-1", 10.0, destination_host="HostZ"),
        make_event("ctx-2", 20.0, destination_host="HostZ"),
        make_event("win-1", 100.0, destination_host="HostZ"),
        make_event("win-2", 150.0, destination_host="HostZ"),
    ]
    dataset = _window_dataset_from_events(
        events,
        ground_truth_from(),
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    emitted = [
        sample.event_id
        for name in SPLIT_NAMES
        for sample in dataset.iter_split(name)
    ]
    assert emitted == ["win-1", "win-2"]
    assert dataset.events_processed == 2
    assert dataset.context_events_processed == 2


def test_context_events_are_visible_as_history_to_the_first_sample() -> None:
    """The extractor must NOT be reset at the window boundary."""
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [
        make_event("ctx-1", 10.0, destination_host="HostZ"),
        make_event("ctx-2", 20.0, destination_host="HostZ"),
        make_event("win-1", 100.0, destination_host="HostZ"),
    ]
    dataset = _window_dataset_from_events(
        events,
        ground_truth_from(),
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    names = dataset.feature_names
    sample = next(iter(dataset.iter_split(SPLIT_TRAIN)))
    # The first emitted sample already carries the two context events.
    assert sample.features[names.index("user_destination_seen")] == 1
    assert sample.features[names.index("user_destination_count")] == 2

    # Without the context region the same event would start from zero history.
    cold = MLDataset.from_events(
        [events[2]],
        ground_truth_from(),
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )
    cold_sample = next(iter(cold.iter_split(SPLIT_TRAIN)))
    assert cold_sample.features[names.index("user_destination_count")] == 0


def test_events_after_the_window_are_excluded(tmp_path) -> None:
    path = tmp_path / "auth.txt.gz"
    write_auth_gz(
        path,
        [
            (10, "U1@DOM1", "C1", "C2", "Success"),
            (100, "U1@DOM1", "C1", "C3", "Success"),
            (150, "U1@DOM1", "C1", "C4", "Success"),
            (500, "U1@DOM1", "C1", "C5", "Success"),
            (900, "U1@DOM1", "C1", "C6", "Success"),
        ],
    )
    spec = WindowSpec(
        context_start=10.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )

    events = list(iter_window_events(path, spec))
    assert [event.timestamp for event in events] == [10.0, 100.0, 150.0]


def test_adapter_timestamp_range_stops_early_and_keeps_event_ids(tmp_path) -> None:
    path = tmp_path / "auth.txt.gz"
    rows = [
        (10, "U1@DOM1", "C1", "C2", "Success"),
        (20, "U2@DOM1", "C1", "C3", "Success"),
        (30, "U3@DOM1", "C1", "C4", "Success"),
        (40, "U4@DOM1", "C1", "C5", "Success"),
    ]
    write_auth_gz(path, rows)

    everything = list(iter_lanl_events(path))
    ranged = list(iter_lanl_events(path, start_timestamp=20.0, end_timestamp=30.0))

    assert [event.timestamp for event in ranged] == [20.0, 30.0]
    # Line numbering is preserved across skipped records, so ids are stable.
    assert [event.event_id for event in ranged] == [
        everything[1].event_id,
        everything[2].event_id,
    ]

    with pytest.raises(ValueError):
        list(iter_lanl_events(path, start_timestamp=50.0, end_timestamp=10.0))


# ---------------------------------------------------------------------------
# 4. + 5. exact redteam matching, no synthetic labels
# ---------------------------------------------------------------------------


def test_labels_use_exact_matching_only() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [
        make_event("win-1", 100.0, user="U1", source_host="C1", destination_host="C2"),
        make_event("win-2", 150.0, user="U1", source_host="C1", destination_host="C3"),
        make_event("win-3", 180.0, user="U2", source_host="C1", destination_host="C4"),
    ]
    ground_truth = ground_truth_from(
        (150.0, "U1", "C1", "C3"),   # exact match -> positive
        (180.0, "U2", "C1", "C9"),   # destination differs -> not a label
        (181.0, "U2", "C1", "C4"),   # timestamp differs -> not a label
    )
    dataset = _window_dataset_from_events(
        events,
        ground_truth,
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    samples = list(dataset.iter_split(SPLIT_TRAIN))
    assert [sample.label for sample in samples] == [0, 1, 0]
    assert dataset.total_positive_count() == 1


def test_nearby_redteam_timestamps_do_not_create_positives() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [make_event(f"win-{i}", 100.0 + i) for i in range(5)]
    # A redteam record one second away from every event, matching on all
    # identifiers. Proximity must never be enough.
    ground_truth = ground_truth_from((99.5, "UserA", "HostA", "HostB"))
    dataset = _window_dataset_from_events(
        events,
        ground_truth,
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    assert dataset.total_positive_count() == 0
    assert all(
        sample.label == 0 for sample in dataset.iter_split(SPLIT_TRAIN)
    )


def test_redteam_records_in_window_filters_by_emitted_region() -> None:
    ground_truth = ground_truth_from(
        (50.0, "U1", "C1", "C2"),
        (120.0, "U1", "C1", "C3"),
        (190.0, "U2", "C1", "C4"),
        (900.0, "U3", "C1", "C5"),
    )
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=2,
    )

    in_range = redteam_records_in_window(ground_truth, spec)
    assert [record.timestamp for record in in_range] == [120.0, 190.0]


# ---------------------------------------------------------------------------
# 6. + 7. + 11. chronological ordering, leakage, split ordering
# ---------------------------------------------------------------------------


def test_split_ordering_is_chronological_over_the_emitted_region() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [make_event("ctx-1", 10.0)] + [
        make_event(f"win-{i}", 100.0 + i, destination_host=f"Host{i}")
        for i in range(0, 101, 5)
    ]
    dataset = _window_dataset_from_events(events, ground_truth_from(), spec)

    train = list(dataset.get_split(SPLIT_TRAIN).timestamps)
    validation = list(dataset.get_split(SPLIT_VALIDATION).timestamps)
    test = list(dataset.get_split(SPLIT_TEST).timestamps)

    assert train and validation and test
    assert max(train) <= min(validation)
    assert max(validation) <= min(test)
    # Boundaries follow the window spec, not the observed events.
    boundaries = dataset.split_boundaries()
    assert boundaries["train_end"] == 170.0
    assert boundaries["validation_end"] == 185.0


def test_no_future_information_reaches_earlier_feature_vectors() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [
        make_event("win-1", 100.0, destination_host="HostZ"),
        make_event("win-2", 120.0, destination_host="HostZ"),
        make_event("win-3", 140.0, destination_host="HostZ"),
    ]
    dataset = _window_dataset_from_events(
        events,
        ground_truth_from(),
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    index = dataset.feature_names.index("user_destination_count")
    counts = [
        sample.features[index] for sample in dataset.iter_split(SPLIT_TRAIN)
    ]
    assert counts == [0, 1, 2]


def test_same_timestamp_window_events_cannot_see_each_other() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [
        make_event("win-1", 120.0, destination_host="HostZ"),
        make_event("win-2", 120.0, destination_host="HostZ"),
        make_event("win-3", 130.0, destination_host="HostZ"),
    ]
    dataset = _window_dataset_from_events(
        events,
        ground_truth_from(),
        spec,
        split_config=ChronologicalSplitConfig(1.0, 0.0, 0.0),
    )

    index = dataset.feature_names.index("user_destination_count")
    counts = [
        sample.features[index] for sample in dataset.iter_split(SPLIT_TRAIN)
    ]
    assert counts == [0, 0, 2]


# ---------------------------------------------------------------------------
# 8. + 9. + 10. dimension, determinism, counts
# ---------------------------------------------------------------------------


def test_feature_dimension_remains_17() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=0,
    )
    events = [make_event("ctx", 10.0)] + [
        make_event(f"win-{i}", 100.0 + i * 10) for i in range(8)
    ]
    dataset = _window_dataset_from_events(events, ground_truth_from(), spec)

    assert dataset.feature_dimension() == 17
    assert dataset.feature_names == EventFeatures.feature_names()
    for name in SPLIT_NAMES:
        for sample in dataset.iter_split(name):
            assert len(sample.features) == 17


def test_window_dataset_output_is_deterministic() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=1,
    )
    events = [make_event("ctx", 10.0)] + [
        make_event(f"win-{i}", 100.0 + i * 7, destination_host=f"Host{i % 3}")
        for i in range(12)
    ]
    ground_truth = ground_truth_from((114.0, "UserA", "HostA", "Host2"))

    first = _window_dataset_from_events(events, ground_truth, spec)
    second = _window_dataset_from_events(events, ground_truth, spec)

    assert first.split_sample_counts() == second.split_sample_counts()
    assert first.split_label_counts() == second.split_label_counts()
    for name in SPLIT_NAMES:
        assert [
            sample.to_dict(first.feature_names)
            for sample in first.iter_split(name)
        ] == [
            sample.to_dict(second.feature_names)
            for sample in second.iter_split(name)
        ]


def test_positive_and_negative_counts_are_correct() -> None:
    spec = WindowSpec(
        context_start=0.0,
        emit_start=100.0,
        emit_end=200.0,
        redteam_records_in_window=2,
    )
    events = [
        make_event("ctx", 10.0, destination_host="C0"),
        make_event("win-1", 110.0, user="U1", source_host="C1", destination_host="C2"),
        make_event("win-2", 120.0, user="U1", source_host="C1", destination_host="C3"),
        make_event("win-3", 190.0, user="U2", source_host="C1", destination_host="C4"),
    ]
    ground_truth = ground_truth_from(
        (120.0, "U1", "C1", "C3"),
        (190.0, "U2", "C1", "C4"),
    )
    dataset = _window_dataset_from_events(events, ground_truth, spec)

    counts = dataset.split_sample_counts()
    labels = dataset.split_label_counts()
    assert sum(counts.values()) == 3
    assert dataset.total_positive_count() == 2
    for name in SPLIT_NAMES:
        assert labels[name]["positive"] + labels[name]["negative"] == counts[name]
    # The context event is never counted as a sample or a label.
    assert dataset.context_events_processed == 1


def test_window_statistics_count_only_emitted_events() -> None:
    stats = WindowStatistics()
    stats.observe(make_event("ctx", 10.0, user="U0", source_host="C0"), emitted=False)
    stats.observe(
        make_event("w1", 100.0, user="U1", source_host="C1", destination_host="C2"),
        emitted=True,
    )
    stats.observe(
        make_event("w2", 110.0, user="U2", source_host="C1", destination_host="C3"),
        emitted=True,
    )

    assert stats.as_dict() == {
        "context_events": 1,
        "window_events": 2,
        "unique_users": 2,
        "unique_source_hosts": 1,
        "unique_destination_hosts": 2,
    }


# ---------------------------------------------------------------------------
# 12. + 13. no shuffling, bounded memory
# ---------------------------------------------------------------------------


def test_no_random_shuffling_in_window_module() -> None:
    import ml.preprocessing.ml_window as ml_window

    source = inspect.getsource(ml_window)
    for forbidden in ("import random", "random.", "shuffle", "np.random", "seed("):
        assert forbidden not in source


def _tiny_window_files(tmp_path):
    path = tmp_path / "auth.txt.gz"
    redteam = tmp_path / "redteam.txt.gz"
    write_auth_gz(
        path,
        [(t, "U1@DOM1", "C1", f"C{t}", "Success") for t in range(1, 60)],
    )
    with gzip.open(redteam, "wt", encoding="utf-8") as handle:
        handle.write("20,U1@DOM1,C1,C20\n")
    return path, redteam


def test_window_build_refuses_to_truncate(tmp_path) -> None:
    """The event ceiling is a guard, not a trim (PROJECT_STATE section 33).

    Silently returning a partial window is exactly what cut W2 short by its last
    19 minutes and 16 of its 92 redteam records.
    """
    path, redteam = _tiny_window_files(tmp_path)

    with pytest.raises(WindowTruncationError, match="ceiling"):
        build_window_dataset(
            path,
            redteam,
            span_seconds=30.0,
            context_seconds=10.0,
            max_events=5,
        )

    with pytest.raises(WindowSelectionError):
        build_window_dataset(path, redteam, max_events=0)


def test_window_build_succeeds_below_the_ceiling(tmp_path) -> None:
    path, redteam = _tiny_window_files(tmp_path)

    built = build_window_dataset(
        path, redteam, span_seconds=30.0, context_seconds=10.0, max_events=5_000
    )

    streamed = (
        built.dataset.events_processed + built.dataset.context_events_processed
    )
    assert 0 < streamed < 5_000
    assert built.dataset.retained_samples is True


def test_counts_only_mode_retains_no_rows(tmp_path) -> None:
    path = tmp_path / "auth.txt.gz"
    redteam = tmp_path / "redteam.txt.gz"
    write_auth_gz(
        path,
        [(t, "U1@DOM1", "C1", f"C{t % 7}", "Success") for t in range(1, 80)],
    )
    with gzip.open(redteam, "wt", encoding="utf-8") as handle:
        handle.write("40,U1@DOM1,C1,C5\n")

    retained = build_window_dataset(
        path, redteam, span_seconds=20.0, context_seconds=10.0
    )
    counts_only = build_window_dataset(
        path,
        redteam,
        span_seconds=20.0,
        context_seconds=10.0,
        retain_samples=False,
    )

    assert (
        counts_only.dataset.split_sample_counts()
        == retained.dataset.split_sample_counts()
    )
    assert (
        counts_only.dataset.split_label_counts()
        == retained.dataset.split_label_counts()
    )
    for name in SPLIT_NAMES:
        assert len(counts_only.dataset.get_split(name).feature_buffer()) == 0


# ---------------------------------------------------------------------------
# End-to-end on a tiny synthetic gzip file (software test only, not research)
# ---------------------------------------------------------------------------


def test_build_window_dataset_end_to_end(tmp_path) -> None:
    path = tmp_path / "auth.txt.gz"
    redteam = tmp_path / "redteam.txt.gz"
    rows = [
        (t, f"U{t % 5}@DOM1", f"C{t % 3}", f"C{(t * 7) % 11}", "Success")
        for t in range(1, 200)
    ]
    write_auth_gz(path, rows)
    # Pick three real rows from the file as ground truth.
    picks = [rows[120], rows[140], rows[160]]
    with gzip.open(redteam, "wt", encoding="utf-8") as handle:
        for timestamp, user, source, destination, _status in picks:
            handle.write(f"{timestamp},{user},{source},{destination}\n")

    built = build_window_dataset(
        path, redteam, span_seconds=60.0, context_seconds=20.0
    )

    assert built.spec.emit_start == float(picks[0][0])
    assert built.spec.emit_end == built.spec.emit_start + 60.0
    assert built.redteam_records_in_range == 3
    assert built.positive_count() == 3
    assert built.negative_count() == built.dataset.events_processed - 3
    assert 0.0 < built.positive_percentage() < 100.0
    assert built.dataset.context_events_processed == 20

    summary = built.summary()
    assert summary["feature_dimension"] == 17
    assert summary["class_distribution"]["positive"] == 3
    assert summary["redteam_in_range"]["matched_events"] == 3
    assert summary["window"]["span_seconds"] == 60.0
    assert summary["class_distribution"]["unique_users"] >= 1


# ---------------------------------------------------------------------------
# M4.1 -- second, temporally disjoint window selection and identity overlap
# ---------------------------------------------------------------------------


def test_exclusion_selects_a_disjoint_second_window() -> None:
    ground_truth = ground_truth_from(
        (1000.0, "U1", "C1", "C2"),
        (1050.0, "U1", "C1", "C3"),
        (1100.0, "U2", "C1", "C4"),
        (9000.0, "U3", "C9", "C5"),
        (9100.0, "U4", "C9", "C6"),
    )
    first = select_densest_redteam_window(
        ground_truth, span_seconds=200.0, context_seconds=100.0
    )
    second = select_densest_redteam_window(
        ground_truth,
        span_seconds=200.0,
        context_seconds=100.0,
        exclude=[first],
    )

    assert first.emit_start == 1000.0
    assert second.emit_start == 9000.0
    # Read regions, context included, must not touch.
    assert second.context_start > first.emit_end
    assert second.span_seconds == first.span_seconds
    assert second.context_seconds == first.context_seconds


def test_exclusion_accounts_for_the_context_region() -> None:
    ground_truth = ground_truth_from(
        (1000.0, "U1", "C1", "C2"),
        (1500.0, "U2", "C1", "C3"),
    )
    first = select_densest_redteam_window(
        ground_truth, span_seconds=100.0, context_seconds=600.0
    )
    # The second candidate's context would reach back into the first window.
    with pytest.raises(WindowSelectionError):
        select_densest_redteam_window(
            ground_truth,
            span_seconds=100.0,
            context_seconds=600.0,
            exclude=[first],
        )


def test_exclusion_raises_when_nothing_disjoint_remains() -> None:
    ground_truth = ground_truth_from((100.0, "U1", "C1", "C2"))
    first = select_densest_redteam_window(
        ground_truth, span_seconds=50.0, context_seconds=0.0
    )

    with pytest.raises(WindowSelectionError):
        select_densest_redteam_window(
            ground_truth, span_seconds=50.0, context_seconds=0.0, exclude=[first]
        )


def test_window_selection_stays_deterministic_with_exclusions() -> None:
    ground_truth = ground_truth_from(
        (100.0, "U1", "C1", "C2"),
        (120.0, "U1", "C1", "C3"),
        (5000.0, "U2", "C2", "C4"),
        (5010.0, "U3", "C2", "C5"),
        (5020.0, "U4", "C2", "C6"),
    )
    first = select_densest_redteam_window(
        ground_truth, span_seconds=100.0, context_seconds=50.0
    )
    runs = [
        select_densest_redteam_window(
            ground_truth,
            span_seconds=100.0,
            context_seconds=50.0,
            exclude=[first],
        )
        for _ in range(3)
    ]
    assert all(run == runs[0] for run in runs)


def test_redteam_identity_profile_summarises_records() -> None:
    ground_truth = ground_truth_from(
        (10.0, "U1", "C1", "C2"),
        (20.0, "U2", "C1", "C3"),
        (30.0, "U1", "C1", "C2"),
    )
    profile = redteam_identity_profile(list(ground_truth.iter_records()))

    assert profile.record_count == 3
    assert profile.users == frozenset({"U1", "U2"})
    assert profile.source_hosts == frozenset({"C1"})
    assert profile.destination_hosts == frozenset({"C2", "C3"})
    assert profile.as_dict()["unique_users"] == 2
    assert profile.as_dict()["source_hosts"] == ["C1"]


def test_identity_overlap_reports_shared_and_distinct() -> None:
    first = redteam_identity_profile(
        list(ground_truth_from((10.0, "U1", "C1", "C2"), (20.0, "U2", "C1", "C3"))
             .iter_records())
    )
    second = redteam_identity_profile(
        list(ground_truth_from((90.0, "U2", "C1", "C4"), (95.0, "U3", "C9", "C5"))
             .iter_records())
    )

    overlap = identity_overlap(first, second)

    assert overlap["users"]["shared"] == 1
    assert overlap["users"]["shared_values"] == ["U2"]
    assert overlap["users"]["first_only"] == 1
    assert overlap["users"]["second_only"] == 1
    assert overlap["users"]["union"] == 3
    assert overlap["users"]["jaccard"] == pytest.approx(1 / 3)
    assert overlap["source_hosts"]["shared_values"] == ["C1"]
    assert overlap["destination_hosts"]["shared"] == 0


def test_identity_overlap_handles_empty_profiles() -> None:
    empty = redteam_identity_profile([])
    overlap = identity_overlap(empty, empty)

    assert overlap["users"]["union"] == 0
    assert overlap["users"]["jaccard"] == 0.0
