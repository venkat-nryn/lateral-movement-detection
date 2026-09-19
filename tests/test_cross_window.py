"""Unit tests for the M4.6 frozen cross-window evaluation.

Fixtures are tiny, deterministic and in-memory. Models trained here exist only
to exercise the protocol; no generalization claim is derived from them. The real
W1 -> W2 evaluation lives in ``scripts/run_cross_window.py``.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from ml.baselines.xgboost_baseline import (
    XGBoostBaselineConfig,
    predict_scores,
    select_threshold_on_validation,
    split_matrix,
)
from ml.evaluation import cross_window
from ml.evaluation.cross_window import (
    CrossWindowError,
    FrozenDetector,
    assert_generalization_windows,
    evaluate_frozen,
    evaluation_window,
    fit_frozen_detector,
    model_digest,
    positive_source_hosts,
    quantile_summary,
)
from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.ml_dataset import (
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    MLDataset,
)
from ml.preprocessing.ml_window import WindowSpec
from ml.preprocessing.schema import CanonicalEvent

SMALL = XGBoostBaselineConfig(n_estimators=10, max_depth=3, n_jobs=1)


def make_events(count: int = 90, start: float = 0.0):
    events, positives = [], []
    for index in range(count):
        attack = index % 7 == 3
        event = CanonicalEvent(
            event_id=f"evt-{index}",
            timestamp=start + index,
            user="UserB" if attack else "UserA",
            source_host="HostX" if attack else "HostA",
            destination_host=f"Host{index}" if attack else "HostB",
            event_type="authentication",
            success=not attack,
        )
        events.append(event)
        if attack:
            positives.append(event)
    return events, positives


def ground_truth_for(positives) -> RedteamGroundTruth:
    return RedteamGroundTruth(
        iter(
            RedteamRecord(
                redteam_id=f"rt-{i}",
                timestamp=e.timestamp,
                user=e.user,
                source_host=e.source_host,
                destination_host=e.destination_host,
            )
            for i, e in enumerate(positives)
        )
    )


def make_dataset(count: int = 90, start: float = 0.0) -> MLDataset:
    events, positives = make_events(count, start)
    return MLDataset.from_events(events, ground_truth_for(positives))


@pytest.fixture(scope="module")
def development() -> MLDataset:
    return make_dataset()


@pytest.fixture(scope="module")
def detector(development) -> FrozenDetector:
    return fit_frozen_detector(development, config=SMALL)


# ----------------------------------------------------------------------
# 1. Fitting reads TRAIN and VALIDATION only
# ----------------------------------------------------------------------


def test_fit_trains_on_the_training_split_only(development, monkeypatch):
    seen = {}
    real_train = cross_window.train_xgboost

    def spy(X, y, **kwargs):
        seen["rows"], seen["positives"] = X.shape[0], int(np.count_nonzero(y))
        return real_train(X, y, **kwargs)

    monkeypatch.setattr(cross_window, "train_xgboost", spy)
    frozen = fit_frozen_detector(development, config=SMALL)
    train = development.get_split(SPLIT_TRAIN)

    assert seen["rows"] == len(train) == frozen.train_rows
    assert seen["positives"] == train.positive_count() == frozen.train_positives


def test_threshold_is_selected_on_validation_only(development, detector):
    X_val, y_val = split_matrix(development.get_split(SPLIT_VALIDATION))
    expected = select_threshold_on_validation(
        y_val, predict_scores(detector.model, X_val)
    )

    assert detector.threshold == pytest.approx(expected)
    assert detector.validation_rows == len(X_val)
    assert detector.validation_metrics.threshold == detector.threshold


def test_fit_refuses_a_training_split_without_positives():
    events, _ = make_events()
    no_labels = MLDataset.from_events(events, RedteamGroundTruth(iter([])))
    with pytest.raises(CrossWindowError, match="at least one positive"):
        fit_frozen_detector(no_labels, config=SMALL)


# ----------------------------------------------------------------------
# 2. Freezing is enforced
# ----------------------------------------------------------------------


def test_frozen_detector_cannot_be_mutated(detector):
    with pytest.raises(dataclasses.FrozenInstanceError):
        detector.threshold = 0.5  # type: ignore[misc]


def test_digest_is_unchanged_by_evaluation(detector):
    window = evaluation_window(make_dataset(start=1000.0))
    before = model_digest(detector.model)
    evaluate_frozen(detector, window)

    assert model_digest(detector.model) == before == detector.digest


def test_a_substituted_model_is_detected(development, detector):
    other = fit_frozen_detector(
        development, config=XGBoostBaselineConfig(n_estimators=3, max_depth=2, n_jobs=1)
    )
    tampered = dataclasses.replace(detector, model=other.model)
    window = evaluation_window(make_dataset(start=1000.0))

    with pytest.raises(CrossWindowError, match="modified"):
        evaluate_frozen(tampered, window)


def test_evaluation_never_refits_or_reselects(detector, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not fit or select a threshold")

    monkeypatch.setattr(cross_window, "train_xgboost", forbidden)
    monkeypatch.setattr(cross_window, "select_threshold_on_validation", forbidden)
    metrics, scores = evaluate_frozen(detector, evaluation_window(make_dataset(start=1000.0)))

    assert metrics.threshold == detector.threshold
    assert scores.shape == (90,)


def test_evaluation_source_contains_no_fitting_call():
    source = inspect.getsource(evaluate_frozen)
    assert "train_xgboost" not in source
    assert "select_threshold" not in source
    assert set(inspect.signature(evaluate_frozen).parameters) == {"detector", "window"}


def test_evaluation_window_labels_cannot_change_predictions(detector):
    window = evaluation_window(make_dataset(start=1000.0))
    flipped = dataclasses.replace(window, y=(1 - window.y).astype(np.int8))

    metrics, scores = evaluate_frozen(detector, window)
    flipped_metrics, flipped_scores = evaluate_frozen(detector, flipped)

    assert np.array_equal(scores, flipped_scores)
    assert metrics.predicted_positive == flipped_metrics.predicted_positive
    assert metrics.tp != flipped_metrics.tp or metrics.fp != flipped_metrics.fp


def test_evaluation_is_deterministic(detector):
    window = evaluation_window(make_dataset(start=1000.0))
    _, first = evaluate_frozen(detector, window)
    _, second = evaluate_frozen(detector, window)
    assert np.array_equal(first, second)


def test_scores_validate_the_feature_width(detector):
    with pytest.raises(CrossWindowError, match="feature columns"):
        detector.scores(np.zeros((2, 5)))


# ----------------------------------------------------------------------
# 3. The evaluation window is the whole emitted region, in order
# ----------------------------------------------------------------------


def test_evaluation_window_covers_every_emitted_event_in_order():
    dataset = make_dataset(start=500.0)
    window = evaluation_window(dataset)

    assert window.rows == dataset.events_processed == 90
    assert window.positives == dataset.total_positive_count()
    assert np.all(np.diff(window.timestamps) >= 0)
    assert len(set(window.event_ids)) == window.rows
    # Concatenation order is train -> validation -> test.
    expected = [eid for name in SPLIT_NAMES for eid in dataset.get_split(name).event_ids]
    assert list(window.event_ids) == expected


def test_evaluation_window_excludes_context_events():
    events, positives = make_events(90)
    dataset = MLDataset.from_event_source(
        event_source_factory=lambda: iter(events),
        ground_truth=ground_truth_for(positives),
        emit_from_timestamp=30.0,
    )
    window = evaluation_window(dataset)

    assert window.rows == 60
    assert window.timestamps.min() >= 30.0


def test_evaluation_window_requires_retained_samples():
    events, positives = make_events()
    counts_only = MLDataset.from_events(
        events, ground_truth_for(positives), retain_samples=False
    )
    with pytest.raises(CrossWindowError, match="retained samples"):
        evaluation_window(counts_only)


# ----------------------------------------------------------------------
# 4. Temporal separation of the windows
# ----------------------------------------------------------------------


def spec(context_start, emit_start, emit_end):
    return WindowSpec(
        context_start=context_start,
        emit_start=emit_start,
        emit_end=emit_end,
        redteam_records_in_window=1,
    )


def test_a_later_disjoint_window_is_accepted():
    assert_generalization_windows(spec(0, 100, 200), spec(300, 400, 500))


@pytest.mark.parametrize(
    "evaluation",
    [
        spec(200, 300, 400),  # context starts exactly at development emit_end
        spec(150, 250, 350),  # context overlaps the development emission
        spec(0, 50, 90),  # entirely earlier
    ],
)
def test_overlapping_touching_or_earlier_windows_are_rejected(evaluation):
    with pytest.raises(CrossWindowError, match="strictly after"):
        assert_generalization_windows(spec(0, 100, 200), evaluation)


def test_the_recorded_w1_and_w2_satisfy_the_guard():
    w1 = WindowSpec(760506.0, 764106.0, 771306.0, 102)
    w2 = WindowSpec(1067648.0, 1071248.0, 1078448.0, 92)
    assert_generalization_windows(w1, w2)
    with pytest.raises(CrossWindowError):
        assert_generalization_windows(w2, w1)


# ----------------------------------------------------------------------
# 5. Reporting helpers
# ----------------------------------------------------------------------


def test_quantile_summary_values_and_empty_input():
    summary = quantile_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["min"] == 1.0 and summary["max"] == 5.0
    assert summary["median"] == 3.0

    empty = quantile_summary([])
    assert empty["n"] == 0
    assert empty["median"] is None


def test_positive_source_hosts_refuse_to_guess():
    def record(ts, host):
        return RedteamRecord(
            redteam_id=f"rt-{ts}-{host}",
            timestamp=ts,
            user="U1",
            source_host=host,
            destination_host="D1",
        )

    records = [
        record(10.0, "C1"),
        record(20.0, "C1"),
        record(20.0, "C1"),
        record(30.0, "C1"),
        record(30.0, "C2"),
    ]
    hosts = positive_source_hosts([10.0, 20.0, 30.0, 40.0], records)

    assert hosts == ["C1", "C1", None, None]
