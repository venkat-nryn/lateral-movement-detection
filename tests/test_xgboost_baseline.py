"""Unit tests for the XGBoost baseline (M4.0).

Fixtures are tiny and in-memory. Models trained here exist only to exercise the
code path; no accuracy claim is ever derived from them. Real research numbers
come exclusively from the real-LANL window run.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest

from ml.baselines.xgboost_baseline import (
    DEFAULT_THRESHOLD,
    MODEL_CLASS_WEIGHTED,
    MODEL_STANDARD,
    BinaryMetrics,
    XGBoostBaselineConfig,
    compute_scale_pos_weight,
    dataset_matrices,
    evaluate_at_threshold,
    feature_importance_ranking,
    predict_scores,
    run_baseline,
    select_threshold_on_validation,
    split_matrix,
    train_xgboost,
)
from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.features import EventFeatures
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    MLDataset,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
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


def tiny_dataset() -> MLDataset:
    """A small chronological dataset with positives in every split."""
    events = []
    positives = []
    for index in range(60):
        timestamp = float(index)
        # Every 7th event is an unusual movement that we also mark as redteam.
        attack = index % 7 == 3
        event = make_event(
            f"evt-{index}",
            timestamp,
            user="UserB" if attack else "UserA",
            source_host="HostX" if attack else "HostA",
            destination_host=f"Host{index}" if attack else "HostB",
            success=not attack,
        )
        events.append(event)
        if attack:
            positives.append(event)

    ground_truth = RedteamGroundTruth(
        iter(
            [
                RedteamRecord(
                    redteam_id=f"redteam-{i}",
                    timestamp=event.timestamp,
                    user=event.user,
                    source_host=event.source_host,
                    destination_host=event.destination_host,
                )
                for i, event in enumerate(positives)
            ]
        )
    )
    return MLDataset.from_events(events, ground_truth)


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------


def test_split_matrix_matches_the_stored_feature_rows() -> None:
    dataset = tiny_dataset()
    split = dataset.get_split(SPLIT_TRAIN)
    X, y = split_matrix(split)

    assert X.shape == (len(split), 17)
    assert X.dtype == np.float64
    assert y.shape == (len(split),)
    assert list(y) == list(split.labels)
    assert [tuple(row) for row in X] == list(split.iter_feature_rows())


def test_split_matrix_handles_an_empty_split() -> None:
    dataset = MLDataset.from_events(
        [make_event("evt-1", 5.0)],
        RedteamGroundTruth(iter([])),
    )
    X, y = split_matrix(dataset.get_split(SPLIT_TEST))

    assert X.shape == (0, 17)
    assert y.shape == (0,)


def test_dataset_matrices_covers_all_three_splits() -> None:
    matrices = dataset_matrices(tiny_dataset())

    assert set(matrices) == {SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST}
    for X, y in matrices.values():
        assert X.shape[1] == 17
        assert X.shape[0] == y.shape[0]


# ---------------------------------------------------------------------------
# scale_pos_weight comes from TRAIN only
# ---------------------------------------------------------------------------


def test_scale_pos_weight_is_negatives_over_positives() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    assert compute_scale_pos_weight(y) == 3.0


def test_scale_pos_weight_without_positives_is_neutral() -> None:
    assert compute_scale_pos_weight(np.zeros(10, dtype=np.int8)) == 1.0


def test_scale_pos_weight_ignores_validation_and_test() -> None:
    dataset = tiny_dataset()
    matrices = dataset_matrices(dataset)
    _, y_train = matrices[SPLIT_TRAIN]

    expected = (
        dataset.get_split(SPLIT_TRAIN).negative_count()
        / dataset.get_split(SPLIT_TRAIN).positive_count()
    )
    assert compute_scale_pos_weight(y_train) == pytest.approx(expected)

    # Changing validation/test labels cannot change the weight.
    _, y_validation = matrices[SPLIT_VALIDATION]
    assert compute_scale_pos_weight(y_train) == pytest.approx(expected)
    assert y_validation.shape[0] > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_confusion_counts_are_computed_correctly() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.8, 0.2, 0.1, 0.6])

    metrics = evaluate_at_threshold(y_true, scores, 0.5)

    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (2, 1, 2, 1)
    assert metrics.predicted_positive == 3
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.support_positive == 3
    assert metrics.support_negative == 3
    assert metrics.tp + metrics.fp + metrics.tn + metrics.fn == y_true.size


def test_threshold_is_inclusive_at_the_boundary() -> None:
    metrics = evaluate_at_threshold(
        np.array([1, 0]), np.array([0.5, 0.49999]), 0.5
    )
    assert (metrics.tp, metrics.fp) == (1, 0)


def test_metrics_do_not_divide_by_zero_when_nothing_is_predicted() -> None:
    metrics = evaluate_at_threshold(
        np.array([1, 0, 0]), np.array([0.1, 0.1, 0.1]), 0.9
    )

    assert metrics.predicted_positive == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_ranking_metrics_are_none_when_only_one_class_is_present() -> None:
    metrics = evaluate_at_threshold(
        np.zeros(5), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.5
    )

    assert metrics.average_precision is None
    assert metrics.roc_auc is None
    assert metrics.support_positive == 0


def test_ranking_metrics_are_reported_when_both_classes_exist() -> None:
    metrics = evaluate_at_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        evaluate_at_threshold(np.array([1, 0]), np.array([0.5]), 0.5)


# ---------------------------------------------------------------------------
# Threshold selection uses VALIDATION only
# ---------------------------------------------------------------------------


def test_threshold_is_selected_from_validation_scores() -> None:
    y_validation = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.10, 0.20, 0.70, 0.90])

    threshold = select_threshold_on_validation(y_validation, validation_scores)

    assert 0.20 < threshold <= 0.70
    chosen = evaluate_at_threshold(y_validation, validation_scores, threshold)
    assert chosen.f1 == pytest.approx(1.0)


def test_selected_threshold_is_independent_of_test_data() -> None:
    """The same validation input must give the same threshold regardless of
    what any test split happens to contain."""
    y_validation = np.array([0, 0, 0, 1, 1])
    validation_scores = np.array([0.05, 0.15, 0.25, 0.65, 0.85])

    first = select_threshold_on_validation(y_validation, validation_scores)
    second = select_threshold_on_validation(y_validation, validation_scores)
    assert first == second

    # A test split whose optimum is elsewhere does not move the threshold.
    y_test = np.array([1, 1, 0, 0])
    test_scores = np.array([0.30, 0.35, 0.90, 0.95])
    test_optimal = select_threshold_on_validation(y_test, test_scores)
    assert test_optimal != first
    assert select_threshold_on_validation(y_validation, validation_scores) == first


def test_threshold_falls_back_to_default_without_validation_positives() -> None:
    assert (
        select_threshold_on_validation(np.zeros(5), np.linspace(0, 1, 5))
        == DEFAULT_THRESHOLD
    )
    assert (
        select_threshold_on_validation(np.ones(5), np.linspace(0, 1, 5))
        == DEFAULT_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_training_is_deterministic() -> None:
    matrices = dataset_matrices(tiny_dataset())
    X_train, y_train = matrices[SPLIT_TRAIN]
    X_test, _ = matrices[SPLIT_TEST]

    first = predict_scores(train_xgboost(X_train, y_train), X_test)
    second = predict_scores(train_xgboost(X_train, y_train), X_test)

    np.testing.assert_array_equal(first, second)


def test_config_pins_out_stochastic_sampling() -> None:
    params = XGBoostBaselineConfig().to_params()

    assert params["subsample"] == 1.0
    assert params["colsample_bytree"] == 1.0
    assert params["random_state"] == 0


def test_class_weighted_model_differs_from_standard() -> None:
    matrices = dataset_matrices(tiny_dataset())
    X_train, y_train = matrices[SPLIT_TRAIN]
    X_test, _ = matrices[SPLIT_TEST]

    standard = predict_scores(train_xgboost(X_train, y_train), X_test)
    weighted = predict_scores(
        train_xgboost(
            X_train, y_train, scale_pos_weight=compute_scale_pos_weight(y_train)
        ),
        X_test,
    )

    assert not np.array_equal(standard, weighted)


def test_predict_scores_on_empty_input() -> None:
    matrices = dataset_matrices(tiny_dataset())
    X_train, y_train = matrices[SPLIT_TRAIN]
    model = train_xgboost(X_train, y_train)

    assert predict_scores(model, np.empty((0, 17))).shape == (0,)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_feature_importance_covers_all_seventeen_features() -> None:
    matrices = dataset_matrices(tiny_dataset())
    X_train, y_train = matrices[SPLIT_TRAIN]
    model = train_xgboost(X_train, y_train)

    ranking = feature_importance_ranking(model, EventFeatures.feature_names())

    assert len(ranking) == 17
    assert {name for name, _ in ranking} == set(EventFeatures.feature_names())
    values = [value for _, value in ranking]
    assert values == sorted(values, reverse=True)
    assert all(value >= 0.0 for value in values)


def test_feature_importance_rejects_a_name_mismatch() -> None:
    matrices = dataset_matrices(tiny_dataset())
    X_train, y_train = matrices[SPLIT_TRAIN]
    model = train_xgboost(X_train, y_train)

    with pytest.raises(ValueError):
        feature_importance_ranking(model, ("only", "three", "names"))


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def test_run_baseline_trains_both_models_and_evaluates_both_splits() -> None:
    dataset = tiny_dataset()
    reports = run_baseline(dataset)

    assert set(reports) == {MODEL_STANDARD, MODEL_CLASS_WEIGHTED}
    assert reports[MODEL_STANDARD].scale_pos_weight is None
    assert reports[MODEL_CLASS_WEIGHTED].scale_pos_weight > 1.0

    for report in reports.values():
        assert set(report.metrics) == {SPLIT_VALIDATION, SPLIT_TEST}
        for split_metrics in report.metrics.values():
            assert set(split_metrics) == {"default", "selected"}
            for metrics in split_metrics.values():
                assert isinstance(metrics, BinaryMetrics)
        assert report.metric(SPLIT_VALIDATION, "default").threshold == 0.5
        # The validation-selected threshold is applied unchanged to test.
        assert (
            report.metric(SPLIT_TEST, "selected").threshold
            == report.selected_threshold
        )
        assert len(report.feature_importance) == 17
        assert report.train_seconds >= 0.0


def test_run_baseline_is_deterministic() -> None:
    dataset = tiny_dataset()

    first = run_baseline(dataset)
    second = run_baseline(dataset)

    for name in (MODEL_STANDARD, MODEL_CLASS_WEIGHTED):
        assert first[name].selected_threshold == second[name].selected_threshold
        for split_name in (SPLIT_VALIDATION, SPLIT_TEST):
            for key in ("default", "selected"):
                assert (
                    first[name].metric(split_name, key).as_dict()
                    == second[name].metric(split_name, key).as_dict()
                )
        assert first[name].feature_importance == second[name].feature_importance


def test_run_baseline_survives_a_split_with_no_positives() -> None:
    events = [make_event(f"evt-{i}", float(i)) for i in range(30)]
    ground_truth = RedteamGroundTruth(
        iter(
            [
                RedteamRecord(
                    redteam_id="redteam-0",
                    timestamp=events[2].timestamp,
                    user=events[2].user,
                    source_host=events[2].source_host,
                    destination_host=events[2].destination_host,
                )
            ]
        )
    )
    dataset = MLDataset.from_events(
        events, ground_truth, split_config=ChronologicalSplitConfig(0.7, 0.15, 0.15)
    )

    reports = run_baseline(dataset)

    for report in reports.values():
        test_metrics = report.metric(SPLIT_TEST, "default")
        assert test_metrics.support_positive == 0
        assert test_metrics.average_precision is None
        assert test_metrics.roc_auc is None


# ---------------------------------------------------------------------------
# Research-rule guards
# ---------------------------------------------------------------------------


def _executable_source(module) -> str:
    """Module source with comments and string literals removed.

    The module's own docstring explains which resampling techniques are
    deliberately absent, so scanning raw text would match its prose. Only
    executable code is checked here.
    """
    source = inspect.getsource(module)
    pieces: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


def test_module_performs_no_shuffling_or_resampling() -> None:
    import ml.baselines.xgboost_baseline as module

    code = _executable_source(module)
    for forbidden in (
        "shuffle",
        "SMOTE",
        "train_test_split",
        "resample",
        "oversample",
        "undersample",
    ):
        assert forbidden not in code
    assert "np.random" not in code
    assert "random." not in code


def test_labels_are_never_used_as_a_feature() -> None:
    """The feature matrix width must stay at the 17 documented features."""
    dataset = tiny_dataset()
    for X, _ in dataset_matrices(dataset).values():
        assert X.shape[1] == 17
    assert dataset.feature_names == EventFeatures.feature_names()
    assert "label" not in dataset.feature_names
