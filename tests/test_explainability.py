"""Unit tests for the M4.5 TreeSHAP explainability module.

Fixtures are tiny, deterministic and in-memory. Models trained here exist only
to exercise attribution behaviour; no research claim is derived from them. The
real W1 attribution run lives in ``scripts/run_explainability.py``.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest
from xgboost import XGBClassifier

from ml.baselines.xgboost_baseline import (
    evaluate_at_threshold,
    predict_scores,
    split_matrix,
    train_xgboost,
    XGBoostBaselineConfig,
)
from ml.evaluation import explainability
from ml.evaluation.explainability import (
    OUTCOME_NAMES,
    ExplainabilityError,
    deletion_check,
    explain_tree_model,
    global_importance,
    model_margins,
    outcome_indices,
    rank_correlation,
    top_contributions,
    train_reference_values,
)
from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.features import EventFeatures
from ml.preprocessing.ml_dataset import SPLIT_TRAIN, MLDataset
from ml.preprocessing.schema import CanonicalEvent

NAMES = ("signal", "noise_a", "noise_b", "constant")


def synthetic_problem():
    """Label depends on column 0 only; column 3 is constant and never split."""
    rng = np.random.default_rng(0)
    X = rng.random((400, 4))
    X[:, 3] = 0.5
    y = (X[:, 0] > 0.6).astype(np.int64)
    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        tree_method="hist",
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def problem():
    return synthetic_problem()


# ----------------------------------------------------------------------
# 1. Shapley properties
# ----------------------------------------------------------------------


def test_contributions_have_one_column_per_feature(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)

    assert explanation.contributions.shape == (400, 4)
    assert explanation.bias.shape == (400,)
    assert explanation.margins.shape == (400,)
    assert explanation.feature_names == NAMES


def test_local_accuracy_contributions_sum_to_the_margin(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)

    assert explanation.local_accuracy_error() < 1e-4


def test_a_never_split_feature_receives_exactly_zero(problem):
    """The Shapley dummy property: a feature the trees ignore gets no credit."""
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)

    assert np.all(explanation.contributions[:, NAMES.index("constant")] == 0.0)


def test_the_label_driving_feature_dominates_globally(problem):
    model, X, _ = problem
    ranking = global_importance(explain_tree_model(model, X, NAMES))

    assert ranking[0].feature == "signal"
    assert ranking[0].share > 0.8
    assert ranking[-1].feature == "constant"
    assert ranking[-1].mean_abs == 0.0


def test_explained_margin_is_the_thresholded_probability(problem):
    """sigmoid(margin) must equal the score the detector thresholds."""
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    probabilities = model.predict_proba(X)[:, 1]

    assert np.allclose(1.0 / (1.0 + np.exp(-explanation.margins)), probabilities, atol=1e-6)
    assert np.allclose(model_margins(model, X), explanation.margins)


def test_signed_contribution_follows_the_feature_direction(problem):
    """High signal values push toward the positive class, low ones away."""
    model, X, _ = problem
    phi = explain_tree_model(model, X, NAMES).contributions[:, 0]

    assert phi[X[:, 0] > 0.8].mean() > 0.0
    assert phi[X[:, 0] < 0.3].mean() < 0.0


# ----------------------------------------------------------------------
# 2. Determinism and label separation
# ----------------------------------------------------------------------


def test_explanations_are_byte_identical_across_calls(problem):
    model, X, _ = problem
    first = explain_tree_model(model, X, NAMES)
    second = explain_tree_model(model, X, NAMES)

    assert np.array_equal(first.contributions, second.contributions)
    assert np.array_equal(first.bias, second.bias)


def test_explainer_signature_admits_no_labels():
    parameters = set(inspect.signature(explain_tree_model).parameters)
    assert parameters == {"model", "X", "feature_names"}
    for forbidden in ("y", "labels", "y_true", "ground_truth"):
        assert forbidden not in parameters


def executable_source(module) -> str:
    raw = inspect.getsource(module)
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(raw).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept).lower()


def test_module_never_touches_redteam_ground_truth():
    code = executable_source(explainability)
    assert "redteam" not in code
    assert "label_for_event" not in code
    assert "ground_truth" not in code


def test_only_the_outcome_grouping_reads_labels():
    """Labels may group finished explanations; no other routine accepts them."""
    for function in (
        explain_tree_model,
        global_importance,
        top_contributions,
        deletion_check,
        train_reference_values,
        model_margins,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"y", "y_true", "labels"}, function.__name__
    assert "y_true" in inspect.signature(outcome_indices).parameters


# ----------------------------------------------------------------------
# 3. Global and local summaries
# ----------------------------------------------------------------------


def test_global_importance_is_sorted_and_shares_sum_to_one(problem):
    model, X, _ = problem
    ranking = global_importance(explain_tree_model(model, X, NAMES))
    values = [item.mean_abs for item in ranking]

    assert values == sorted(values, reverse=True)
    assert sum(item.share for item in ranking) == pytest.approx(1.0)
    assert all(0.0 <= item.nonzero_fraction <= 1.0 for item in ranking)


def test_global_importance_refuses_zero_events(problem):
    model, _, _ = problem
    empty = explain_tree_model(model, np.empty((0, 4)), NAMES)
    with pytest.raises(ExplainabilityError, match="zero events"):
        global_importance(empty)


def test_top_contributions_report_values_and_magnitude_order(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    local = top_contributions(explanation, X, row=0, k=4)

    magnitudes = [abs(item.contribution) for item in local]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert {item.feature for item in local} == set(NAMES)
    assert local[0].value == pytest.approx(X[0, NAMES.index(local[0].feature)])


def test_top_contributions_validate_their_inputs(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    with pytest.raises(ExplainabilityError, match="out of range"):
        top_contributions(explanation, X, row=400)
    with pytest.raises(ExplainabilityError, match="positive"):
        top_contributions(explanation, X, row=0, k=0)
    with pytest.raises(ExplainabilityError, match="row counts"):
        top_contributions(explanation, X[:10], row=0)


def test_rank_correlation_behaviour():
    a = {"x": 3.0, "y": 2.0, "z": 1.0}
    assert rank_correlation(a, a) == pytest.approx(1.0)
    assert rank_correlation(a, {"x": 1.0, "y": 2.0, "z": 3.0}) == pytest.approx(-1.0)
    assert rank_correlation(a, {"x": 0.0, "y": 0.0, "z": 0.0}) is None
    with pytest.raises(ExplainabilityError, match="different features"):
        rank_correlation(a, {"x": 1.0, "y": 2.0})


# ----------------------------------------------------------------------
# 4. Faithfulness (deletion check)
# ----------------------------------------------------------------------


def test_deletion_removes_more_score_through_top_features(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    flagged = np.flatnonzero(explanation.margins > 0)
    reference = train_reference_values(X)

    result = deletion_check(
        model, X[flagged], explanation.subset(flagged), reference, k=1
    )

    assert result.rows == flagged.size > 0
    assert result.mean_drop_top > result.mean_drop_random
    assert result.mean_drop_top > result.mean_drop_bottom
    assert result.top_exceeds_random_fraction > 0.5


def test_neutralising_a_zero_attribution_feature_changes_nothing(problem):
    """Bottom-1 is the constant column; replacing it with its median is a no-op."""
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    result = deletion_check(
        model, X, explanation, train_reference_values(X), k=1
    )
    assert result.mean_drop_bottom == 0.0


def test_deletion_check_is_reproducible_for_a_fixed_seed(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    reference = train_reference_values(X)
    first = deletion_check(model, X, explanation, reference, k=2, seed=7)
    second = deletion_check(model, X, explanation, reference, k=2, seed=7)
    assert first == second


def test_deletion_check_validates_its_inputs(problem):
    model, X, _ = problem
    explanation = explain_tree_model(model, X, NAMES)
    reference = train_reference_values(X)
    with pytest.raises(ExplainabilityError, match="k must be"):
        deletion_check(model, X, explanation, reference, k=5)
    with pytest.raises(ExplainabilityError, match="reference must"):
        deletion_check(model, X, explanation, reference[:2], k=1)
    with pytest.raises(ExplainabilityError, match="row counts"):
        deletion_check(model, X[:5], explanation, reference, k=1)


def test_reference_values_are_train_medians_and_refuse_empty_input():
    X_train = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])
    assert train_reference_values(X_train).tolist() == [2.0, 20.0]
    with pytest.raises(ExplainabilityError, match="non-empty"):
        train_reference_values(np.empty((0, 2)))


# ----------------------------------------------------------------------
# 5. Outcome grouping
# ----------------------------------------------------------------------


def test_outcome_indices_agree_with_the_m40_confusion_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.5, 0.5])
    groups = outcome_indices(y, scores, threshold=0.5)
    metrics = evaluate_at_threshold(y, scores, 0.5)

    assert set(groups) == set(OUTCOME_NAMES)
    assert groups["tp"].tolist() == [0, 4]
    assert groups["fp"].tolist() == [2, 5]
    assert groups["fn"].tolist() == [1]
    assert groups["tn"].tolist() == [3]
    assert (groups["tp"].size, groups["fp"].size) == (metrics.tp, metrics.fp)
    assert (groups["fn"].size, groups["tn"].size) == (metrics.fn, metrics.tn)


def test_outcome_indices_reject_misaligned_input():
    with pytest.raises(ExplainabilityError, match="align"):
        outcome_indices([1, 0], [0.5], threshold=0.5)


# ----------------------------------------------------------------------
# 6. Input validation
# ----------------------------------------------------------------------


def test_feature_name_count_must_match_the_model(problem):
    model, X, _ = problem
    with pytest.raises(ExplainabilityError, match="expects 4 features"):
        explain_tree_model(model, X, NAMES[:3])


def test_feature_matrix_width_must_match(problem):
    model, X, _ = problem
    with pytest.raises(ExplainabilityError, match="feature columns"):
        explain_tree_model(model, X[:, :3], NAMES)


def test_empty_matrix_yields_an_empty_but_valid_explanation(problem):
    model, _, _ = problem
    explanation = explain_tree_model(model, np.empty((0, 4)), NAMES)

    assert explanation.num_rows == 0
    assert explanation.contributions.shape == (0, 4)
    assert explanation.local_accuracy_error() == 0.0


# ----------------------------------------------------------------------
# 7. Integration with the real 17-feature M4.0 pipeline
# ----------------------------------------------------------------------


def make_event(event_id, timestamp, user, source_host, destination_host, success):
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
    """Chronological dataset with a periodic unusual-movement pattern."""
    events, positives = [], []
    for index in range(80):
        attack = index % 7 == 3
        event = make_event(
            f"evt-{index}",
            float(index),
            "UserB" if attack else "UserA",
            "HostX" if attack else "HostA",
            f"Host{index}" if attack else "HostB",
            not attack,
        )
        events.append(event)
        if attack:
            positives.append(event)
    ground_truth = RedteamGroundTruth(
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
    return MLDataset.from_events(events, ground_truth)


def test_attribution_covers_all_17_m34_features_of_an_m40_model():
    dataset = tiny_dataset()
    X_train, y_train = split_matrix(dataset.get_split(SPLIT_TRAIN))
    model = train_xgboost(
        X_train,
        y_train,
        config=XGBoostBaselineConfig(n_estimators=10, max_depth=3, n_jobs=1),
    )
    names = dataset.feature_names
    explanation = explain_tree_model(model, X_train, names)

    assert names == EventFeatures.feature_names()
    assert explanation.contributions.shape == (len(X_train), 17)
    assert explanation.local_accuracy_error() < 1e-4
    scores = predict_scores(model, X_train)
    assert np.allclose(1.0 / (1.0 + np.exp(-explanation.margins)), scores, atol=1e-6)
