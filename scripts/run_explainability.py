#!/usr/bin/env python3
"""Real-LANL explainability run on the M3.6 W1 window (M4.5).

Explains the model selected from the M4.4 comparison -- the standard M4.0
XGBoost, chosen on VALIDATION PR-AUC -- with exact TreeSHAP, then validates the
attributions on real data.

Steps
-----
1. Rebuild W1 with the existing M3.6 rule and M3.5 chronological split.
2. Retrain the standard M4.0 XGBoost exactly as M4.0 did, select the threshold on
   validation, and check the test result reproduces the recorded M4.0 figures.
3. Report validation PR-AUC for the selection (XGBoost computed here; the M4.4
   neural variants quoted from PROJECT_STATE.md section 28.5).
4. Compute TreeSHAP contributions for validation and test. Label-free.
5. Validate: local accuracy, decision consistency, val/test ranking stability,
   agreement with XGBoost gain importance, and a deletion check that uses
   TRAIN-only median reference values.
6. Print local explanations for every flagged test event and every missed one.

W2 is never read. Nothing is written to disk.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/run_explainability.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.baselines.xgboost_baseline import (  # noqa: E402
    XGBoostBaselineConfig,
    compute_scale_pos_weight,
    dataset_matrices,
    evaluate_at_threshold,
    feature_importance_ranking,
    predict_scores,
    select_threshold_on_validation,
    train_xgboost,
)
from ml.evaluation.explainability import (  # noqa: E402
    deletion_check,
    explain_tree_model,
    global_importance,
    outcome_indices,
    rank_correlation,
    top_contributions,
    train_reference_values,
)
from ml.preprocessing.ml_dataset import (  # noqa: E402
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
)
from ml.preprocessing.ml_window import build_window_dataset  # noqa: E402

#: M4.0 standard XGBoost, W1 test, as recorded in PROJECT_STATE.md section 14.
RECORDED_M40 = {
    "threshold": 0.015060,
    "tp": 11,
    "fp": 3,
    "tn": 200822,
    "fn": 0,
    "pr_auc": 0.9517,
}

#: Validation PR-AUC at the selected epoch for the M4.4 neural variants, quoted
#: from PROJECT_STATE.md section 28.5 (deterministic run).
RECORDED_M44_VALIDATION_PR_AUC = {
    "gnn:behavioral_only": 0.9261,
    "gnn:hybrid": 0.7350,
    "gnn:graph_only": 0.1404,
}


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data" / "raw" / "lanl"
    )
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def show_ranking(title: str, ranking) -> None:
    print(f"\n    {title}")
    print(
        f"      {'rank':>4}  {'feature':<38}{'mean|phi|':>11}"
        f"{'share':>8}{'mean phi':>11}{'nonzero':>9}"
    )
    for position, item in enumerate(ranking, start=1):
        print(
            f"      {position:>4}  {item.feature:<38}{item.mean_abs:>11.5f}"
            f"{item.share:>8.3f}{item.mean_signed:>11.5f}"
            f"{item.nonzero_fraction:>9.3f}"
        )


def main() -> int:
    args = parse_args()
    auth_path = args.data_dir / "auth.txt.gz"
    redteam_path = args.data_dir / "redteam.txt.gz"
    for path in (auth_path, redteam_path):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    print("=" * 78)
    print("M4.5 REAL-LANL EXPLAINABILITY (W1 only; W2 is never read)")
    print("=" * 78)

    print("\n[1] Building W1 with the existing M3.6 rule ...")
    started = time.time()
    window = build_window_dataset(auth_path, redteam_path)
    dataset = window.dataset
    print(f"    built in {time.time() - started:.1f}s, rss={rss_mb():.0f} MB")
    counts = dataset.split_label_counts()
    for name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        print(
            f"    {name}: n={len(dataset.get_split(name))} "
            f"positive={counts[name]['positive']}"
        )

    matrices = dataset_matrices(dataset)
    X_train, y_train = matrices[SPLIT_TRAIN]
    X_val, y_val = matrices[SPLIT_VALIDATION]
    X_test, y_test = matrices[SPLIT_TEST]
    names = dataset.feature_names

    print("\n[2] Retraining M4.0 XGBoost (standard and class-weighted) ...")
    config = XGBoostBaselineConfig()
    started = time.time()
    model = train_xgboost(X_train, y_train, config=config)
    weighted = train_xgboost(
        X_train,
        y_train,
        config=config,
        scale_pos_weight=compute_scale_pos_weight(y_train),
    )
    print(f"    trained both in {time.time() - started:.1f}s")

    val_scores = predict_scores(model, X_val)
    threshold = select_threshold_on_validation(y_val, val_scores)
    val_metrics = evaluate_at_threshold(y_val, val_scores, threshold)
    weighted_val = evaluate_at_threshold(
        y_val, predict_scores(weighted, X_val), 0.5
    )

    test_scores = predict_scores(model, X_test)
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)
    reproduced = (
        abs(threshold - RECORDED_M40["threshold"]) < 5e-6
        and test_metrics.tp == RECORDED_M40["tp"]
        and test_metrics.fp == RECORDED_M40["fp"]
        and test_metrics.tn == RECORDED_M40["tn"]
        and test_metrics.fn == RECORDED_M40["fn"]
        and abs(test_metrics.average_precision - RECORDED_M40["pr_auc"]) < 5e-5
    )
    print(
        f"    standard: threshold={threshold:.6f} (validation-selected) "
        f"test TP={test_metrics.tp} FP={test_metrics.fp} "
        f"TN={test_metrics.tn} FN={test_metrics.fn} "
        f"PR-AUC={test_metrics.average_precision:.6f}"
    )
    print(
        "    reproduces recorded M4.0 result: "
        + ("YES" if reproduced else "NO -- investigate before trusting attributions")
    )

    print("\n[3] Model selection on VALIDATION PR-AUC (test not consulted) ...")
    candidates = {
        "xgboost:standard": val_metrics.average_precision,
        "xgboost:class_weighted": weighted_val.average_precision,
        **RECORDED_M44_VALIDATION_PR_AUC,
    }
    for name, value in sorted(candidates.items(), key=lambda kv: -kv[1]):
        source = "this run" if name.startswith("xgboost") else "PROJECT_STATE 28.5"
        print(f"    {name:<26} validation PR-AUC={value:.4f}   ({source})")
    selected = max(candidates, key=candidates.get)
    print(f"    selected: {selected}")
    if selected != "xgboost:standard":
        print("    ERROR: selection differs from the model explained below")
        return 1

    print("\n[4] Exact TreeSHAP (label-free) on validation and test ...")
    started = time.time()
    val_expl = explain_tree_model(model, X_val, names)
    test_expl = explain_tree_model(model, X_test, names)
    print(f"    {val_expl.num_rows + test_expl.num_rows} events in {time.time() - started:.1f}s")

    print("\n[5] Attribution validity checks ...")
    print(
        f"    local accuracy max |sum(phi)+bias-margin|: "
        f"validation={val_expl.local_accuracy_error():.2e} "
        f"test={test_expl.local_accuracy_error():.2e}"
    )
    consistency = float(
        np.max(np.abs(1.0 / (1.0 + np.exp(-test_expl.margins)) - test_scores))
    )
    print(f"    decision consistency max |sigmoid(margin)-score| (test): {consistency:.2e}")
    bias_values = np.unique(np.round(test_expl.bias, 6))
    print(f"    bias (expected log-odds over training cover): {bias_values.tolist()}")

    test_ranking = global_importance(test_expl)
    val_ranking = global_importance(val_expl)
    show_ranking("Global importance, TEST (all 200,836 events)", test_ranking)

    test_map = {item.feature: item.mean_abs for item in test_ranking}
    val_map = {item.feature: item.mean_abs for item in val_ranking}
    gain_map = dict(feature_importance_ranking(model, names))
    print(
        f"\n    Spearman(validation, test) mean|phi|: "
        f"{rank_correlation(val_map, test_map):.4f}"
    )
    gain_rho = rank_correlation(gain_map, test_map)
    print(
        f"    Spearman(XGBoost gain, TreeSHAP test): "
        f"{'n/a' if gain_rho is None else f'{gain_rho:.4f}'}"
    )
    print("    gain ranking (M4.0 built-in, for comparison):")
    for position, (feature, gain) in enumerate(
        sorted(gain_map.items(), key=lambda kv: (-kv[1], kv[0])), start=1
    ):
        print(f"      {position:>4}  {feature:<38}{gain:>11.5f}")

    groups = outcome_indices(y_test, test_scores, threshold)
    positives = np.flatnonzero(y_test == 1)
    negatives = np.flatnonzero(y_test == 0)
    print(
        "\n    Mean phi by class, TEST (labels used only to group finished "
        "attributions):"
    )
    print(f"      {'feature':<38}{'positives':>11}{'negatives':>11}")
    pos_mean = test_expl.contributions[positives].mean(axis=0)
    neg_mean = test_expl.contributions[negatives].mean(axis=0)
    for index in np.argsort(-(pos_mean - neg_mean), kind="stable"):
        print(f"      {names[index]:<38}{pos_mean[index]:>11.4f}{neg_mean[index]:>11.4f}")

    print("\n    Deletion check (TRAIN-median reference; flagged events only):")
    reference = train_reference_values(X_train)
    val_flagged = np.flatnonzero(val_scores >= threshold)
    test_flagged = np.flatnonzero(test_scores >= threshold)
    for label, X_split, expl, rows in (
        ("validation", X_val, val_expl, val_flagged),
        ("test", X_test, test_expl, test_flagged),
    ):
        for k in (1, 2, 3):
            result = deletion_check(
                model, X_split[rows], expl.subset(rows), reference, k=k, seed=0
            )
            print(
                f"      {label:<10} k={k} rows={result.rows:>3}  "
                f"drop top={result.mean_drop_top:>8.3f}  "
                f"random={result.mean_drop_random:>8.3f}  "
                f"bottom={result.mean_drop_bottom:>8.3f}  "
                f"top>random in {result.top_exceeds_random_fraction:.0%}"
            )

    print("\n[6] Local explanations at the frozen threshold (log-odds) ...")
    split = dataset.get_split(SPLIT_TEST)
    for outcome in ("tp", "fp", "fn"):
        rows = groups[outcome]
        print(f"\n    {outcome.upper()} ({rows.size} events)")
        for row in rows:
            local = top_contributions(test_expl, X_test, int(row), k=args.top_k)
            detail = ", ".join(
                f"{item.feature}={item.value:g} ({item.contribution:+.2f})"
                for item in local
            )
            print(
                f"      t={split.timestamps[row]:.0f} {split.event_ids[row]} "
                f"score={test_scores[row]:.4f} margin={test_expl.margins[row]:+.2f}"
            )
            print(f"        {detail}")

    print(f"\n    peak rss={rss_mb():.0f} MB")
    print("\nDone. Nothing was written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
