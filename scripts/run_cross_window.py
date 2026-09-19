#!/usr/bin/env python3
"""Frozen W1 -> W2 generalization of the selected detector (M4.6).

Steps
-----
1. Select W1 and W2 with the existing M3.6 rule and verify they match the
   recorded definitions; require W2 to be read entirely after W1.
2. Build W1, fit the standard M4.0 XGBoost on W1 TRAIN, select the threshold on
   W1 VALIDATION, freeze both. Check W1 test reproduces PROJECT_STATE section 14.
   W1 is then released from memory before W2 is opened.
3. Build W2 and evaluate the frozen detector on every emitted W2 event. The W2
   built here is the corrected full window (PROJECT_STATE section 33); the
   superseded truncated-W2 figures are printed for comparison only.
4. Compare W1 and W2 side by side.
5. Test the M4.5 hypothesis (section 30.8): attacker identity overlap, fan-out
   profiles of W1 training attacks versus W2 attacks, per-source-host detection,
   and TreeSHAP explanations of what the frozen model saw in W2.

W2 is never used to fit, tune, re-threshold or select anything. W2 labels are
used only to compute metrics and to group finished results. Nothing is written
to disk.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/run_cross_window.py
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.baselines.xgboost_baseline import (  # noqa: E402
    evaluate_at_threshold,
    split_matrix,
)
from ml.evaluation.cross_window import (  # noqa: E402
    assert_generalization_windows,
    evaluate_frozen,
    evaluation_window,
    fit_frozen_detector,
    positive_source_hosts,
    quantile_summary,
)
from ml.evaluation.explainability import (  # noqa: E402
    explain_tree_model,
    outcome_indices,
    top_contributions,
)
from ml.evaluation.redteam import RedteamGroundTruth  # noqa: E402
from ml.preprocessing.ml_dataset import SPLIT_TEST, SPLIT_TRAIN  # noqa: E402
from ml.preprocessing.ml_window import (  # noqa: E402
    build_window_dataset,
    identity_overlap,
    redteam_identity_profile,
    redteam_records_in_window,
    select_densest_redteam_window,
)

FANOUT = "source_unique_destination_count"

RECORDED_W1 = (760506.0, 764106.0, 771306.0)
RECORDED_W2 = (1067648.0, 1071248.0, 1078448.0)
#: Corrected W2 extent, after the M4.7 truncation fix (PROJECT_STATE section 33).
RECORDED_W2_EMITTED = 2_174_232
RECORDED_W2_POSITIVES = 92
RECORDED_M40_TEST = {"threshold": 0.015060, "tp": 11, "fp": 3, "tn": 200822, "fn": 0, "pr_auc": 0.9517}
#: The superseded M4.1/M4.6 W2 result, measured on the TRUNCATED W2 (1,832,857
#: of 2,174,232 events; 76 of 92 positives). Printed for comparison only.
TRUNCATED_W2_RESULT = {"tp": 36, "fp": 430, "tn": 1832351, "fn": 40, "pr_auc": 0.0843, "roc_auc": 0.9970}


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def matches(metrics, recorded: dict, *, threshold: float | None = None) -> bool:
    ok = all(getattr(metrics, key) == recorded[key] for key in ("tp", "fp", "tn", "fn"))
    ok = ok and abs(metrics.average_precision - recorded["pr_auc"]) < 5e-5
    if "roc_auc" in recorded:
        ok = ok and abs(metrics.roc_auc - recorded["roc_auc"]) < 5e-5
    if threshold is not None:
        ok = ok and abs(threshold - recorded["threshold"]) < 5e-6
    return ok


def print_profiles(profiles: dict) -> None:
    print(
        f"      {'group':<24}{'n':>9}{'min':>8}{'p25':>8}{'median':>8}"
        f"{'p75':>8}{'p99':>8}{'max':>8}"
    )
    for name, summary in profiles.items():
        cells = "".join(
            f"{fmt(summary[key], 0):>8}"
            for key in ("min", "p25", "median", "p75", "p99", "max")
        )
        print(f"      {name:<24}{summary['n']:>9}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data" / "raw" / "lanl"
    )
    args = parser.parse_args()
    auth_path = args.data_dir / "auth.txt.gz"
    redteam_path = args.data_dir / "redteam.txt.gz"
    for path in (auth_path, redteam_path):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    run_started = time.time()
    print("=" * 78)
    print("M4.6 FROZEN CROSS-WINDOW GENERALIZATION (fit W1 -> evaluate W2)")
    print("=" * 78)

    # ------------------------------------------------------------------
    print("\n[1] Window selection (M3.6 rule, redteam.txt.gz only) ...")
    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    w1 = select_densest_redteam_window(ground_truth)
    w2 = select_densest_redteam_window(ground_truth, exclude=[w1])
    for name, spec, recorded in (("W1", w1, RECORDED_W1), ("W2", w2, RECORDED_W2)):
        actual = (spec.context_start, spec.emit_start, spec.emit_end)
        print(
            f"    {name}: context={spec.context_start:.0f} "
            f"emit=[{spec.emit_start:.0f}, {spec.emit_end:.0f}] "
            f"redteam_records={spec.redteam_records_in_window}  "
            f"matches recorded: {'YES' if actual == recorded else 'NO'}"
        )
    assert_generalization_windows(w1, w2)
    print("    W2 read region starts strictly after W1 emission ends: YES")

    # ------------------------------------------------------------------
    print("\n[2] W1: fit on TRAIN, threshold on VALIDATION, freeze ...")
    started = time.time()
    window1 = build_window_dataset(auth_path, redteam_path, spec=w1)
    dataset1 = window1.dataset
    print(f"    W1 built in {time.time() - started:.1f}s, rss={rss_mb():.0f} MB")

    started = time.time()
    detector = fit_frozen_detector(dataset1)
    print(
        f"    fitted in {time.time() - started:.1f}s on {detector.train_rows} TRAIN "
        f"events ({detector.train_positives} positive); threshold "
        f"{detector.threshold:.6f} from {detector.validation_rows} VALIDATION "
        f"events ({detector.validation_positives} positive)"
    )
    print(f"    frozen model digest sha256={detector.digest[:16]}...")

    names = detector.feature_names
    fan = names.index(FANOUT)
    X_test, y_test = split_matrix(dataset1.get_split(SPLIT_TEST))
    w1_scores = detector.scores(X_test)
    w1_metrics = evaluate_at_threshold(y_test, w1_scores, detector.threshold)
    print(
        "    W1 test reproduces PROJECT_STATE section 14: "
        + ("YES" if matches(w1_metrics, RECORDED_M40_TEST, threshold=detector.threshold) else "NO")
    )

    X_train, y_train = split_matrix(dataset1.get_split(SPLIT_TRAIN))
    profiles = {
        "W1 train positives": quantile_summary(X_train[y_train == 1, fan]),
        "W1 train negatives": quantile_summary(X_train[y_train == 0, fan]),
        "W1 test positives": quantile_summary(X_test[y_test == 1, fan]),
    }
    boundaries = dataset1.split_boundaries()
    train_end = boundaries["train_end"]
    w1_test_hours = (w1.emit_end - boundaries["validation_end"]) / 3600.0
    w1_test_rows = int(X_test.shape[0])
    w1_test_positives = int(np.count_nonzero(y_test))

    del window1, dataset1, X_train, y_train, X_test, y_test, w1_scores
    gc.collect()
    print(f"    W1 released before W2 is opened; rss={rss_mb():.0f} MB")

    # ------------------------------------------------------------------
    print("\n[3] W2: frozen evaluation on every emitted event ...")
    started = time.time()
    window2 = build_window_dataset(auth_path, redteam_path, spec=w2)
    w2_records_in_range = window2.redteam_records_in_range
    evaluation = evaluation_window(window2.dataset)
    del window2
    gc.collect()
    print(f"    W2 built in {time.time() - started:.1f}s, rss={rss_mb():.0f} MB")
    print(
        f"    events={evaluation.rows} (recorded {RECORDED_W2_EMITTED}) "
        f"positives={evaluation.positives} (recorded {RECORDED_W2_POSITIVES}) "
        f"redteam records in range={w2_records_in_range}"
    )

    w2_metrics, w2_scores = evaluate_frozen(detector, evaluation)
    print(
        "    superseded truncated-W2 result (sections 16 and 31.3): "
        f"TP={TRUNCATED_W2_RESULT['tp']} FP={TRUNCATED_W2_RESULT['fp']} "
        f"FN={TRUNCATED_W2_RESULT['fn']} "
        f"PR-AUC={TRUNCATED_W2_RESULT['pr_auc']} -- measured on 1,832,857 of "
        "2,174,232 events, so it is not comparable with the figures below"
    )

    # ------------------------------------------------------------------
    print("\n[4] W1 versus W2 at the frozen threshold ...")
    w2_hours = (w2.emit_end - w2.emit_start) / 3600.0
    print(
        f"    {'window':<16}{'events':>10}{'pos':>5}{'PR-AUC':>9}{'ROC-AUC':>10}"
        f"{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>6}{'FN':>5}{'pred+':>7}{'FP/h':>8}"
    )
    for label, metrics, rows, positives, hours in (
        ("W1 test", w1_metrics, w1_test_rows, w1_test_positives, w1_test_hours),
        ("W2 emitted", w2_metrics, evaluation.rows, evaluation.positives, w2_hours),
    ):
        print(
            f"    {label:<16}{rows:>10}{positives:>5}{fmt(metrics.average_precision):>9}"
            f"{fmt(metrics.roc_auc, 6):>10}{metrics.precision:>8.4f}"
            f"{metrics.recall:>8.4f}{metrics.f1:>8.4f}{metrics.tp:>5}{metrics.fp:>6}"
            f"{metrics.fn:>5}{metrics.predicted_positive:>7}{metrics.fp / hours:>8.1f}"
        )
    print(f"    W2 TN={w2_metrics.tn}  W1 test TN={w1_metrics.tn}")

    # ------------------------------------------------------------------
    print("\n[5] Hypothesis test: attacker identity overlap (redteam.txt.gz) ...")
    records1 = redteam_records_in_window(ground_truth, w1)
    records2 = redteam_records_in_window(ground_truth, w2)
    train_hosts = {
        record.source_host
        for record in ground_truth.filter_by_timestamp(w1.emit_start, train_end)
    }
    profile1 = redteam_identity_profile(records1)
    profile2 = redteam_identity_profile(records2)
    for category, comparison in identity_overlap(profile1, profile2).items():
        shared = comparison["shared_values"]
        shown = shared[:12] + (["..."] if len(shared) > 12 else [])
        print(
            f"    {category:<18} W1={comparison['first_only'] + comparison['shared']:>3} "
            f"W2={comparison['second_only'] + comparison['shared']:>3} "
            f"shared={comparison['shared']:>3} jaccard={comparison['jaccard']:.3f} "
            f"{shown}"
        )
    w2_host_records = defaultdict(int)
    for record in records2:
        w2_host_records[record.source_host] += 1
    print(f"    W2 redteam records by source host: {dict(sorted(w2_host_records.items()))}")
    print(f"    W1 TRAIN-period attacker source hosts: {sorted(train_hosts)}")

    # ------------------------------------------------------------------
    print(f"\n[6] Hypothesis test: fan-out profile ({FANOUT}) ...")
    groups = outcome_indices(evaluation.y, w2_scores, detector.threshold)
    X2 = evaluation.X
    profiles.update(
        {
            "W2 positives": quantile_summary(X2[evaluation.y == 1, fan]),
            "W2 TP": quantile_summary(X2[groups["tp"], fan]),
            "W2 FN": quantile_summary(X2[groups["fn"], fan]),
            "W2 FP": quantile_summary(X2[groups["fp"], fan]),
            "W2 negatives": quantile_summary(X2[evaluation.y == 0, fan]),
        }
    )
    print_profiles(profiles)

    # ------------------------------------------------------------------
    print("\n[7] Per-source-host detection of W2 positives ...")
    positive_rows = np.flatnonzero(evaluation.y == 1)
    hosts = positive_source_hosts(evaluation.timestamps[positive_rows], records2)
    detected = w2_scores[positive_rows] >= detector.threshold
    by_host: dict[str, list[int]] = defaultdict(list)
    for position, host in enumerate(hosts):
        by_host[host if host is not None else "<ambiguous>"].append(position)
    print(
        f"    {'source host':<14}{'pos':>5}{'TP':>5}{'FN':>5}"
        f"{'median fan-out':>16}{'median score':>14}  in W1 TRAIN"
    )
    for host, positions in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
        rows = positive_rows[positions]
        hits = int(detected[positions].sum())
        print(
            f"    {host:<14}{len(positions):>5}{hits:>5}{len(positions) - hits:>5}"
            f"{np.median(X2[rows, fan]):>16.0f}{np.median(w2_scores[rows]):>14.4f}  "
            f"{'yes' if host in train_hosts else 'no'}"
        )

    # ------------------------------------------------------------------
    print("\n[8] TreeSHAP of the frozen model on W2 positives and false positives ...")
    explanation = explain_tree_model(detector.model, X2[positive_rows], names)
    print(f"    local accuracy max error: {explanation.local_accuracy_error():.2e}")
    tp_mean = explanation.contributions[detected].mean(axis=0) if detected.any() else np.zeros(len(names))
    fn_mean = explanation.contributions[~detected].mean(axis=0) if (~detected).any() else np.zeros(len(names))
    fp_explanation = explain_tree_model(detector.model, X2[groups["fp"]], names)
    fp_mean = fp_explanation.contributions.mean(axis=0) if fp_explanation.num_rows else np.zeros(len(names))
    print(f"      {'feature':<38}{'W2 TP':>9}{'W2 FN':>9}{'W2 FP':>9}")
    for index in np.argsort(-(tp_mean - fn_mean), kind="stable"):
        print(
            f"      {names[index]:<38}{tp_mean[index]:>9.3f}"
            f"{fn_mean[index]:>9.3f}{fp_mean[index]:>9.3f}"
        )

    print("\n    Missed W2 attacks (FN), top-3 contributions in log-odds:")
    for position in np.flatnonzero(~detected):
        row = int(positive_rows[position])
        local = top_contributions(explanation, X2[positive_rows], int(position), k=3)
        detail = ", ".join(
            f"{item.feature}={item.value:g} ({item.contribution:+.2f})" for item in local
        )
        print(
            f"      t={evaluation.timestamps[row]:.0f} host={hosts[position]} "
            f"score={w2_scores[row]:.5f}  {detail}"
        )

    # ------------------------------------------------------------------
    detector.verify_unchanged()
    print(f"\n    frozen model digest unchanged at the end of the run: YES")
    print(f"    total runtime {time.time() - run_started:.1f}s, peak rss={rss_mb():.0f} MB")
    print("\nDone. Nothing was written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
