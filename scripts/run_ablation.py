#!/usr/bin/env python3
"""Information-source ablation on W1 with frozen W2 evaluation (M4.7).

Steps
-----
1. Select W1 and W2 with the M3.6 rule; require W2 to be read after W1.
2. Build W1 graph/behavioural batches once and fit the behavioural scaler on
   W1 TRAIN only.
3. Train every ablation variant with every seed on W1 TRAIN, selecting the
   epoch and threshold on W1 VALIDATION. Deterministic CUDA is on by default
   (PROJECT_STATE section 29). Seed 0 of the M4.4 variants must reproduce
   section 28.5.
4. Release W1. Only then build W2, relabel it as evaluation data, and score
   every frozen model on every emitted W2 event.
5. Report per-model results, per-variant seed summaries, and the marginal value
   of each information source on W1 test and W2.

XGBoost is quoted from PROJECT_STATE section 35.2 (the corrected full W2), not
retrained: it is deterministic and is reproduced exactly from code by
scripts/run_cross_window.py.

Nothing is tuned: every variant uses the M4.4 configuration unchanged, and seeds
estimate variance rather than being searched over. W2 is never used to fit,
select or threshold anything. Nothing is written to disk.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/run_ablation.py
    .\\.venv\\Scripts\\python.exe scripts/run_ablation.py --seeds 0 --variants target_only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.evaluation.ablation import (  # noqa: E402
    as_evaluation_batches,
    evaluate_trained_variant,
    marginal_contributions,
    seed_summary,
    train_variant,
)
from ml.evaluation.cross_window import assert_generalization_windows  # noqa: E402
from ml.evaluation.redteam import RedteamGroundTruth  # noqa: E402
from ml.models.gnn import (  # noqa: E402
    ABLATION_VARIANTS,
    VARIANT_INPUTS,
    BatchedEventDataset,
    BehavioralScaler,
    GNNExperimentConfig,
    enable_deterministic_cuda,
    iter_event_batches,
    resolve_device,
)
from ml.preprocessing.ml_dataset import (  # noqa: E402
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    build_label_index,
)
from ml.preprocessing.ml_window import (  # noqa: E402
    iter_window_events,
    select_densest_redteam_window,
)

#: Seed-0 W1 test PR-AUC from the deterministic M4.4 run (section 28.5).
RECORDED_M44_SEED0 = {"behavioral_only": 0.908838, "hybrid": 0.844415, "graph_only": 0.140273}

#: Standard XGBoost reference on the corrected full W2, quoted from section
#: 35.2. Section 31.3's figures (0.0843 / 0.4737 / 215) were measured on a
#: truncated W2 and are not comparable with this script's full-W2 results.
RECORDED_XGBOOST = {
    "w1_test_pr_auc": 0.9517,
    "w2_pr_auc": 0.1257,
    "w2_recall": 0.4891,
    "w2_fp_per_hour": 253.0,
}

RECORDED_W1_SPLITS = {
    SPLIT_TRAIN: (892_424, 69),
    SPLIT_VALIDATION: (186_504, 15),
    SPLIT_TEST: (200_836, 11),
}
#: Corrected full W2 extent (PROJECT_STATE section 35).
RECORDED_W2 = (2_174_232, 92)

W1_TEST_HOURS = 1080.0 / 3600.0
W2_HOURS = 7200.0 / 3600.0


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def inputs_code(variant: str) -> str:
    graph, behavioural, target = VARIANT_INPUTS[variant]
    return ("G" if graph else "-") + ("B" if behavioural else "-") + ("T" if target else "-")


def build_batches(auth_path, spec, label_index, config) -> BatchedEventDataset:
    return BatchedEventDataset.from_batches(
        iter_event_batches(
            iter_window_events(auth_path, spec),
            label_index,
            timestamp_range=spec.timestamp_range(),
            emit_from_timestamp=spec.emit_start,
            config=config,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default=",".join(ABLATION_VARIANTS))
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="disable deterministic CUDA (results will vary run to run)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data" / "raw" / "lanl"
    )
    args = parser.parse_args()

    if not args.nondeterministic:
        enable_deterministic_cuda()
    import torch

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_INPUTS]
    if unknown or not seeds:
        print(f"ERROR: unknown variants {unknown} or no seeds")
        return 1

    auth_path = args.data_dir / "auth.txt.gz"
    redteam_path = args.data_dir / "redteam.txt.gz"
    for path in (auth_path, redteam_path):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    run_started = time.time()
    device = resolve_device(args.device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print("=" * 78)
    print("M4.7 INFORMATION-SOURCE ABLATION (train W1 -> frozen W1 test and W2)")
    print("=" * 78)
    print(
        f"    device={device} deterministic={not args.nondeterministic} "
        f"seeds={seeds} epochs={args.epochs}"
    )
    print(f"    variants (G=graph, B=17 behavioural, T=7 target attributes):")
    for variant in variants:
        print(f"      {inputs_code(variant)}  {variant}")

    # ------------------------------------------------------------------
    print("\n[1] Windows ...")
    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    w1 = select_densest_redteam_window(ground_truth)
    w2 = select_densest_redteam_window(ground_truth, exclude=[w1])
    assert_generalization_windows(w1, w2)
    label_index = build_label_index(ground_truth)
    base_config = GNNExperimentConfig(epochs=args.epochs)
    print(
        f"    W1 emit=[{w1.emit_start:.0f}, {w1.emit_end:.0f}]  "
        f"W2 emit=[{w2.emit_start:.0f}, {w2.emit_end:.0f}]  W2 read after W1: YES"
    )

    # ------------------------------------------------------------------
    print("\n[2] W1 batches and TRAIN-only behavioural scaler ...")
    started = time.time()
    w1_data = build_batches(auth_path, w1, label_index, base_config)
    counts = w1_data.counts()
    print(f"    built {len(w1_data.batches)} batches in {time.time() - started:.1f}s, rss={rss_mb():.0f} MB")
    for name, (events, positives) in RECORDED_W1_SPLITS.items():
        actual = (counts[name]["events"], counts[name]["positive"])
        print(
            f"    {name:<10} events={actual[0]} positive={actual[1]}  "
            f"matches recorded: {'YES' if actual == (events, positives) else 'NO'}"
        )
    scaler = BehavioralScaler().fit(w1_data.split(SPLIT_TRAIN))
    frozen_mean, frozen_scale = scaler.mean.copy(), scaler.scale.copy()
    print(f"    scaler fitted on {scaler.fitted_on_events} W1 TRAIN events")

    # ------------------------------------------------------------------
    print("\n[3] Training every variant and seed on W1 ...")
    trained_models = []
    for variant in variants:
        for seed in seeds:
            config = GNNExperimentConfig(epochs=args.epochs, seed=seed)
            trained = train_variant(
                w1_data, scaler, variant=variant, config=config, device=device
            )
            trained_models.append(trained)
            test = trained.test_metrics
            print(
                f"    {variant:<22}seed={seed} params={trained.parameters:>6} "
                f"epoch={trained.best_epoch:>2} "
                f"val_pr_auc={fmt(trained.best_validation_average_precision)} "
                f"thr={trained.threshold:.6f} "
                f"W1 test pr_auc={fmt(test.average_precision, 6)} "
                f"TP={test.tp} FP={test.fp} FN={test.fn} "
                f"({trained.training_seconds:.0f}s)"
            )

    print("\n    Seed-0 reproduction of section 28.5 (deterministic M4.4 run):")
    for trained in trained_models:
        if trained.seed == 0 and trained.variant in RECORDED_M44_SEED0:
            recorded = RECORDED_M44_SEED0[trained.variant]
            actual = trained.test_metrics.average_precision
            same = actual is not None and abs(actual - recorded) < 5e-6
            print(
                f"      {trained.variant:<18} recorded={recorded:.6f} "
                f"now={fmt(actual, 6)}  {'YES' if same else 'NO'}"
            )

    del w1_data
    gc.collect()
    print(f"    W1 released before W2 is built; rss={rss_mb():.0f} MB")

    # ------------------------------------------------------------------
    print("\n[4] W2 batches (evaluation only) ...")
    started = time.time()
    w2_data = build_batches(auth_path, w2, label_index, base_config)
    evaluation = as_evaluation_batches(w2_data.batches)
    del w2_data
    gc.collect()
    w2_events = sum(batch.num_targets for batch in evaluation)
    w2_positives = sum(batch.positive_count() for batch in evaluation)
    print(
        f"    built {len(evaluation)} batches in {time.time() - started:.1f}s, "
        f"rss={rss_mb():.0f} MB; events={w2_events} positives={w2_positives}  "
        f"matches recorded: {'YES' if (w2_events, w2_positives) == RECORDED_W2 else 'NO'}"
    )

    print("\n[5] Frozen evaluation of every trained model on W2 ...")
    w2_results = {}
    for trained in trained_models:
        config = GNNExperimentConfig(epochs=args.epochs, seed=trained.seed)
        metrics, _ = evaluate_trained_variant(
            trained, evaluation, scaler, config=config, device=device
        )
        w2_results[(trained.variant, trained.seed)] = metrics
        print(
            f"    {trained.variant:<22}seed={trained.seed} "
            f"W2 pr_auc={fmt(metrics.average_precision, 6)} "
            f"P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f} "
            f"TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} "
            f"pred+={metrics.predicted_positive} FP/h={metrics.fp / W2_HOURS:.1f}"
        )

    # ------------------------------------------------------------------
    print("\n[6] Per-variant summary over seeds: mean [min, max] ...")
    print(
        f"    {'variant':<22}{'in':>4}{'n':>3}  {'W1 test PR-AUC':<26}"
        f"{'W2 PR-AUC':<26}{'W2 recall':>10}{'W2 FP/h':>9}"
    )

    def cell(summary) -> str:
        return (
            f"{fmt(summary['mean'])} [{fmt(summary['min'])}, {fmt(summary['max'])}]"
        )

    w1_means, w2_means = {}, {}
    for variant in variants:
        rows = [t for t in trained_models if t.variant == variant]
        w1_summary = seed_summary([t.test_metrics.average_precision for t in rows])
        w2_summary = seed_summary(
            [w2_results[(t.variant, t.seed)].average_precision for t in rows]
        )
        recall = seed_summary([w2_results[(t.variant, t.seed)].recall for t in rows])
        fp_rate = seed_summary(
            [w2_results[(t.variant, t.seed)].fp / W2_HOURS for t in rows]
        )
        w1_means[variant], w2_means[variant] = w1_summary["mean"], w2_summary["mean"]
        print(
            f"    {variant:<22}{inputs_code(variant):>4}{w1_summary['n']:>3}  "
            f"{cell(w1_summary):<26}{cell(w2_summary):<26}"
            f"{fmt(recall['mean']):>10}{fmt(fp_rate['mean'], 1):>9}"
        )
    print(
        f"    {'xgboost:standard (35.2)':<22}{'-B-':>4}{'':>3}  "
        f"{RECORDED_XGBOOST['w1_test_pr_auc']:<26.4f}{RECORDED_XGBOOST['w2_pr_auc']:<26.4f}"
        f"{RECORDED_XGBOOST['w2_recall']:>10.4f}{RECORDED_XGBOOST['w2_fp_per_hour']:>9.1f}"
    )

    print("\n[7] Marginal value of each information source (difference of seed means) ...")
    print(f"    {'contribution':<30}{'W1 test PR-AUC':>16}{'W2 PR-AUC':>12}")
    w2_marginals = dict(marginal_contributions(w2_means))
    for label, w1_delta in marginal_contributions(w1_means):
        w2_delta = w2_marginals[label]
        print(
            f"    {label:<30}"
            f"{('n/a' if w1_delta is None else f'{w1_delta:+.4f}'):>16}"
            f"{('n/a' if w2_delta is None else f'{w2_delta:+.4f}'):>12}"
        )

    # ------------------------------------------------------------------
    unchanged = np.array_equal(scaler.mean, frozen_mean) and np.array_equal(
        scaler.scale, frozen_scale
    )
    print(f"\n    W1 TRAIN scaler unchanged at end of run: {'YES' if unchanged else 'NO'}")
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0
    print(
        f"    total runtime {time.time() - run_started:.0f}s, "
        f"peak rss={rss_mb():.0f} MB, peak VRAM={peak_vram:.0f} MB"
    )
    print("\nDone. Nothing was written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
