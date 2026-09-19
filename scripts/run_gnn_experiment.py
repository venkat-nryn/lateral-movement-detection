#!/usr/bin/env python3
"""Bounded real-LANL experiment on the M3.6 W1 window (M4.3 / M4.4).

Runs every model over the *same* bounded real window and reports them together:

1. the existing M4.0 XGBoost baseline on the 17 M3.4 tabular features;
2. ``graph_only``      -- the M4.3 temporal GAT on the M4.2 graph representation;
3. ``behavioral_only`` -- the M4.4 no-graph control, same head, 17 features only;
4. ``hybrid``          -- the M4.4 model, graph embeddings + the 17 features.

The three neural variants share one preprocessing pass, one device choice and
one TRAIN-fitted behavioural scaler, so the only thing that differs between them
is which branches the network has. That is what makes the M4.4 question --
"do graph embeddings add anything to the 17 features?" -- answerable: compare
``hybrid`` against ``behavioral_only``, not against XGBoost, which would
confound the representation with the model family.

All models use the existing chronological W1 train/validation/test split, the
existing exact 4-field redteam labels, and the M4.0 evaluation philosophy: the
threshold is chosen on validation and then frozen for test. W2 is never read.

Nothing is written to disk: no model file, no dataset dump, no report file.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/run_gnn_experiment.py
    .\\.venv\\Scripts\\python.exe scripts/run_gnn_experiment.py --device cuda
    .\\.venv\\Scripts\\python.exe scripts/run_gnn_experiment.py --skip-xgboost
    .\\.venv\\Scripts\\python.exe scripts/run_gnn_experiment.py --variants hybrid
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

from ml.baselines.xgboost_baseline import (  # noqa: E402
    DEFAULT_THRESHOLD,
    MODEL_CLASS_WEIGHTED,
    MODEL_STANDARD,
    evaluate_at_threshold,
    run_baseline,
    select_threshold_on_validation,
)
from ml.evaluation.redteam import RedteamGroundTruth  # noqa: E402
from ml.models.gnn import (  # noqa: E402
    MODEL_VARIANTS,
    BatchedEventDataset,
    BehavioralScaler,
    GNNExperimentConfig,
    build_model,
    count_parameters,
    enable_deterministic_cuda,
    estimate_device_bytes,
    iter_event_batches,
    predict_scores,
    resolve_device,
    train_model,
)
from ml.preprocessing.ml_dataset import (  # noqa: E402
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    build_label_index,
)
from ml.preprocessing.ml_window import (  # noqa: E402
    build_window_dataset,
    iter_window_events,
    select_densest_redteam_window,
)

#: Refuse to move a batch to a 4 GB card if the estimate exceeds this.
GPU_BYTE_LIMIT = 1_500_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--max-history-events", type=int, default=20_000)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=400)
    parser.add_argument(
        "--variants",
        default=",".join(MODEL_VARIANTS),
        help="comma-separated subset of " + ",".join(MODEL_VARIANTS),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "force bitwise-reproducible CUDA kernels. The GAT scatter "
            "reduction is otherwise order-dependent on GPU, which makes the "
            "graph-bearing variants vary run to run"
        ),
    )
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data" / "raw" / "lanl"
    )
    return parser.parse_args()


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def print_metrics(title: str, metrics) -> None:
    d = metrics.as_dict()
    print(f"    {title}")
    print(
        f"      threshold={d['threshold']:.6f} "
        f"predicted_positive={d['predicted_positive']}"
    )
    print(
        f"      TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']} "
        f"(support +{d['support_positive']} / -{d['support_negative']})"
    )
    ap = d["average_precision"]
    roc = d["roc_auc"]
    print(
        f"      precision={d['precision']:.6f} recall={d['recall']:.6f} "
        f"f1={d['f1']:.6f}"
    )
    print(
        f"      pr_auc={'n/a' if ap is None else f'{ap:.6f}'} "
        f"roc_auc={'n/a' if roc is None else f'{roc:.6f}'}"
    )


def run_xgboost(auth_path: Path, redteam_path: Path) -> dict:
    """The unmodified M4.0 baseline on the same W1 window."""
    print("\n[A] M4.0 XGBoost baseline on W1 ...")
    started = time.time()
    window = build_window_dataset(auth_path, redteam_path)
    build_seconds = time.time() - started
    dataset = window.dataset

    counts = dataset.split_label_counts()
    print(f"    window build: {build_seconds:.1f}s, rss={rss_mb():.0f} MB")
    print(f"    events={dataset.events_processed} " f"context={dataset.context_events_processed}")
    for name in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST):
        print(
            f"    {name}: n={len(dataset.get_split(name))} "
            f"positive={counts[name]['positive']}"
        )

    train_started = time.time()
    reports = run_baseline(dataset)
    train_seconds = time.time() - train_started

    result = {"reports": {}, "train_seconds": train_seconds}
    for name, report in reports.items():
        print(f"\n    --- {name} (train {report.train_seconds:.1f}s) ---")
        print_metrics(
            "validation @ selected", report.metric(SPLIT_VALIDATION, "selected")
        )
        print_metrics("test @ selected", report.metric(SPLIT_TEST, "selected"))
        print_metrics("test @ 0.50", report.metric(SPLIT_TEST, "default"))
        result["reports"][name] = {
            "validation_selected": report.metric(
                SPLIT_VALIDATION, "selected"
            ).as_dict(),
            "test_selected": report.metric(SPLIT_TEST, "selected").as_dict(),
            "test_default": report.metric(SPLIT_TEST, "default").as_dict(),
            "threshold": report.selected_threshold,
        }

    del window, dataset, reports
    gc.collect()
    return result


def build_gnn_dataset(
    auth_path: Path, redteam_path: Path, config: GNNExperimentConfig
) -> tuple[BatchedEventDataset, float]:
    print("\n[B] Building GNN temporal batches from W1 ...")
    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    spec = select_densest_redteam_window(ground_truth)
    label_index = build_label_index(ground_truth)
    print(
        f"    W1 emit=[{spec.emit_start:.0f}, {spec.emit_end:.0f}] "
        f"redteam_records_in_window={spec.redteam_records_in_window}"
    )

    started = time.time()
    dataset = BatchedEventDataset.from_batches(
        iter_event_batches(
            iter_window_events(auth_path, spec),
            label_index,
            timestamp_range=spec.timestamp_range(),
            emit_from_timestamp=spec.emit_start,
            config=config,
        )
    )
    seconds = time.time() - started

    summary = dataset.summary()
    print(f"    built {summary['batches']} batches in {seconds:.1f}s")
    print(
        f"    cpu_bytes={summary['cpu_bytes'] / 1e6:.1f} MB "
        f"peak_batch_nodes={summary['peak_batch_nodes']} "
        f"peak_batch_edges={summary['peak_batch_edges']}"
    )
    for name, counts in summary["counts"].items():
        print(
            f"    {name}: batches={counts['batches']} events={counts['events']} "
            f"positive={counts['positive']}"
        )
    print(f"    rss={rss_mb():.0f} MB")
    return dataset, seconds


def resolve_gnn_device(
    dataset: BatchedEventDataset, args: argparse.Namespace
) -> str:
    """Pick the device once, applying the 4 GB safety limit."""
    device = resolve_device(args.device)
    estimate = estimate_device_bytes(dataset, args.hidden_dim, args.heads)
    print(f"\n[C] Device selection: requested={args.device} resolved={device}")
    print(f"    estimated peak device bytes per batch: {estimate / 1e6:.1f} MB")
    if device.startswith("cuda") and estimate > GPU_BYTE_LIMIT:
        print("    estimate above the 4 GB safety limit; falling back to CPU")
        device = "cpu"
    return device


def fit_scaler(dataset: BatchedEventDataset) -> BehavioralScaler:
    """Fit the behavioural standardiser on TRAIN batches only, then freeze it."""
    scaler = BehavioralScaler().fit(dataset.split(SPLIT_TRAIN))
    print(
        f"    behavioural scaler fitted on {scaler.fitted_on_events} TRAIN "
        f"events only; validation and test are never consulted"
    )
    return scaler


def run_gnn(
    dataset: BatchedEventDataset,
    args: argparse.Namespace,
    *,
    variant: str,
    device: str,
    scaler: BehavioralScaler,
) -> dict:
    import torch

    config = GNNExperimentConfig(
        block_size=args.block_size,
        max_history_events=args.max_history_events,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    print(f"\n--- variant: {variant} ---")
    torch.manual_seed(config.seed)
    model = build_model(config, variant=variant).to(device)
    parameters = count_parameters(model)
    print(f"    parameters: {parameters}")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[D] Training on TRAIN only ({config.epochs} epochs) ...")
    history = train_model(
        model, dataset, config=config, device=device, scaler=scaler, verbose=True
    )
    print(
        f"    training seconds: {history.seconds:.1f} "
        f"best_epoch={history.best_epoch} "
        f"best_validation_pr_auc={history.best_validation_average_precision}"
    )

    peak_device_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0
    )

    print("\n[E] Threshold selection on VALIDATION, frozen for TEST ...")
    y_validation = dataset.labels(SPLIT_VALIDATION)
    validation_scores = predict_scores(
        model, dataset.split(SPLIT_VALIDATION), device=device, scaler=scaler
    )
    threshold = select_threshold_on_validation(y_validation, validation_scores)
    validation_metrics = evaluate_at_threshold(
        y_validation, validation_scores, threshold
    )
    print_metrics("validation @ selected", validation_metrics)

    y_test = dataset.labels(SPLIT_TEST)
    test_scores = predict_scores(
        model, dataset.split(SPLIT_TEST), device=device, scaler=scaler
    )
    test_selected = evaluate_at_threshold(y_test, test_scores, threshold)
    test_default = evaluate_at_threshold(y_test, test_scores, DEFAULT_THRESHOLD)
    print_metrics("test @ selected (frozen)", test_selected)
    print_metrics("test @ 0.50", test_default)

    print(
        f"\n    device={device} peak_device_mb={peak_device_mb:.1f} "
        f"rss={rss_mb():.0f} MB"
    )
    return {
        "variant": variant,
        "device": device,
        "parameters": parameters,
        "training_seconds": history.seconds,
        "best_epoch": history.best_epoch,
        "threshold": threshold,
        "peak_device_mb": peak_device_mb,
        "validation_selected": validation_metrics.as_dict(),
        "test_selected": test_selected.as_dict(),
        "test_default": test_default.as_dict(),
    }


def enable_determinism() -> None:
    """Make CUDA kernels bitwise reproducible.

    Must run before torch initialises cuBLAS, hence the environment variable is
    set here rather than left to the caller. Without this the GAT's scatter
    reduction accumulates in a non-deterministic order on GPU: differences of
    order 1e-9 per epoch compound, and because validation PR-AUC is unstable
    across epochs for the graph-bearing variants, validation-based epoch
    selection can land on a different epoch and change the reported test
    metrics.
    """
    enable_deterministic_cuda()
    print("    deterministic CUDA algorithms enabled")


def main() -> int:
    args = parse_args()
    if args.deterministic:
        enable_determinism()
    auth_path = args.data_dir / "auth.txt.gz"
    redteam_path = args.data_dir / "redteam.txt.gz"
    for path in (auth_path, redteam_path):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    print("=" * 76)
    print("M4.3/M4.4 REAL-LANL GNN EXPERIMENT (W1 only; W2 is never read)")
    print("=" * 76)

    xgboost_result = None
    if not args.skip_xgboost:
        xgboost_result = run_xgboost(auth_path, redteam_path)

    gnn_result = None
    if not args.skip_gnn:
        config = GNNExperimentConfig(
            block_size=args.block_size,
            max_history_events=args.max_history_events,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_batches=args.max_batches,
        )
        dataset, preprocess_seconds = build_gnn_dataset(
            auth_path, redteam_path, config
        )
        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in requested if v not in MODEL_VARIANTS]
        if unknown:
            print(f"ERROR: unknown variant(s) {unknown}; expected {MODEL_VARIANTS}")
            return 1
        device = resolve_gnn_device(dataset, args)
        scaler = fit_scaler(dataset)
        gnn_result = {
            variant: run_gnn(
                dataset, args, variant=variant, device=device, scaler=scaler
            )
            for variant in requested
        }
        for result in gnn_result.values():
            result["preprocess_seconds"] = preprocess_seconds

    print("\n" + "=" * 76)
    print("COMPARISON ON W1 TEST (threshold selected on validation, frozen)")
    print("=" * 76)
    header = f"{'model':<28}{'PR-AUC':>10}{'ROC-AUC':>10}{'P':>10}{'R':>10}{'F1':>10}"
    print(header)

    def row(name: str, metrics: dict) -> None:
        def fmt(value):
            return "n/a" if value is None else f"{value:.6f}"

        print(
            f"{name:<28}{fmt(metrics['average_precision']):>10}"
            f"{fmt(metrics['roc_auc']):>10}{metrics['precision']:>10.6f}"
            f"{metrics['recall']:>10.6f}{metrics['f1']:>10.6f}"
        )

    if xgboost_result:
        for name in (MODEL_STANDARD, MODEL_CLASS_WEIGHTED):
            row(f"xgboost:{name}", xgboost_result["reports"][name]["test_selected"])
    if gnn_result:
        for variant, result in gnn_result.items():
            row(f"gnn:{variant}", result["test_selected"])

    print("\nDone. Nothing was written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
