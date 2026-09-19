#!/usr/bin/env python3
"""Real LANL data evaluation of temporal behavior baseline detector (M3.3).

Streams a bounded prefix of auth.txt.gz, evaluates predictions against
redteam.txt.gz ground truth, and reports metrics.

The prefix defaults to the M3.5 development bound (1,000,000 events).
Processing the complete file requires the explicit ``--full-dataset`` flag,
matching the project rule that a full-dataset run must never happen by
accident.

Does NOT use redteam labels in detector feature calculation.
Maintains only required state (never loads full datasets into RAM).
"""

import sys
import time
import os
import psutil
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.baselines.temporal_behavior import TemporalBehaviorDetector
from ml.evaluation.redteam import RedteamGroundTruth
from ml.preprocessing.lanl_adapter import iter_lanl_events
from ml.preprocessing.ml_dataset import DEVELOPMENT_MAX_EVENTS
from ml.preprocessing.schema import CanonicalEvent


def evaluate_baseline(limit=DEVELOPMENT_MAX_EVENTS, verbose=True):
    """Evaluate baseline detector on auth.txt.gz against redteam ground truth.

    Args:
        limit: Maximum number of auth events to process. Defaults to the
            M3.5 development bound; pass ``None`` only for a deliberate
            complete-file research run.
        verbose: If True, print progress information.

    Returns:
        Dictionary of metrics.
    """
    auth_gz = project_root / "data" / "raw" / "lanl" / "auth.txt.gz"
    redteam_gz = project_root / "data" / "raw" / "lanl" / "redteam.txt.gz"

    if not auth_gz.exists():
        print(f"ERROR: {auth_gz} not found")
        return None

    if not redteam_gz.exists():
        print(f"ERROR: {redteam_gz} not found")
        return None

    if verbose:
        print(f"\n{'='*80}")
        print("LANL BASELINE DETECTOR EVALUATION")
        print(f"{'='*80}\n")

        if limit:
            print(f"[1] Loading redteam ground truth (first {limit} auth events)...")
        else:
            print(f"[1] Loading redteam ground truth (full auth file)...")

    # Load redteam ground truth
    try:
        gt = RedteamGroundTruth.from_file(redteam_gz)
        if verbose:
            print(f"[OK] Loaded {gt.number_of_records()} redteam records")
    except Exception as e:
        print(f"ERROR loading redteam: {e}")
        return None

    # Exact-match index: (timestamp, user, source_host, destination_host) ->
    # every redteam record with that key. Duplicate records are kept, so the
    # lookup finds exactly the records matches_canonical_event would find when
    # scanning all 749 records, in O(1) per event instead of O(records).
    redteam_by_key = defaultdict(list)
    for redteam_record in gt.iter_records():
        redteam_by_key[
            (
                redteam_record.timestamp,
                redteam_record.user,
                redteam_record.source_host,
                redteam_record.destination_host,
            )
        ].append(redteam_record)

    # Create detector
    detector = TemporalBehaviorDetector()

    if verbose:
        print(f"\n[2] Processing authentication events...")

    # Process auth events
    events_processed = 0
    events_rejected = 0
    start_time = time.time()

    # Track matches between predictions and redteam
    matched_redteam: set[str] = set()
    seen_redteam: set[str] = set()
    tp_predictions = 0
    fp_predictions = 0

    # Feature statistics
    feature_stats = {
        "new_user_destination": 0,
        "new_source_destination": 0,
        "source_differs_destination": 0,
        "rapid_user_movement": 0,
        "failed_authentication": 0,
    }

    # Memory tracking
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    try:
        for event in iter_lanl_events(auth_gz, limit=limit):
            events_processed += 1

            # Process event with detector
            try:
                prediction = detector.process_event(event)

                if prediction.features.new_user_destination:
                    feature_stats["new_user_destination"] += 1
                if prediction.features.new_source_destination:
                    feature_stats["new_source_destination"] += 1
                if prediction.features.source_differs_destination:
                    feature_stats["source_differs_destination"] += 1
                if prediction.features.rapid_user_movement:
                    feature_stats["rapid_user_movement"] += 1
                if prediction.features.failed_authentication:
                    feature_stats["failed_authentication"] += 1

                # Check if this prediction matches any redteam record
                is_redteam = False
                key = (
                    event.timestamp,
                    event.user,
                    event.source_host,
                    event.destination_host,
                )
                for redteam_record in redteam_by_key.get(key, ()):
                    is_redteam = True
                    seen_redteam.add(redteam_record.redteam_id)
                    if prediction.suspicious:
                        matched_redteam.add(redteam_record.redteam_id)
                
                if prediction.suspicious:
                    if is_redteam:
                        tp_predictions += 1
                    else:
                        fp_predictions += 1
            except Exception as e:
                if verbose and events_rejected < 5:
                    print(f"  Warning: Failed to process event: {e}")
                events_rejected += 1
                continue

            # Progress indicator
            if verbose and events_processed % max(1, limit // 10 if limit else 1000000) == 0:
                elapsed = time.time() - start_time
                rate = events_processed / elapsed if elapsed > 0 else 0
                current_rss_mb = process.memory_info().rss / (1024 * 1024)
                if verbose:
                    print(
                        f"  Processed {events_processed} events "
                        f"({rate:.0f} events/sec) - RSS: {current_rss_mb:.1f} MB",
                        flush=True
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return None
    except Exception as e:
        print(f"ERROR processing events: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed_time = time.time() - start_time
    rss_after_mb = process.memory_info().rss / (1024 * 1024)
    rss_increase_mb = rss_after_mb - rss_before_mb

    if verbose:
        print(f"[OK] Processed {events_processed} events in {elapsed_time:.1f}s")

    # Compile results
    suspicious_predictions = detector.number_of_suspicious_predictions()
    redteam_records = gt.number_of_records()
    redteam_records_seen = len(seen_redteam)

    # Metrics calculation
    # TP (Records): unique redteam ground-truth records detected
    tp_records = len(matched_redteam)

    # FP: suspicious predictions that were NOT redteam events
    fp = fp_predictions

    # FN: redteam records in ground truth that were NOT detected
    fn = redteam_records - tp_records

    # Precision: out of suspicious predictions, how many were true positives?
    precision = tp_predictions / (tp_predictions + fp) if (tp_predictions + fp) > 0 else 0.0
    
    # Recall: out of all ground truth records, how many were detected?
    recall = tp_records / redteam_records if redteam_records > 0 else 0.0
    
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    detection_rate = tp_records / redteam_records if redteam_records > 0 else 0.0

    # Note: TN is undefined in this context (we don't know what "normal" events are)

    # Compile metrics dictionary
    metrics = {
        "events_processed": events_processed,
        "events_rejected": events_rejected,
        "predicted_suspicious": suspicious_predictions,
        "redteam_records": redteam_records,
        "tp": tp_records,
        "tp_predictions": tp_predictions,
        "fp": fp,
        "fn": fn,
        "redteam_records_seen": redteam_records_seen,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_rate": detection_rate,
        "elapsed_time": elapsed_time,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_increase_mb": rss_increase_mb,
        "feature_stats": feature_stats,
        "state_size": detector.get_state_size_estimate(),
    }

    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}\n")

        print(f"Events processed: {events_processed}")
        if events_rejected > 0:
            print(f"Events rejected: {events_rejected}")

        print(f"\nPredicted suspicious: {suspicious_predictions}")
        print(f"Redteam ground-truth records: {redteam_records}")
        print(f"Redteam records encountered in stream: {redteam_records_seen}")

        print(f"\nTrue Positives (Records): {tp_records}")
        print(f"False Positives (Predictions): {fp}")
        print(f"False Negatives (Records missed from total GT): {fn}")
        
        if redteam_records_seen == 0:
            print(f"\nNOTE: 0 TP/Recall is expected because no redteam timestamps occurred in the processed events.")

        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print(f"\nRedteam events detected: {tp_records}/{redteam_records}")
        print(f"Redteam detection rate: {detection_rate:.2%}")

        print(f"\nRuntime: {elapsed_time:.1f} seconds")
        if elapsed_time > 0:
            print(f"Throughput: {events_processed / elapsed_time:.0f} events/second")

        print(f"\nMemory Profile:")
        print(f"  RSS Before: {rss_before_mb:.1f} MB")
        print(f"  RSS After:  {rss_after_mb:.1f} MB")
        print(f"  RSS Increase: {rss_increase_mb:.1f} MB")

        print(f"\nFeature occurrences:")
        for feature, count in feature_stats.items():
            pct = 100.0 * count / events_processed if events_processed > 0 else 0
            print(f"  {feature}: {count} ({pct:.1f}%)")

        print(f"\nDetector state size:")
        for key, size in detector.get_state_size_estimate().items():
            print(f"  {key}: {size}")

        print(f"\n{'='*80}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate temporal behavior baseline on LANL data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEVELOPMENT_MAX_EVENTS,
        help=(
            "process only the first N events (default: "
            f"{DEVELOPMENT_MAX_EVENTS:,}, the M3.5 development bound)"
        ),
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help=(
            "process the COMPLETE auth.txt.gz and ignore --limit; reserved for "
            "deliberate research runs"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with 10,000 events first",
    )

    args = parser.parse_args()
    if not args.full_dataset and args.limit <= 0:
        parser.error("--limit must be positive (use --full-dataset for a full run)")
    run_limit = None if args.full_dataset else args.limit

    if args.smoke_test:
        print("\n" + "="*80)
        print("SMOKE TEST: 10,000 EVENTS")
        print("="*80)
        metrics_smoke = evaluate_baseline(limit=10000, verbose=True)

        if metrics_smoke is None:
            print("Smoke test failed")
            sys.exit(1)

        print(f"\n[OK] Smoke test succeeded")
        print(f"  Precision: {metrics_smoke['precision']:.4f}")
        print(f"  Recall: {metrics_smoke['recall']:.4f}")
        print(f"  F1: {metrics_smoke['f1']:.4f}")

        scope = "the complete file" if run_limit is None else f"{run_limit:,} events"
        print(f"\nProceeding to the main run ({scope})...\n")

    metrics = evaluate_baseline(limit=run_limit, verbose=True)

    if metrics is None:
        sys.exit(1)

    # Exit with success
    sys.exit(0)
