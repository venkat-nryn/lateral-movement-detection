#!/usr/bin/env python3
"""Real-LANL smoke test for the M4.2 graph representation.

Streams a bounded slice of the existing M3.6 real-LANL window, converts it to
the M4.2 tensor representation and checks the representation only. It trains
nothing, writes nothing to disk and never processes the full dataset: reading
stops after ``--max-events`` events inside the window, and the window itself is
the bounded one selected from ``redteam.txt.gz`` in M3.6.

Redteam ground truth is used here solely to locate the window (the M3.6 rule);
no label enters the graph.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/smoke_test_graph_data.py
    .\\.venv\\Scripts\\python.exe scripts/smoke_test_graph_data.py --max-events 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.evaluation.redteam import RedteamGroundTruth  # noqa: E402
from ml.models.graph_data import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_graph_snapshot,
    pyg_available,
    torch_available,
)
from ml.preprocessing.ml_window import (  # noqa: E402
    iter_window_events,
    select_densest_redteam_window,
)

#: Bounded by default. 50k events produce 100k edges, roughly 4 MB of tensors.
DEFAULT_MAX_EVENTS = 50_000

#: Refuse to touch the GPU unless the snapshot is a tiny fraction of the 4 GB
#: card, leaving room for the CUDA context itself.
GPU_BYTE_LIMIT = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"events read from inside the window (default {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw" / "lanl",
        help="directory holding auth.txt.gz and redteam.txt.gz",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also move the snapshot to CUDA if it fits well inside 4 GB",
    )
    return parser.parse_args()


def collect_window_events(auth_path: Path, spec, max_events: int) -> list:
    """Read at most ``max_events`` emitted events of the window."""
    events = []
    scanned = 0
    started = time.time()
    for event in iter_window_events(auth_path, spec):
        scanned += 1
        if event.timestamp < spec.emit_start:
            continue  # context region: history only, not part of this slice
        events.append(event)
        if len(events) >= max_events:
            break
    elapsed = time.time() - started
    print(
        f"[2] read {len(events)} emitted events "
        f"({scanned} records scanned) in {elapsed:.1f}s"
    )
    return events


def check_finite(name: str, array: np.ndarray) -> None:
    if array.size and not np.isfinite(array).all():
        raise SystemExit(f"FAIL: {name} contains NaN or Inf")


def main() -> int:
    args = parse_args()
    auth_path = args.data_dir / "auth.txt.gz"
    redteam_path = args.data_dir / "redteam.txt.gz"

    for path in (auth_path, redteam_path):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    print("=" * 72)
    print("M4.2 REAL-LANL GRAPH REPRESENTATION SMOKE TEST")
    print("=" * 72)

    print("\n[1] Selecting the M3.6 real window from redteam.txt.gz ...")
    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    spec = select_densest_redteam_window(ground_truth)
    print(
        f"    W1 emit=[{spec.emit_start:.0f}, {spec.emit_end:.0f}] "
        f"context_start={spec.context_start:.0f} "
        f"redteam_records={spec.redteam_records_in_window}"
    )

    print(f"\n[2] Streaming at most {args.max_events} events of the window ...")
    events = collect_window_events(auth_path, spec, args.max_events)
    if not events:
        print("ERROR: no events read from the window")
        return 1

    print("\n[3] Building the CPU snapshot ...")
    started = time.time()
    snapshot = build_graph_snapshot(events)
    build_seconds = time.time() - started
    summary = snapshot.summary()
    for key, value in summary.items():
        print(f"    {key}: {value}")
    print(f"    build_seconds: {build_seconds:.2f}")
    print(f"    cpu_bytes: {snapshot.nbytes()} ({snapshot.nbytes() / 1e6:.1f} MB)")

    print("\n[4] Tensor dimension and numeric checks ...")
    n, e, m = snapshot.num_nodes, snapshot.num_edges, snapshot.num_events
    assert snapshot.x.shape == (n, len(NODE_FEATURE_NAMES))
    assert snapshot.edge_index.shape == (2, e)
    assert snapshot.edge_attr.shape == (e, len(EDGE_FEATURE_NAMES))
    assert e == 2 * m
    for name in ("x", "edge_attr", "edge_time", "event_timestamps"):
        check_finite(name, getattr(snapshot, name))
    snapshot.validate()
    print(f"    x={snapshot.x.shape} edge_index={snapshot.edge_index.shape} "
          f"edge_attr={snapshot.edge_attr.shape}")
    print("    no NaN/Inf; validate() passed")

    print("\n[5] Temporal checks on real data ...")
    if np.any(np.diff(snapshot.edge_time) < 0):
        print("FAIL: edge times are not ordered")
        return 1
    midpoint = events[len(events) // 2]
    bounded = build_graph_snapshot(events, cutoff_timestamp=midpoint.timestamp)
    if bounded.num_edges and float(bounded.edge_time.max()) >= midpoint.timestamp:
        print("FAIL: a future edge survived the cutoff")
        return 1
    prefix = [ev for ev in events if ev.timestamp < midpoint.timestamp]
    rebuilt = build_graph_snapshot(prefix, cutoff_timestamp=midpoint.timestamp)
    identical = (
        bounded.node_ids == rebuilt.node_ids
        and np.array_equal(bounded.x, rebuilt.x)
        and np.array_equal(bounded.edge_index, rebuilt.edge_index)
        and np.array_equal(bounded.edge_attr, rebuilt.edge_attr)
    )
    if not identical:
        print("FAIL: future events changed the pre-cutoff representation")
        return 1
    print(f"    edge_time non-decreasing over {e} real edges")
    print(
        f"    cutoff={midpoint.timestamp:.0f} -> {bounded.num_events} events kept, "
        f"{bounded.excluded_future_events} future events excluded"
    )
    print("    dropping the future changed nothing before the cutoff")

    print("\n[6] Deterministic conversion ...")
    again = build_graph_snapshot(events)
    if not (
        again.node_ids == snapshot.node_ids
        and np.array_equal(again.x, snapshot.x)
        and np.array_equal(again.edge_index, snapshot.edge_index)
        and np.array_equal(again.edge_attr, snapshot.edge_attr)
    ):
        print("FAIL: repeated conversion produced different tensors")
        return 1
    print("    two builds of the same events are byte-identical")

    print("\n[7] Tensor library conversion ...")
    print(f"    torch installed: {torch_available()}")
    print(f"    torch_geometric installed: {pyg_available()}")
    if torch_available():
        import torch

        tensors = snapshot.to_torch()
        print(
            f"    cpu torch: x={tuple(tensors['x'].shape)} "
            f"edge_index={tuple(tensors['edge_index'].shape)} "
            f"edge_attr={tuple(tensors['edge_attr'].shape)}"
        )
        if pyg_available():
            data = snapshot.to_pyg_data()
            print(
                f"    pyg Data: num_nodes={data.num_nodes} "
                f"num_edges={data.num_edges}"
            )
        if args.gpu:
            if not torch.cuda.is_available():
                print("    gpu: CUDA not available; skipped")
            elif snapshot.nbytes() > GPU_BYTE_LIMIT:
                print(
                    f"    gpu: snapshot is {snapshot.nbytes() / 1e6:.1f} MB, "
                    "above the safety limit for a 4 GB card; skipped"
                )
            else:
                cuda_tensors = snapshot.to_torch(device="cuda")
                allocated = torch.cuda.memory_allocated() / 1e6
                print(
                    f"    gpu: moved to CUDA, "
                    f"x={tuple(cuda_tensors['x'].shape)}, "
                    f"allocated={allocated:.1f} MB"
                )
                del cuda_tensors
                torch.cuda.empty_cache()
        else:
            print("    gpu: not requested (pass --gpu)")
    else:
        print("    skipped: PyTorch is not installed; numpy path verified above")

    print("\n" + "=" * 72)
    print("SMOKE TEST PASSED (representation only; nothing was trained)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
