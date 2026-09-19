#!/usr/bin/env python3
"""Smoke test: real LANL data with TemporalGraph (M3.1).

Tests iter_lanl_events() and build_lanl_graph() against the real
auth.txt.gz file with bounded limits.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.preprocessing.lanl_adapter import build_lanl_graph, iter_lanl_events
from ml.preprocessing.schema import CanonicalEvent


def smoke_test(limit: int) -> None:
    """Run smoke test with real LANL data."""
    auth_gz = project_root / "data" / "raw" / "lanl" / "auth.txt.gz"

    print(f"\n{'='*80}")
    print(f"SMOKE TEST: Real LANL auth.txt.gz with limit={limit}")
    print(f"{'='*80}")

    if not auth_gz.exists():
        print(f"ERROR: {auth_gz} not found")
        sys.exit(1)

    # Test 1: Stream events
    print(f"\n[1] Testing iter_lanl_events()...")
    events_list = []
    try:
        for i, event in enumerate(iter_lanl_events(auth_gz, limit=limit)):
            events_list.append(event)
            if (i + 1) % max(1, limit // 10) == 0 or (i + 1) == limit:
                print(f"  Parsed {i + 1} events...")

        print(f"[OK] Successfully parsed {len(events_list)} events")

        if len(events_list) == 0:
            print("ERROR: No events parsed")
            sys.exit(1)

    except Exception as e:
        print(f"[FAIL] Error parsing events: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 2: Check first 5 events
    print(f"\n[2] First 5 CanonicalEvents:")
    for i, event in enumerate(events_list[:5]):
        print(f"  {i+1}. event_id={event.event_id}")
        print(f"     timestamp={event.timestamp}, user={event.user}")
        print(f"     src={event.source_host} -> dst={event.destination_host}")
        print(f"     success={event.success}")

    # Test 3: Build graph
    print(f"\n[3] Building TemporalGraph...")
    try:
        graph = build_lanl_graph(auth_gz, limit=limit)
        print(f"[OK] Graph built successfully")
    except Exception as e:
        print(f"[FAIL] Error building graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 4: Graph statistics
    print(f"\n[4] Graph Statistics:")
    print(f"  Events: {graph.number_of_events()}")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    # Separate user and host nodes
    user_nodes = 0
    host_nodes = 0
    for node_id, attrs in graph.iter_nodes():
        if node_id.startswith("user:"):
            user_nodes += 1
        elif node_id.startswith("host:"):
            host_nodes += 1

    print(f"  User nodes: {user_nodes}")
    print(f"  Host nodes: {host_nodes}")

    # Test 5: First 10 edges
    print(f"\n[5] First 10 Edges:")
    edges = list(graph.iter_edges())
    for i, (src, dst, key, attrs) in enumerate(edges[:10]):
        print(f"  {i+1}. {src} -> {dst} (event={key[:16]}...)")

    # Test 6: Cardinalities from events
    print(f"\n[6] Cardinalities:")
    users = set()
    sources = set()
    destinations = set()
    success_count = 0
    fail_count = 0

    for event in events_list:
        users.add(event.user)
        sources.add(event.source_host)
        destinations.add(event.destination_host)
        if event.success:
            success_count += 1
        else:
            fail_count += 1

    print(f"  Unique users: {len(users)}")
    print(f"  Unique source hosts: {len(sources)}")
    print(f"  Unique destination hosts: {len(destinations)}")
    print(f"  Successful events: {success_count}")
    print(f"  Failed events: {fail_count}")

    # Test 7: Time range
    if events_list:
        min_ts = min(e.timestamp for e in events_list)
        max_ts = max(e.timestamp for e in events_list)
        print(f"\n[7] Time Range:")
        print(f"  Min timestamp: {min_ts}")
        print(f"  Max timestamp: {max_ts}")
        print(f"  Duration: {max_ts - min_ts}")

    print(f"\n{'='*80}")
    print(f"[OK] SMOKE TEST PASSED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test with small limit first
    print("\n>>> Testing with limit=1000...")
    smoke_test(1000)

    # Test with larger limit
    print("\n>>> Testing with limit=10000...")
    smoke_test(10000)

    print("\n[OK] All smoke tests completed successfully!")
