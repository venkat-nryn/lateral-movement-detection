#!/usr/bin/env python3
"""Start the SOC dashboard API on http://127.0.0.1:8000.

Requires the export from ``scripts/build_dashboard_data.py``. On start-up the
store re-scores a sample of stored rows with the loaded model and refuses to
serve if any score differs.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/run_dashboard_api.py
    .\\.venv\\Scripts\\python.exe scripts/run_dashboard_api.py --speed 20 --paused
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aiohttp import web  # noqa: E402

from backend.api.server import create_app  # noqa: E402
from backend.config.settings import EXPORT_DIR, HOST, PORT  # noqa: E402
from backend.services.replay import SPEEDS, ReplayClock  # noqa: E402
from backend.services.store import DashboardStore, StoreError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--speed", type=float, default=1.0, choices=SPEEDS)
    parser.add_argument("--paused", action="store_true", help="start with the replay paused")
    args = parser.parse_args()

    started = time.time()
    try:
        store = DashboardStore(EXPORT_DIR)
    except StoreError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"Loaded {store.rows:,} W2 events, {store.alert_rows.size} alerts in "
        f"{time.time() - started:.1f}s. Integrity: {store.integrity.rows_checked} rows "
        f"re-scored identically; model digest match: {store.integrity.digest_matches}."
    )
    clock = ReplayClock(store.start, store.end, speed=args.speed, playing=not args.paused)
    print(f"Serving on http://{HOST}:{args.port} (replay speed {args.speed:g}x)")
    web.run_app(create_app(store, clock=clock), host=HOST, port=args.port, access_log=None, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
