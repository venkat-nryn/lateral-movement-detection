#!/usr/bin/env python3
"""Build the real-LANL dataset served by the SOC dashboard.

Runs the existing research pipeline, unmodified, over the M3.6 windows:

* fits the frozen M4.0 XGBoost on W1 TRAIN, threshold on W1 VALIDATION;
* scores every emitted W2 event with the frozen model (the M4.6 protocol);
* recovers user/host identities for W2, checked row by row against event ids.

Writes to ``data/processed/dashboard/`` (gitignored). Takes about 12 minutes and
roughly 2 GB of RAM. Reads a bounded region of auth.txt.gz only; the full dataset
is never processed.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/build_dashboard_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config.settings import AUTH_PATH, EXPORT_DIR, REDTEAM_PATH  # noqa: E402
from backend.services.export import build_export  # noqa: E402

#: Recorded results the export must reproduce (PROJECT_STATE sections 14 and 35).
RECORDED = {
    "w1_test": {"tp": 11, "fp": 3, "tn": 200822, "fn": 0, "average_precision": 0.9517},
    "w2": {"tp": 45, "fp": 506, "tn": 2173634, "fn": 47, "average_precision": 0.1257},
}


def main() -> int:
    for path in (AUTH_PATH, REDTEAM_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    print("=" * 72)
    print("SOC DASHBOARD DATA EXPORT (real LANL; W1 fit -> frozen W2 scoring)")
    print("=" * 72)
    manifest = build_export(AUTH_PATH, REDTEAM_PATH, EXPORT_DIR)

    print("\nReproduction check against PROJECT_STATE.md:")
    ok = True
    for key, expected in RECORDED.items():
        actual = manifest["metrics"][key]
        same = all(actual[k] == expected[k] for k in ("tp", "fp", "tn", "fn")) and (
            abs(actual["average_precision"] - expected["average_precision"]) < 5e-5
        )
        ok &= same
        print(
            f"  {key:<8} TP={actual['tp']} FP={actual['fp']} FN={actual['fn']} "
            f"PR-AUC={actual['average_precision']:.4f}  "
            f"{'matches record' if same else 'DOES NOT MATCH RECORD'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
