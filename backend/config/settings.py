"""Paths and server settings for the dashboard backend."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Read-only research data (RESEARCH_CONSTRAINTS section 21).
LANL_DIR = PROJECT_ROOT / "data" / "raw" / "lanl"
AUTH_PATH = LANL_DIR / "auth.txt.gz"
REDTEAM_PATH = LANL_DIR / "redteam.txt.gz"

#: Derived dashboard dataset. ``data/processed/*`` is gitignored, and every file
#: here can be regenerated with ``scripts/build_dashboard_data.py``.
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"

#: The API binds to the loopback interface only: this is a local research tool.
HOST = "127.0.0.1"
PORT = 8000

#: Origins allowed to call the API from a browser (the Next.js dev server).
ALLOWED_ORIGINS = frozenset({"http://localhost:3000", "http://127.0.0.1:3000"})
