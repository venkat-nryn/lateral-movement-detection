"""Read-only in-memory store of the exported real-LANL dashboard dataset.

Everything served comes from ``scripts/build_dashboard_data.py``, which runs the
research pipeline unmodified. On load the store re-scores a sample of stored
rows -- every alert plus an even spread of other events -- with the loaded
model, and refuses to serve if a single score differs. The dashboard therefore
cannot silently show numbers the frozen detector did not produce.

Two presentation conventions live here and are labelled as such wherever they
are served:

* **severity** -- fixed bands on the model score (:data:`SEVERITY_BANDS`). The
  model emits a probability, not a severity.
* **triage** -- analyst workflow state, persisted beside the export. It is
  never an input to any model or metric.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from ml.baselines.xgboost_baseline import predict_scores
from ml.evaluation.cross_window import model_digest
from ml.evaluation.explainability import explain_tree_model

#: Presentation bands on the frozen model's score for events at or above the
#: validation-selected threshold. Scores below ``medium`` but at or above the
#: threshold are ``low``.
SEVERITY_BANDS: tuple[tuple[str, float], ...] = (
    ("critical", 0.9),
    ("high", 0.5),
    ("medium", 0.1),
)
SEVERITY_ORDER = ("critical", "high", "medium", "low")

TRIAGE_STATUSES = ("new", "investigating", "escalated", "resolved", "false_positive")

ACTIVITY_BIN_SECONDS = 60.0


class StoreError(RuntimeError):
    """Raised when the export is missing, inconsistent or fails verification."""


def severity_for(score: float, threshold: float) -> str | None:
    """Presentation severity for a score, or ``None`` below the threshold."""
    if score < threshold:
        return None
    for name, cut in SEVERITY_BANDS:
        if score >= cut:
            return name
    return "low"


@dataclass(frozen=True, slots=True)
class Integrity:
    rows_checked: int
    scores_identical: bool
    digest_matches: bool


class DashboardStore:
    """The exported W2 replay dataset, the frozen model and W1 evaluation arrays."""

    def __init__(
        self,
        export_dir: Path | str,
        *,
        triage_path: Path | str | None = None,
        verify_rows: int = 512,
    ) -> None:
        export_dir = Path(export_dir)
        manifest_path = export_dir / "manifest.json"
        if not manifest_path.is_file():
            raise StoreError(
                f"no dashboard export in {export_dir}; run "
                "scripts/build_dashboard_data.py first"
            )
        self.export_dir = export_dir
        self.manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_info = self.manifest["model"]
        self.threshold = float(model_info["threshold"])
        self.feature_names: tuple[str, ...] = tuple(model_info["feature_names"])

        with np.load(export_dir / "w2.npz", allow_pickle=False) as w2:
            self.timestamps = w2["timestamps"].astype(np.float64)
            self.event_ids = w2["event_ids"]
            self.features = w2["features"].astype(np.float64)
            self.labels = w2["labels"].astype(np.int8)
            self.scores = w2["scores"].astype(np.float64)
            self.user = w2["user"].astype(np.int64)
            self.source = w2["source"].astype(np.int64)
            self.destination = w2["destination"].astype(np.int64)
            self.success = w2["success"].astype(bool)
            self.users = [str(value) for value in w2["users"]]
            self.hosts = [str(value) for value in w2["hosts"]]
        with np.load(export_dir / "w1.npz", allow_pickle=False) as w1:
            self.w1 = {key: w1[key] for key in w1.files}

        rows = self.timestamps.shape[0]
        for name in ("event_ids", "features", "labels", "scores", "user", "source", "destination", "success"):
            if getattr(self, name).shape[0] != rows:
                raise StoreError(f"w2.npz column {name!r} has the wrong length")
        if self.features.shape[1:] != (len(self.feature_names),):
            raise StoreError("feature width does not match the manifest")
        if rows and np.any(np.diff(self.timestamps) < 0):
            raise StoreError("w2.npz rows are not in chronological order")

        self.model = XGBClassifier()
        self.model.load_model(str(export_dir / "model.ubj"))

        window = self.manifest["windows"]["w2"]
        self.start = float(window["emit_start"])
        self.end = float(window["emit_end"])

        self.alert_rows = np.flatnonzero(self.scores >= self.threshold)
        self._alert_row_by_id = {self.event_id(int(r)): int(r) for r in self.alert_rows}
        self._alert_tp_cumulative = np.cumsum(self.labels[self.alert_rows], dtype=np.int64)
        self._positive_cumulative = np.cumsum(self.labels, dtype=np.int64)
        self.user_code = {name: code for code, name in enumerate(self.users)}
        self.host_code = {name: code for code, name in enumerate(self.hosts)}

        self.integrity = self._verify(verify_rows)
        self._bins = self._activity_bins()

        self.triage_path = Path(triage_path) if triage_path else export_dir / "triage.json"
        self.triage: dict[str, dict] = self._load_triage()

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------
    def _verify(self, spread: int) -> Integrity:
        rows = self.timestamps.shape[0]
        sample = np.unique(
            np.concatenate(
                [self.alert_rows, np.linspace(0, max(rows - 1, 0), num=min(spread, rows)).astype(np.int64)]
            )
        ) if rows else np.empty(0, dtype=np.int64)
        rescored = predict_scores(self.model, self.features[sample])
        identical = bool(np.array_equal(rescored, self.scores[sample]))
        if not identical:
            raise StoreError(
                "stored scores differ from the loaded model's scores; rebuild the "
                "export with scripts/build_dashboard_data.py"
            )
        return Integrity(
            rows_checked=int(sample.size),
            scores_identical=identical,
            digest_matches=model_digest(self.model) == self.manifest["model"]["digest"],
        )

    # ------------------------------------------------------------------
    # Rows and records
    # ------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return int(self.timestamps.shape[0])

    def event_id(self, row: int) -> str:
        return self.event_ids[row].decode("ascii")

    def cursor(self, until: float) -> int:
        """Number of events with a timestamp at or before ``until``."""
        return int(np.searchsorted(self.timestamps, until, side="right"))

    def event(self, row: int) -> dict:
        score = float(self.scores[row])
        return {
            "row": int(row),
            "event_id": self.event_id(row),
            "timestamp": float(self.timestamps[row]),
            "user": self.users[self.user[row]],
            "source": self.hosts[self.source[row]],
            "destination": self.hosts[self.destination[row]],
            "success": bool(self.success[row]),
            "score": score,
            "alert": score >= self.threshold,
            "severity": severity_for(score, self.threshold),
            "ground_truth": bool(self.labels[row]),
        }

    def alert_record(self, row: int) -> dict:
        record = self.event(row)
        triage = self.triage.get(record["event_id"], {})
        record["status"] = triage.get("status", "new")
        record["note"] = triage.get("note", "")
        record["status_updated"] = triage.get("updated_at")
        return record

    def feature_rows(self, row: int) -> list[dict]:
        return [
            {"name": name, "value": float(value)}
            for name, value in zip(self.feature_names, self.features[row])
        ]

    # ------------------------------------------------------------------
    # Replay-gated views
    # ------------------------------------------------------------------
    def counters(self, until: float) -> dict:
        """Totals over the events the replay has reached.

        The ``evaluation`` block uses research ground truth (the exact redteam
        match). A production SOC would not have it; it is shown so the analyst
        can see how the frozen detector is actually doing.
        """
        cursor = self.cursor(until)
        alerts = int(np.searchsorted(self.alert_rows, cursor, side="left"))
        tp = int(self._alert_tp_cumulative[alerts - 1]) if alerts else 0
        positives = int(self._positive_cumulative[cursor - 1]) if cursor else 0
        return {
            "events": cursor,
            "alerts": alerts,
            "evaluation": {
                "true_positives": tp,
                "false_positives": alerts - tp,
                "missed": positives - tp,
                "positives_seen": positives,
                "precision": tp / alerts if alerts else None,
                "recall": tp / positives if positives else None,
            },
        }

    def _activity_bins(self) -> dict[str, np.ndarray]:
        count = max(1, int(np.ceil((self.end - self.start) / ACTIVITY_BIN_SECONDS)))
        index = np.clip(
            ((self.timestamps - self.start) // ACTIVITY_BIN_SECONDS).astype(np.int64), 0, count - 1
        )
        return {
            "events": np.bincount(index, minlength=count),
            "alerts": np.bincount(index[self.alert_rows], minlength=count),
            "positives": np.bincount(index, weights=self.labels, minlength=count).astype(np.int64),
        }

    def activity(self, until: float) -> list[dict]:
        """Per-minute events, alerts and ground-truth positives up to ``until``."""
        reached = int(np.clip(np.ceil((until - self.start) / ACTIVITY_BIN_SECONDS), 0, self._bins["events"].size))
        return [
            {
                "t": self.start + i * ACTIVITY_BIN_SECONDS,
                "events": int(self._bins["events"][i]),
                "alerts": int(self._bins["alerts"][i]),
                "positives": int(self._bins["positives"][i]),
            }
            for i in range(reached)
        ]

    def recent_events(self, until: float, limit: int = 50) -> list[dict]:
        cursor = self.cursor(until)
        return [self.event(r) for r in range(cursor - 1, max(cursor - limit, 0) - 1, -1)]

    def alert_row(self, event_id: str, until: float) -> int | None:
        """Row of a revealed alert, or ``None`` if unknown or not yet replayed."""
        row = self._alert_row_by_id.get(event_id)
        if row is None or self.timestamps[row] > until:
            return None
        return row

    def row_for_event(self, event_id: str, until: float) -> int | None:
        """Row of any replayed event (alerts are indexed; others are searched)."""
        row = self._alert_row_by_id.get(event_id)
        if row is None:
            encoded = event_id.encode("ascii", "ignore")
            cursor = self.cursor(until)
            matches = np.flatnonzero(self.event_ids[:cursor] == encoded)
            row = int(matches[0]) if matches.size else None
        if row is None or self.timestamps[row] > until:
            return None
        return row

    def alerts(
        self,
        *,
        until: float,
        user: str | None = None,
        source: str | None = None,
        destination: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        ground_truth: bool | None = None,
        t_from: float | None = None,
        t_to: float | None = None,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> dict:
        """Filter the alerts the replay has reached. Text filters are substrings."""
        cursor = self.cursor(until)
        revealed = self.alert_rows[: int(np.searchsorted(self.alert_rows, cursor, side="left"))]
        order = revealed[::-1] if newest_first else revealed

        def contains(value: str, needle: str | None) -> bool:
            return needle is None or needle.lower() in value.lower()

        matched: list[dict] = []
        counts = {name: 0 for name in SEVERITY_ORDER}
        for row in order:
            record = self.alert_record(int(row))
            if t_from is not None and record["timestamp"] < t_from:
                continue
            if t_to is not None and record["timestamp"] > t_to:
                continue
            if not (
                contains(record["user"], user)
                and contains(record["source"], source)
                and contains(record["destination"], destination)
            ):
                continue
            if status and record["status"] != status:
                continue
            if ground_truth is not None and record["ground_truth"] != ground_truth:
                continue
            if query and not any(
                query.lower() in record[key].lower()
                for key in ("event_id", "user", "source", "destination")
            ):
                continue
            counts[record["severity"]] += 1
            if severity and record["severity"] != severity:
                continue
            matched.append(record)

        return {
            "total": len(matched),
            "revealed": int(revealed.size),
            "severity_counts": counts,
            "items": matched[offset : offset + limit],
        }

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def explain(self, row: int) -> dict:
        """Exact TreeSHAP for one event (M4.5), in log-odds space."""
        explanation = explain_tree_model(self.model, self.features[row : row + 1], self.feature_names)
        contributions = [
            {"name": name, "value": float(value), "contribution": float(phi)}
            for name, value, phi in zip(
                self.feature_names, self.features[row], explanation.contributions[0]
            )
        ]
        contributions.sort(key=lambda item: (-abs(item["contribution"]), item["name"]))
        return {
            "method": "exact TreeSHAP (XGBoost pred_contribs), log-odds space",
            "bias": float(explanation.bias[0]),
            "margin": float(explanation.margins[0]),
            "score": float(self.scores[row]),
            "local_accuracy_error": explanation.local_accuracy_error(),
            "contributions": contributions,
        }

    # ------------------------------------------------------------------
    # Triage (analyst workflow state; never a model input)
    # ------------------------------------------------------------------
    def _load_triage(self) -> dict[str, dict]:
        if not self.triage_path.is_file():
            return {}
        try:
            data = json.loads(self.triage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {key: value for key, value in data.items() if key in self._alert_row_by_id}

    def set_triage(self, event_id: str, *, status: str, note: str = "") -> dict:
        if event_id not in self._alert_row_by_id:
            raise KeyError(event_id)
        if status not in TRIAGE_STATUSES:
            raise ValueError(f"status must be one of {TRIAGE_STATUSES}")
        entry = {
            "status": status,
            "note": note[:2000],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.triage[event_id] = entry
        self.triage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.triage_path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.triage, handle, indent=2)
        os.replace(temporary, self.triage_path)
        return entry
