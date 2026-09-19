"""Offline export of the real-LANL dataset the SOC dashboard serves.

Nothing here changes detection logic. It calls the existing research pipeline
exactly as ``scripts/run_cross_window.py`` does:

* **windows**  -- M3.6 ``select_densest_redteam_window`` (W2 excludes W1)
* **model**    -- M4.6 ``fit_frozen_detector``: the standard M4.0 XGBoost fitted
  on W1 TRAIN, with its threshold selected on W1 VALIDATION, then frozen
* **features** -- M3.4/M3.5 via ``build_window_dataset``; labels are the exact
  4-field redteam match (M3.2)
* **scoring**  -- the frozen model over every emitted W2 event (M4.6)

The one addition is identity recovery. ``MLDataset`` keeps event ids but not
the user, source and destination strings an analyst needs, so a second bounded
stream of W2 recovers them. Every row is checked against the dataset's
``event_id`` before it is accepted, so the identities provably belong to the
feature rows they sit next to.

Output (gitignored, ``data/processed/dashboard/``): ``model.ubj``, ``w1.npz``,
``w2.npz`` and ``manifest.json``.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ml.baselines.xgboost_baseline import (
    XGBoostBaselineConfig,
    evaluate_at_threshold,
    split_matrix,
)
from ml.evaluation.cross_window import (
    assert_generalization_windows,
    evaluation_window,
    fit_frozen_detector,
)
from ml.evaluation.redteam import RedteamGroundTruth
from ml.preprocessing.ml_dataset import SPLIT_TEST, SPLIT_VALIDATION
from ml.preprocessing.ml_window import (
    WindowSpec,
    build_window_dataset,
    iter_window_events,
    select_densest_redteam_window,
)
from ml.preprocessing.schema import CanonicalEvent

EXPORT_VERSION = 1

#: LANL event ids are ``lanl_`` plus 16 hex characters (21 bytes).
EVENT_ID_DTYPE = "S32"


class ExportError(RuntimeError):
    """Raised when the export cannot prove its rows are consistent."""


@dataclass(frozen=True, slots=True)
class IdentityColumns:
    """Integer-coded identities for every emitted event, in dataset order."""

    user: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    success: np.ndarray
    users: list[str]
    hosts: list[str]


def collect_identities(
    events: Iterable[CanonicalEvent],
    *,
    emit_start: float,
    expected_event_ids: Sequence[str],
) -> IdentityColumns:
    """Recover identities for the emitted events of a window.

    ``events`` is the window's full stream, context included. Emitted events
    (``timestamp >= emit_start``, the same rule ``MLDataset`` applies) must
    appear in exactly the order of ``expected_event_ids``; any disagreement in
    id, count or order raises :class:`ExportError`, because silently misaligned
    identities would put the wrong user and hosts next to a feature row.
    """
    count = len(expected_event_ids)
    user = np.empty(count, dtype=np.int32)
    source = np.empty(count, dtype=np.int32)
    destination = np.empty(count, dtype=np.int32)
    success = np.empty(count, dtype=bool)
    user_codes: dict[str, int] = {}
    host_codes: dict[str, int] = {}

    def code(table: dict[str, int], value: str) -> int:
        existing = table.get(value)
        if existing is None:
            existing = len(table)
            table[value] = existing
        return existing

    row = 0
    for event in events:
        if event.timestamp < emit_start:
            continue
        if row >= count:
            raise ExportError(
                "the identity stream holds more emitted events than the dataset"
            )
        if event.event_id != expected_event_ids[row]:
            raise ExportError(
                f"row {row}: stream event {event.event_id!r} does not match "
                f"dataset event {expected_event_ids[row]!r}"
            )
        user[row] = code(user_codes, event.user)
        source[row] = code(host_codes, event.source_host)
        destination[row] = code(host_codes, event.destination_host)
        success[row] = event.success
        row += 1

    if row != count:
        raise ExportError(
            f"the identity stream ended after {row} of {count} emitted events"
        )
    return IdentityColumns(
        user=user,
        source=source,
        destination=destination,
        success=success,
        users=list(user_codes),
        hosts=list(host_codes),
    )


def _spec_dict(spec: WindowSpec) -> dict[str, float | int]:
    return {
        "context_start": spec.context_start,
        "emit_start": spec.emit_start,
        "emit_end": spec.emit_end,
        "redteam_records_in_window": spec.redteam_records_in_window,
    }


def _file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def build_export(
    auth_path: Path,
    redteam_path: Path,
    out_dir: Path,
    *,
    specs: tuple[WindowSpec, WindowSpec] | None = None,
    config: XGBoostBaselineConfig | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    """Build the dashboard dataset and return its manifest.

    ``specs`` and ``config`` exist for tests; the real export uses the M3.6
    windows and the unchanged M4.0 configuration.
    """
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    if specs is None:
        w1 = select_densest_redteam_window(ground_truth)
        w2 = select_densest_redteam_window(ground_truth, exclude=[w1])
    else:
        w1, w2 = specs
    assert_generalization_windows(w1, w2)

    log("[1/4] W1: fit the frozen detector on TRAIN, threshold on VALIDATION")
    window1 = build_window_dataset(auth_path, redteam_path, spec=w1)
    dataset1 = window1.dataset
    detector = fit_frozen_detector(dataset1, config=config)
    validation = dataset1.get_split(SPLIT_VALIDATION)
    test = dataset1.get_split(SPLIT_TEST)
    X_val, y_val = split_matrix(validation)
    X_test, y_test = split_matrix(test)
    w1_arrays = {
        "validation_scores": detector.scores(X_val),
        "validation_labels": y_val.astype(np.int8),
        "validation_timestamps": np.asarray(validation.timestamps, dtype=np.float64),
        "test_scores": detector.scores(X_test),
        "test_labels": y_test.astype(np.int8),
        "test_timestamps": np.asarray(test.timestamps, dtype=np.float64),
    }
    w1_counts = dataset1.split_label_counts()
    w1_boundaries = dataset1.split_boundaries()
    w1_context = dataset1.context_events_processed
    w1_emitted = dataset1.events_processed
    del window1, dataset1, validation, test, X_val, X_test
    gc.collect()
    log(f"      threshold={detector.threshold:.6f} digest={detector.digest[:16]}...")

    log("[2/4] W2: features and frozen scores for every emitted event")
    window2 = build_window_dataset(auth_path, redteam_path, spec=w2)
    w2_context = window2.dataset.context_events_processed
    evaluation = evaluation_window(window2.dataset)
    del window2
    gc.collect()
    w2_scores = detector.scores(evaluation.X)
    log(f"      {evaluation.rows} events, {evaluation.positives} positives")

    log("[3/4] W2: recover identities, checking every row against its event_id")
    identities = collect_identities(
        iter_window_events(auth_path, w2),
        emit_start=w2.emit_start,
        expected_event_ids=evaluation.event_ids,
    )
    log(f"      {len(identities.users)} users, {len(identities.hosts)} hosts")

    log("[4/4] write the export")
    detector.model.save_model(str(out_dir / "model.ubj"))
    np.savez_compressed(out_dir / "w1.npz", **w1_arrays)
    np.savez_compressed(
        out_dir / "w2.npz",
        timestamps=evaluation.timestamps,
        event_ids=np.asarray(evaluation.event_ids, dtype=EVENT_ID_DTYPE),
        features=evaluation.X,
        labels=evaluation.y,
        scores=w2_scores,
        user=identities.user,
        source=identities.source,
        destination=identities.destination,
        success=identities.success,
        users=np.asarray(identities.users),
        hosts=np.asarray(identities.hosts),
    )

    threshold = detector.threshold
    import xgboost

    manifest: dict[str, object] = {
        "export_version": EXPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 1),
        "source": {
            "auth": _file_info(auth_path),
            "redteam": _file_info(redteam_path),
            "redteam_records": ground_truth.number_of_records(),
        },
        "windows": {
            "w1": {
                **_spec_dict(w1),
                "context_events": w1_context,
                "emitted_events": w1_emitted,
                "splits": w1_counts,
                "boundaries": w1_boundaries,
            },
            "w2": {
                **_spec_dict(w2),
                "context_events": w2_context,
                "emitted_events": evaluation.rows,
                "positives": evaluation.positives,
                "users": len(identities.users),
                "hosts": len(identities.hosts),
            },
        },
        "model": {
            "name": "XGBoost (standard, M4.0)",
            "selection": "highest validation PR-AUC among compared models (M4.5)",
            "digest": detector.digest,
            "threshold": threshold,
            "threshold_rule": "validation F1-maximising threshold, frozen (M4.0)",
            "feature_names": list(detector.feature_names),
            "train_rows": detector.train_rows,
            "train_positives": detector.train_positives,
            "validation_rows": detector.validation_rows,
            "validation_positives": detector.validation_positives,
        },
        "metrics": {
            "w1_validation": evaluate_at_threshold(
                w1_arrays["validation_labels"], w1_arrays["validation_scores"], threshold
            ).as_dict(),
            "w1_test": evaluate_at_threshold(
                w1_arrays["test_labels"], w1_arrays["test_scores"], threshold
            ).as_dict(),
            "w2": evaluate_at_threshold(evaluation.y, w2_scores, threshold).as_dict(),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log(f"      done in {manifest['runtime_seconds']}s -> {out_dir}")
    return manifest
