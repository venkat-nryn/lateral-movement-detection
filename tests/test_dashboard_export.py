"""Tests for the dashboard export's identity recovery.

Synthetic in-memory fixtures only; the real export is
``scripts/build_dashboard_data.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.export import ExportError, collect_identities
from ml.preprocessing.schema import CanonicalEvent


def event(event_id, timestamp, user, source, destination, success=True):
    return CanonicalEvent(
        event_id=event_id,
        timestamp=timestamp,
        user=user,
        source_host=source,
        destination_host=destination,
        event_type="authentication",
        success=success,
    )


STREAM = [
    event("ctx-1", 5.0, "U9", "H9", "H9"),
    event("ctx-2", 9.0, "U9", "H9", "H1"),
    event("e-1", 10.0, "U1", "H1", "H2"),
    event("e-2", 11.0, "U2", "H2", "H3", success=False),
    event("e-3", 12.0, "U1", "H1", "H3"),
]


def test_identities_are_coded_in_dataset_order_and_skip_context():
    columns = collect_identities(
        STREAM, emit_start=10.0, expected_event_ids=["e-1", "e-2", "e-3"]
    )

    assert columns.users == ["U1", "U2"]
    assert columns.hosts == ["H1", "H2", "H3"]
    assert columns.user.tolist() == [0, 1, 0]
    assert columns.source.tolist() == [0, 1, 0]
    assert columns.destination.tolist() == [1, 2, 2]
    assert columns.success.tolist() == [True, False, True]
    decoded = [
        (columns.users[u], columns.hosts[s], columns.hosts[d])
        for u, s, d in zip(columns.user, columns.source, columns.destination)
    ]
    assert decoded == [("U1", "H1", "H2"), ("U2", "H2", "H3"), ("U1", "H1", "H3")]


def test_a_misaligned_row_is_rejected():
    with pytest.raises(ExportError, match="row 1"):
        collect_identities(
            STREAM, emit_start=10.0, expected_event_ids=["e-1", "e-3", "e-2"]
        )


def test_a_short_stream_is_rejected():
    with pytest.raises(ExportError, match="ended after 3 of 4"):
        collect_identities(
            STREAM, emit_start=10.0, expected_event_ids=["e-1", "e-2", "e-3", "e-4"]
        )


def test_a_long_stream_is_rejected():
    with pytest.raises(ExportError, match="more emitted events"):
        collect_identities(STREAM, emit_start=10.0, expected_event_ids=["e-1", "e-2"])


def test_empty_emission_region_yields_empty_columns():
    columns = collect_identities(STREAM[:2], emit_start=10.0, expected_event_ids=[])
    assert columns.user.shape == (0,)
    assert columns.users == [] and columns.hosts == []
    assert columns.success.dtype == np.bool_
