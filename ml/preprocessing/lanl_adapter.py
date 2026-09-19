"""LANL authentication dataset adapter (Module M3.1).

Converts raw LANL auth.txt.gz records into CanonicalEvent objects
using streaming gzip reads, preserving exact timestamps and identifiers
without modification, extraction, or in-memory loading of the full dataset.

LANL auth.txt.gz format (9 comma-separated columns, no header):
  0: timestamp (integer, relative or absolute seconds)
  1: user@domain (user identifier with domain)
  2: user_object (logon user, often same as column 1)
  3: source_host (originating host identifier)
  4: destination_host (target host identifier)
  5: authentication_protocol (Kerberos, NTLM, Negotiate, ?)
  6: logon_type (Network, Service, Batch, Interactive, ?)
  7: event_orientation (LogOn, LogOff, TGS, TGT, AuthMap)
  8: status (Success, Fail)

Example record:
  1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterator, Optional

from ml.graph.temporal_graph import TemporalGraph
from ml.preprocessing.schema import (
    CanonicalEvent,
    EVENT_TYPE_AUTHENTICATION,
    SchemaValidationError,
)


class LANLAdapterError(Exception):
    """Base class for LANL adapter errors."""


class LANLParseError(LANLAdapterError):
    """Raised when a LANL record cannot be parsed."""


def _make_deterministic_event_id(
    timestamp: float, user: str, source: str, dest: str, line_num: int
) -> str:
    """Generate a deterministic, unique event_id.

    Uses SHA256 hash of (timestamp, user, source, dest, line_num) to create
    a stable identifier that is reproducible across runs but unique per record.
    """
    combined = f"{timestamp}|{user}|{source}|{dest}|{line_num}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    return f"lanl_{digest}"


def iter_lanl_events(
    path: Path | str,
    limit: Optional[int] = None,
    *,
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
) -> Iterator[CanonicalEvent]:
    """Stream LANL authentication events as CanonicalEvent objects.

    Reads auth.txt.gz directly using gzip streaming. Never loads the complete
    dataset into memory. Yields valid CanonicalEvent objects; raises
    LANLParseError for malformed records (does not skip silently).

    ``start_timestamp`` / ``end_timestamp`` select a bounded temporal region.
    LANL records are stored in non-decreasing timestamp order, so iteration
    **stops** as soon as a record past ``end_timestamp`` is reached; the
    remainder of the file is never read. Records before ``start_timestamp`` are
    skipped cheaply: only the leading timestamp field is parsed, and the full
    9-column validation is not applied to records outside the requested region.

    Line numbers are counted across every physical line, including skipped ones,
    so ``event_id`` values are identical to those produced by a full scan.

    Args:
        path: Path to auth.txt.gz file.
        limit: Maximum number of events to yield. If None, reads entire file.
        start_timestamp: Skip records with a timestamp strictly below this.
        end_timestamp: Stop once a record timestamp exceeds this.

    Yields:
        CanonicalEvent instances with fields populated from LANL record.

    Raises:
        LANLAdapterError: If file not found or cannot be opened.
        LANLParseError: If a record is malformed or violates schema.
        FileNotFoundError: If the file does not exist.
    """
    if (
        start_timestamp is not None
        and end_timestamp is not None
        and end_timestamp < start_timestamp
    ):
        raise ValueError(
            f"end_timestamp ({end_timestamp}) must not be earlier than "
            f"start_timestamp ({start_timestamp})"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LANL file not found: {path}")

    events_yielded = 0
    line_num = 0

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line_num_1indexed, line in enumerate(f, start=1):
                line_num = line_num_1indexed
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Bounded temporal region: parse only the leading timestamp so
                # that seeking to the window costs a fraction of a full parse.
                if start_timestamp is not None or end_timestamp is not None:
                    head = line.partition(",")[0].strip()
                    try:
                        record_timestamp = float(head)
                    except ValueError:
                        raise LANLParseError(
                            f"Line {line_num}: timestamp {head!r} is not a "
                            f"valid number"
                        )
                    # Records are chronologically ordered: past the end of the
                    # region there is nothing left to read.
                    if (
                        end_timestamp is not None
                        and record_timestamp > end_timestamp
                    ):
                        break
                    if (
                        start_timestamp is not None
                        and record_timestamp < start_timestamp
                    ):
                        continue

                # Check if we've hit the limit
                if limit is not None and events_yielded >= limit:
                    break

                # Parse the record
                parts = line.split(",")
                if len(parts) != 9:
                    raise LANLParseError(
                        f"Line {line_num}: expected 9 columns, got {len(parts)}"
                    )

                try:
                    timestamp_raw = parts[0].strip()
                    user = parts[1].strip()
                    # parts[2] is user_object (ignored for now; could be preserved)
                    source_host = parts[3].strip()
                    destination_host = parts[4].strip()
                    # parts[5] is auth_protocol (ignored for canonical event)
                    # parts[6] is logon_type (ignored for canonical event)
                    event_orientation = parts[7].strip()
                    status = parts[8].strip()

                    # Validate required fields are non-empty
                    if not timestamp_raw:
                        raise LANLParseError(
                            f"Line {line_num}: timestamp is empty"
                        )
                    if not user:
                        raise LANLParseError(f"Line {line_num}: user is empty")
                    if not source_host:
                        raise LANLParseError(
                            f"Line {line_num}: source_host is empty"
                        )
                    if not destination_host:
                        raise LANLParseError(
                            f"Line {line_num}: destination_host is empty"
                        )
                    if not status:
                        raise LANLParseError(
                            f"Line {line_num}: status is empty"
                        )

                    # Convert timestamp to float
                    try:
                        timestamp = float(timestamp_raw)
                    except ValueError:
                        raise LANLParseError(
                            f"Line {line_num}: timestamp {timestamp_raw!r} "
                            f"is not a valid number"
                        )

                    # Parse success status (strict boolean)
                    success_lower = status.lower()
                    if success_lower == "success":
                        success = True
                    elif success_lower == "fail":
                        success = False
                    else:
                        raise LANLParseError(
                            f"Line {line_num}: status {status!r} is not "
                            f"'Success' or 'Fail'"
                        )

                    # Normalize event_orientation to canonical event_type
                    # (for now, all auth events map to EVENT_TYPE_AUTHENTICATION)
                    event_type = EVENT_TYPE_AUTHENTICATION

                    # Generate deterministic event_id
                    event_id = _make_deterministic_event_id(
                        timestamp, user, source_host, destination_host, line_num
                    )

                    # Create and validate the CanonicalEvent
                    event = CanonicalEvent(
                        event_id=event_id,
                        timestamp=timestamp,
                        user=user,
                        source_host=source_host,
                        destination_host=destination_host,
                        event_type=event_type,
                        success=success,
                    )

                    events_yielded += 1
                    yield event

                except LANLParseError:
                    raise
                except SchemaValidationError as e:
                    raise LANLParseError(
                        f"Line {line_num}: schema validation failed: {e}"
                    )
                except Exception as e:
                    raise LANLParseError(
                        f"Line {line_num}: unexpected error parsing record: {e}"
                    )

    except LANLParseError:
        raise
    except Exception as e:
        raise LANLAdapterError(
            f"Error reading LANL file {path}: {e}"
        ) from e


def build_lanl_graph(
    path: Path | str, limit: Optional[int] = None
) -> TemporalGraph:
    """Build a TemporalGraph from LANL authentication records.

    Streams events from iter_lanl_events() and adds them to a new
    TemporalGraph. Returns the graph when complete (or limit is reached).

    Args:
        path: Path to auth.txt.gz file.
        limit: Maximum number of events to process.

    Returns:
        TemporalGraph with events added in chronological order.

    Raises:
        LANLParseError: If any record is malformed.
        LANLAdapterError: If file cannot be opened.
    """
    graph = TemporalGraph()

    for event in iter_lanl_events(path, limit=limit):
        graph.add_event(event)

    return graph
