"""Ground-truth redteam dataset representation (Module M3.2).

Streams LANL redteam.txt.gz records without modification. Provides
deterministic identifiers and filtering for detector evaluation.

Redteam records are NOT treated as ordinary events; they are kept
separate from TemporalGraph to prevent label leakage during evaluation.

Redteam.txt.gz format (4 comma-separated columns, no header):
  0: timestamp (integer, relative or absolute seconds)
  1: user (user identifier with domain, e.g. "U620@DOM1")
  2: source_host (originating host, e.g. "C17693")
  3: destination_host (target host, e.g. "C1003")

Example record:
  150885,U620@DOM1,C17693,C1003

This module knows nothing about specific detection methods.
Records are stored as-is; no success/failure or event types are
invented.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ml.preprocessing.schema import CanonicalEvent


class RedteamError(Exception):
    """Base class for redteam errors."""


class RedteamParseError(RedteamError):
    """Raised when a redteam record cannot be parsed."""


@dataclass(frozen=True, slots=True)
class RedteamRecord:
    """A single redteam activity record.

    Immutable record representing known attack activity. Contains only
    fields actually present in redteam.txt.gz; no synthetic fields are
    added.
    """

    redteam_id: str
    timestamp: float
    user: str
    source_host: str
    destination_host: str


def _make_deterministic_redteam_id(
    timestamp: float, user: str, source: str, dest: str, line_num: int
) -> str:
    """Generate a deterministic, unique redteam_id.

    Uses SHA256 hash of (timestamp, user, source, dest, line_num) to
    create a stable identifier that is reproducible across runs but
    unique per record.
    """
    combined = f"{timestamp}|{user}|{source}|{dest}|{line_num}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    return f"redteam_{digest}"


def iter_redteam_events(path: Path | str) -> Iterator[RedteamRecord]:
    """Stream redteam activity records.

    Reads redteam.txt.gz directly using gzip streaming. Never loads
    the complete dataset into memory. Yields valid RedteamRecord
    objects; raises RedteamParseError for malformed records (does not
    skip silently).

    Duplicate records are preserved; no deduplication occurs.

    Args:
        path: Path to redteam.txt.gz file.

    Yields:
        RedteamRecord instances with fields populated from redteam record.

    Raises:
        FileNotFoundError: If the file does not exist.
        RedteamError: If file cannot be opened.
        RedteamParseError: If a record is malformed or violates schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Redteam file not found: {path}")

    line_num = 0

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line_num_1indexed, line in enumerate(f, start=1):
                line_num = line_num_1indexed
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Parse the record
                parts = line.split(",")
                if len(parts) != 4:
                    raise RedteamParseError(
                        f"Line {line_num}: expected 4 columns, got {len(parts)}"
                    )

                try:
                    timestamp_raw = parts[0].strip()
                    user = parts[1].strip()
                    source_host = parts[2].strip()
                    destination_host = parts[3].strip()

                    # Validate required fields are non-empty
                    if not timestamp_raw:
                        raise RedteamParseError(
                            f"Line {line_num}: timestamp is empty"
                        )
                    if not user:
                        raise RedteamParseError(f"Line {line_num}: user is empty")
                    if not source_host:
                        raise RedteamParseError(
                            f"Line {line_num}: source_host is empty"
                        )
                    if not destination_host:
                        raise RedteamParseError(
                            f"Line {line_num}: destination_host is empty"
                        )

                    # Convert timestamp to float
                    try:
                        timestamp = float(timestamp_raw)
                    except ValueError:
                        raise RedteamParseError(
                            f"Line {line_num}: timestamp {timestamp_raw!r} "
                            f"is not a valid number"
                        )

                    # Generate deterministic redteam_id
                    redteam_id = _make_deterministic_redteam_id(
                        timestamp, user, source_host, destination_host, line_num
                    )

                    # Create and yield the RedteamRecord
                    record = RedteamRecord(
                        redteam_id=redteam_id,
                        timestamp=timestamp,
                        user=user,
                        source_host=source_host,
                        destination_host=destination_host,
                    )

                    yield record

                except RedteamParseError:
                    raise
                except Exception as e:
                    raise RedteamParseError(
                        f"Line {line_num}: unexpected error parsing record: {e}"
                    )

    except RedteamParseError:
        raise
    except Exception as e:
        raise RedteamError(
            f"Error reading redteam file {path}: {e}"
        ) from e


class RedteamGroundTruth:
    """Immutable redteam ground truth for evaluation.

    Loads all redteam records in memory and provides efficient
    filtering and lookup. Suitable for evaluation datasets (redteam
    is only 4.8 KB).
    """

    def __init__(self, records: Iterator[RedteamRecord]) -> None:
        """Load all records from iterator and build indices."""
        self._records: list[RedteamRecord] = list(records)
        self._by_id: dict[str, RedteamRecord] = {}
        self._by_user: dict[str, list[RedteamRecord]] = {}
        self._by_source_host: dict[str, list[RedteamRecord]] = {}
        self._by_dest_host: dict[str, list[RedteamRecord]] = {}

        for record in self._records:
            self._by_id[record.redteam_id] = record
            self._by_user.setdefault(record.user, []).append(record)
            self._by_source_host.setdefault(
                record.source_host, []
            ).append(record)
            self._by_dest_host.setdefault(
                record.destination_host, []
            ).append(record)

    @classmethod
    def from_file(cls, path: Path | str) -> RedteamGroundTruth:
        """Load ground truth from redteam.txt.gz file."""
        return cls(iter_redteam_events(path))

    def number_of_records(self) -> int:
        """Total number of redteam records."""
        return len(self._records)

    def iter_records(self) -> Iterator[RedteamRecord]:
        """Iterate all records in timestamp order."""
        return iter(sorted(self._records, key=lambda r: (r.timestamp, r.redteam_id)))

    def get_record(self, redteam_id: str) -> Optional[RedteamRecord]:
        """Get a record by its redteam_id, or None if not found."""
        return self._by_id.get(redteam_id)

    def filter_by_timestamp(
        self, start_time: float, end_time: float
    ) -> list[RedteamRecord]:
        """Get records with start_time <= timestamp <= end_time."""
        return [
            r for r in self._records
            if start_time <= r.timestamp <= end_time
        ]

    def filter_by_user(self, user: str) -> list[RedteamRecord]:
        """Get all records for a specific user."""
        return self._by_user.get(user, [])

    def filter_by_source_host(self, source_host: str) -> list[RedteamRecord]:
        """Get all records originating from a specific host."""
        return self._by_source_host.get(source_host, [])

    def filter_by_destination_host(self, dest_host: str) -> list[RedteamRecord]:
        """Get all records targeting a specific host."""
        return self._by_dest_host.get(dest_host, [])


def matches_canonical_event(
    redteam_record: RedteamRecord, event: CanonicalEvent
) -> bool:
    """Check if a redteam record exactly matches a CanonicalEvent.

    Exact match compares:
    - timestamp (exact)
    - user (exact)
    - source_host (exact)
    - destination_host (exact)

    This is deterministic matching only. Fuzzy or probabilistic
    matching is not implemented.

    Args:
        redteam_record: The ground truth record.
        event: The candidate event to compare.

    Returns:
        True if all four fields match exactly, False otherwise.
    """
    return (
        redteam_record.timestamp == event.timestamp
        and redteam_record.user == event.user
        and redteam_record.source_host == event.source_host
        and redteam_record.destination_host == event.destination_host
    )
