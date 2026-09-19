"""Canonical security event schema (Module M2.0).

Defines the dataset-independent internal representation of a security
event. Dataset adapters convert raw dataset records into
:class:`CanonicalEvent` instances; downstream stages (temporal graph,
detection, risk, attack-path reconstruction) consume only this contract.

This module must not assume any undocumented dataset-specific field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"

EVENT_TYPE_AUTHENTICATION = "authentication"

CANONICAL_EVENT_TYPES: frozenset[str] = frozenset({EVENT_TYPE_AUTHENTICATION})

REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "user",
    "source_host",
    "destination_host",
    "event_type",
    "success",
)


class SchemaValidationError(ValueError):
    """Raised when a record violates the canonical schema.

    The error message lists every violated rule explicitly; malformed
    records are never silently repaired or dropped.
    """


def _validate_identifier(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str):
        errors.append(
            f"{name}: expected non-empty str, got {type(value).__name__}"
        )
    elif value.strip() == "":
        errors.append(f"{name}: must not be empty or whitespace-only")


def _validate_timestamp(value: Any, name: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(
            f"{name}: expected numeric Unix epoch seconds "
            f"(int/float), got {type(value).__name__}"
        )
        return
    if not math.isfinite(float(value)):
        errors.append(f"{name}: must be finite (got {value!r})")
    elif value < 0:
        errors.append(f"{name}: must be non-negative (got {value!r})")


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """A single canonical security event.

    Required fields (none are nullable in schema v1):

    - ``event_id``: unique non-empty string identifying the event.
    - ``timestamp``: numeric Unix epoch seconds (UTC), int or float.
    - ``user``: non-empty string account name associated with the event.
    - ``source_host``: non-empty string identifier of the source host.
    - ``destination_host``: non-empty string identifier of the
      destination host.
    - ``event_type``: one of :data:`CANONICAL_EVENT_TYPES`.
    - ``success``: strictly ``True`` or ``False``.

    Optional fields: none in schema v1. Future extensions must remain
    backward compatible and must not make new fields mandatory.
    """

    event_id: str
    timestamp: float
    user: str
    source_host: str
    destination_host: str
    event_type: str
    success: bool

    def __post_init__(self) -> None:
        errors: list[str] = []
        _validate_identifier(self.event_id, "event_id", errors)
        _validate_timestamp(self.timestamp, "timestamp", errors)
        _validate_identifier(self.user, "user", errors)
        _validate_identifier(self.source_host, "source_host", errors)
        _validate_identifier(
            self.destination_host, "destination_host", errors
        )
        if self.event_type not in CANONICAL_EVENT_TYPES:
            errors.append(
                f"event_type: {self.event_type!r} is not a canonical "
                f"event type {sorted(CANONICAL_EVENT_TYPES)}"
            )
        if not isinstance(self.success, bool):
            errors.append(
                f"success: expected True or False, got "
                f"{type(self.success).__name__}"
            )
        if errors:
            raise SchemaValidationError(
                "invalid canonical event: " + "; ".join(errors)
            )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CanonicalEvent":
        """Build a :class:`CanonicalEvent` from a flat mapping.

        Missing required keys and unknown keys are both rejected so that
        adapters must map dataset-specific fields explicitly.
        """
        if not isinstance(record, Mapping):
            raise SchemaValidationError(
                f"malformed record: expected mapping, got "
                f"{type(record).__name__}"
            )
        missing = [k for k in REQUIRED_EVENT_FIELDS if k not in record]
        unknown = sorted(set(record) - set(REQUIRED_EVENT_FIELDS))
        errors: list[str] = []
        if missing:
            errors.append(f"missing required fields: {missing}")
        if unknown:
            errors.append(f"unknown fields not allowed: {unknown}")
        if errors:
            raise SchemaValidationError(
                "malformed record: " + "; ".join(errors)
            )
        return cls(
            event_id=record["event_id"],
            timestamp=record["timestamp"],
            user=record["user"],
            source_host=record["source_host"],
            destination_host=record["destination_host"],
            event_type=record["event_type"],
            success=record["success"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user": self.user,
            "source_host": self.source_host,
            "destination_host": self.destination_host,
            "event_type": self.event_type,
            "success": self.success,
        }
