"""Canonical graph schema (Module M2.0).

Defines the supported node/entity types and the canonical temporal
relationship used to build temporal graphs from
:class:`ml.preprocessing.schema.CanonicalEvent` objects.

The design supports multiple events between the same pair of entities
at different timestamps because every relationship carries its own
``event_id`` and ``timestamp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from ml.preprocessing.schema import (
    CANONICAL_EVENT_TYPES,
    SchemaValidationError,
    _validate_identifier,
    _validate_timestamp,
)

if TYPE_CHECKING:
    from ml.preprocessing.schema import CanonicalEvent

SCHEMA_VERSION = "1.0"


class EntityType(str, Enum):
    """Supported node entity types.

    Additional types must not be introduced until a later module
    requires them and the data contract is revised explicitly.
    """

    USER = "user"
    HOST = "host"


NODE_ENTITY_TYPES: frozenset[EntityType] = frozenset(
    {EntityType.USER, EntityType.HOST}
)

REQUIRED_RELATIONSHIP_FIELDS: tuple[str, ...] = (
    "source",
    "destination",
    "timestamp",
    "event_id",
    "event_type",
    "user",
    "success",
)


@dataclass(frozen=True, slots=True)
class TemporalRelationship:
    """A directed, time-stamped relationship between two entities.

    Required fields (none are nullable in schema v1):

    - ``source``: non-empty string identifier of the source node.
    - ``destination``: non-empty string identifier of the destination
      node.
    - ``timestamp``: numeric Unix epoch seconds (UTC), int or float.
    - ``event_id``: non-empty string linking back to the originating
      :class:`~ml.preprocessing.schema.CanonicalEvent`.
    - ``event_type``: one of
      :data:`ml.preprocessing.schema.CANONICAL_EVENT_TYPES`.
    - ``user``: non-empty string account name associated with the
      interaction.
    - ``success``: strictly ``True`` or ``False``.

    Multiple relationships between the same source and destination are
    valid as long as their ``(source, destination, timestamp,
    event_id)`` tuples differ.
    """

    source: str
    destination: str
    timestamp: float
    event_id: str
    event_type: str
    user: str
    success: bool

    def __post_init__(self) -> None:
        errors: list[str] = []
        _validate_identifier(self.source, "source", errors)
        _validate_identifier(self.destination, "destination", errors)
        _validate_timestamp(self.timestamp, "timestamp", errors)
        _validate_identifier(self.event_id, "event_id", errors)
        if self.event_type not in CANONICAL_EVENT_TYPES:
            errors.append(
                f"event_type: {self.event_type!r} is not a canonical "
                f"event type {sorted(CANONICAL_EVENT_TYPES)}"
            )
        _validate_identifier(self.user, "user", errors)
        if not isinstance(self.success, bool):
            errors.append(
                f"success: expected True or False, got "
                f"{type(self.success).__name__}"
            )
        if errors:
            raise SchemaValidationError(
                "invalid temporal relationship: " + "; ".join(errors)
            )

    @classmethod
    def from_event(cls, event: "CanonicalEvent") -> "TemporalRelationship":
        """Derive a relationship from a canonical authentication event.

        The event's ``source_host``/``destination_host`` become the
        relationship endpoints; all other fields are copied verbatim.
        """
        return cls(
            source=event.source_host,
            destination=event.destination_host,
            timestamp=event.timestamp,
            event_id=event.event_id,
            event_type=event.event_type,
            user=event.user,
            success=event.success,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "TemporalRelationship":
        """Build from a flat mapping with strict key checking."""
        if not isinstance(record, Mapping):
            raise SchemaValidationError(
                f"malformed record: expected mapping, got "
                f"{type(record).__name__}"
            )
        missing = [
            k for k in REQUIRED_RELATIONSHIP_FIELDS if k not in record
        ]
        unknown = sorted(set(record) - set(REQUIRED_RELATIONSHIP_FIELDS))
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
            source=record["source"],
            destination=record["destination"],
            timestamp=record["timestamp"],
            event_id=record["event_id"],
            event_type=record["event_type"],
            user=record["user"],
            success=record["success"],
        )

    def endpoint_key(self) -> tuple[str, str]:
        return (self.source, self.destination)

    def identity_key(self) -> tuple[str, str, float, str]:
        """Distinguishes parallel events between the same endpoints."""
        return (self.source, self.destination, self.timestamp, self.event_id)
