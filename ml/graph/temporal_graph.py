"""Dataset-independent temporal graph engine (Module M2.1).

Consumes validated :class:`~ml.preprocessing.schema.CanonicalEvent`
objects and maintains a temporal, directed multi-graph of security
interactions using NetworkX.

Graph model
-----------
- Nodes are typed and namespaced: ``user:<identifier>`` and
  ``host:<identifier>``, so users and hosts can never collide.
- Each canonical event creates two directed edges in a ``MultiDiGraph``:
  1. ``user:<user> -> host:<destination_host>`` (identity relationship)
  2. ``host:<source_host> -> host:<destination_host>`` (lateral movement)
- Edges are keyed by ``event_id`` so repeated events between the same pair
  are never collapsed or overwritten.
- The full event context (``event_id``, ``timestamp``, ``event_type``,
  ``user``, ``source_host``, ``destination_host``, ``success``) is stored
  on both edges; source and destination hosts are represented as first-class
  nodes so host-side history stays queryable.

This module knows nothing about any specific dataset.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable, Iterator

import networkx as nx

from ml.graph.schema import EntityType
from ml.preprocessing.schema import CanonicalEvent

USER_NODE_PREFIX = "user"
HOST_NODE_PREFIX = "host"

ROLE_USER = "user"
ROLE_SOURCE_HOST = "source_host"
ROLE_DESTINATION_HOST = "destination_host"

_TIME_INDEX_KEY = lambda item: item[0]  # noqa: E731


def user_node_id(user: str) -> str:
    return f"{USER_NODE_PREFIX}:{user}"


def host_node_id(host: str) -> str:
    return f"{HOST_NODE_PREFIX}:{host}"


class TemporalGraphError(Exception):
    """Base class for temporal graph errors."""


class UnknownEventError(TemporalGraphError, KeyError):
    """Raised when an event_id is not present in the graph."""


class UnknownNodeError(TemporalGraphError, KeyError):
    """Raised when a node_id is not present in the graph."""


class DuplicateEventError(TemporalGraphError, ValueError):
    """Raised when an event_id already exists in the graph.

    Existing events are never silently overwritten.
    """


@dataclass(frozen=True, slots=True)
class NodeEventRecord:
    """An event occurrence at a node, annotated with the node's roles.

    ``roles`` is a frozenset drawn from ``{"user", "source_host",
    "destination_host"}`` describing how the node participated.
    """

    event: CanonicalEvent
    roles: frozenset[str]


class TemporalGraph:
    """A directed temporal multigraph of canonical security events.

    Events are indexed by id, by global time ``(timestamp, event_id)``,
    and per node by ``(timestamp, event_id)``, so time-window queries
    use binary search instead of full graph scans. All returned event
    collections are ordered chronologically, with ``event_id`` as the
    deterministic tie-breaker for equal timestamps.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._events: dict[str, CanonicalEvent] = {}
        self._time_index: list[tuple[float, str]] = []
        self._node_events: dict[str, list[tuple[float, str]]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_event(self, event: CanonicalEvent) -> None:
        """Add one validated canonical event to the graph.

        Raises ``TypeError`` if ``event`` is not a
        :class:`CanonicalEvent`, and ``DuplicateEventError`` if an
        event with the same ``event_id`` already exists. Invalid input
        is never repaired or coerced.
        """
        if not isinstance(event, CanonicalEvent):
            raise TypeError(
                f"add_event expects a CanonicalEvent instance, got "
                f"{type(event).__name__}"
            )
        event_id = event.event_id
        if event_id in self._events:
            raise DuplicateEventError(
                f"event_id {event_id!r} already exists in the graph"
            )

        user_node = user_node_id(event.user)
        source_node = host_node_id(event.source_host)
        dest_node = host_node_id(event.destination_host)

        self._graph.add_node(
            user_node,
            entity_type=EntityType.USER.value,
            identifier=event.user,
        )
        for node, raw in ((source_node, event.source_host), (dest_node, event.destination_host)):
            self._graph.add_node(
                node, entity_type=EntityType.HOST.value, identifier=raw
            )

        # Identity relationship: user -> destination_host
        self._graph.add_edge(
            user_node,
            dest_node,
            key=event_id,
            event_id=event.event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            user=event.user,
            source_host=event.source_host,
            destination_host=event.destination_host,
            success=event.success,
        )

        # Movement relationship: source_host -> destination_host
        self._graph.add_edge(
            source_node,
            dest_node,
            key=event_id,
            event_id=event.event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            user=event.user,
            source_host=event.source_host,
            destination_host=event.destination_host,
            success=event.success,
        )

        self._events[event_id] = event
        entry = (event.timestamp, event_id)
        bisect.insort(self._time_index, entry)
        # A set is used so self-referencing events (source == destination)
        # are indexed exactly once per node.
        for node in {user_node, source_node, dest_node}:
            bisect.insort(self._node_events.setdefault(node, []), entry)

    def add_events(self, events: Iterable[CanonicalEvent]) -> int:
        """Atomically add a batch of events and return the count added.

        The batch is fully validated (types and duplicate ids, both
        against the graph and within the batch) before anything is
        inserted, so a failure leaves the graph unchanged.
        """
        materialised = list(events)
        seen: set[str] = set()
        for event in materialised:
            if not isinstance(event, CanonicalEvent):
                raise TypeError(
                    f"add_events expects CanonicalEvent instances, got "
                    f"{type(event).__name__}"
                )
            if event.event_id in self._events or event.event_id in seen:
                raise DuplicateEventError(
                    f"event_id {event.event_id!r} already exists in the "
                    f"graph or batch"
                )
            seen.add(event.event_id)
        for event in materialised:
            self.add_event(event)
        return len(materialised)

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def number_of_events(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Point lookups
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> CanonicalEvent:
        try:
            return self._events[event_id]
        except KeyError:
            raise UnknownEventError(
                f"no event with event_id {event_id!r}"
            ) from None

    def has_event(self, event_id: str) -> bool:
        return event_id in self._events

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, str]]]:
        """Yield ``(node_id, attrs_copy)`` for every node in the graph.

        Includes hosts that appear only as ``source_host`` and are
        therefore not endpoints of any edge.
        """
        for node_id, attrs in self._graph.nodes(data=True):
            yield node_id, dict(attrs)

    def get_node(self, node_id: str) -> dict[str, str]:
        """Return a copy of the node's attributes.

        Attributes include ``entity_type`` (``"user"`` or ``"host"``)
        and ``identifier`` (the unprefixed raw identifier).
        """
        try:
            return dict(self._graph.nodes[node_id])
        except KeyError:
            raise UnknownNodeError(
                f"no node {node_id!r} in the graph"
            ) from None

    # ------------------------------------------------------------------
    # Event queries
    # ------------------------------------------------------------------

    def _window_slice(
        self,
        index: list[tuple[float, str]],
        start_time: float,
        end_time: float,
    ) -> list[tuple[float, str]]:
        lo = bisect.bisect_left(index, start_time, key=_TIME_INDEX_KEY)
        hi = bisect.bisect_right(index, end_time, key=_TIME_INDEX_KEY)
        return index[lo:hi]

    def get_node_events(self, node_id: str) -> list[CanonicalEvent]:
        """Events involving ``node_id`` in any role, chronologically."""
        self._require_node(node_id)
        return [
            self._events[event_id]
            for _, event_id in self._node_events.get(node_id, [])
        ]

    def get_event_history(self, node_id: str) -> list[NodeEventRecord]:
        """Role-annotated interaction history of ``node_id``.

        Chronologically ordered with ``event_id`` as deterministic
        tie-breaker.
        """
        self._require_node(node_id)
        records: list[NodeEventRecord] = []
        for timestamp, event_id in self._node_events.get(node_id, []):
            event = self._events[event_id]
            roles = self._roles_for(node_id, event)
            records.append(NodeEventRecord(event=event, roles=roles))
        return records

    @staticmethod
    def _roles_for(node_id: str, event: CanonicalEvent) -> frozenset[str]:
        roles: set[str] = set()
        if node_id == user_node_id(event.user):
            roles.add(ROLE_USER)
        if node_id == host_node_id(event.source_host):
            roles.add(ROLE_SOURCE_HOST)
        if node_id == host_node_id(event.destination_host):
            roles.add(ROLE_DESTINATION_HOST)
        return frozenset(roles)

    def get_events_between(
        self, start_time: float, end_time: float
    ) -> list[CanonicalEvent]:
        """Events with ``start_time <= timestamp <= end_time``.

        Both bounds are inclusive.
        """
        return [
            self._events[event_id]
            for _, event_id in self._window_slice(
                self._time_index, start_time, end_time
            )
        ]

    def _require_node(self, node_id: str) -> None:
        if not self._graph.has_node(node_id):
            raise UnknownNodeError(f"no node {node_id!r} in the graph")

    # ------------------------------------------------------------------
    # Neighbourhood queries
    # ------------------------------------------------------------------

    def get_neighbors(self, node_id: str) -> list[str]:
        """Graph-adjacent nodes (predecessors + successors), sorted.

        Reflects stored edge structure: identity edges (user -> destination)
        and movement edges (source -> destination) exist, so a host that
        appears solely in one role may have limited adjacency.
        """
        self._require_node(node_id)
        adjacent = set(self._graph.predecessors(node_id)) | set(
            self._graph.successors(node_id)
        )
        return sorted(adjacent)

    def get_temporal_neighbors(
        self, node_id: str, start_time: float, end_time: float
    ) -> list[str]:
        """Nodes co-occurring with ``node_id`` in window events.

        Two nodes are temporal neighbours if some event with
        ``start_time <= timestamp <= end_time`` involves both. Unlike
        :meth:`get_neighbors`, this includes entities that share an
        event without being joined by an edge (e.g. a ``source_host``
        co-occurring with the ``user``).
        """
        self._require_node(node_id)
        neighbours: set[str] = set()
        for _, event_id in self._window_slice(
            self._node_events.get(node_id, []), start_time, end_time
        ):
            event = self._events[event_id]
            candidates = (
                user_node_id(event.user),
                host_node_id(event.source_host),
                host_node_id(event.destination_host),
            )
            neighbours.update(c for c in candidates if c != node_id)
        return sorted(neighbours)

    # ------------------------------------------------------------------
    # Subgraphs
    # ------------------------------------------------------------------

    def get_subgraph(self, start_time: float, end_time: float) -> "TemporalGraph":
        """Independent copy containing only events inside the window."""
        subgraph = TemporalGraph()
        subgraph.add_events(self.get_events_between(start_time, end_time))
        return subgraph

    def iter_edges(self) -> Iterator[tuple[str, str, str, dict]]:
        """Yield ``(user_node, dest_node, event_id, attrs)`` per edge.

        Attribute dicts are copies; safe to mutate by callers.
        """
        for u, v, key, attrs in self._graph.edges(keys=True, data=True):
            yield u, v, key, dict(attrs)
