"""PyTorch Geometric graph representation of real LANL activity (M4.2).

Scope
-----
This module is **representation only**. It converts a bounded, chronologically
ordered sequence of :class:`~ml.preprocessing.schema.CanonicalEvent` objects
into a deterministic tensor bundle that a graph neural network can consume. It
defines no model, trains nothing and writes nothing to disk.

Graph model
-----------
The node/edge model is the one the M2.1
:class:`~ml.graph.temporal_graph.TemporalGraph` already uses, so nothing about
the existing architecture changes:

* nodes are typed and namespaced -- ``user:<identifier>`` / ``host:<identifier>``
  via :func:`~ml.graph.temporal_graph.user_node_id` and
  :func:`~ml.graph.temporal_graph.host_node_id`;
* every event contributes exactly two directed edges,
  ``user -> destination_host`` (identity / authentication) and
  ``source_host -> destination_host`` (lateral movement).

Event payloads are **not** duplicated onto the edges. The event identifiers and
timestamps are stored once, in ``event_ids`` / ``event_timestamps``, and each
edge carries an integer ``edge_event_index`` pointing back at its originating
event. Only the small numeric quantities a message-passing layer actually
consumes (success, relation type, self-loop, event type, relative time) live in
``edge_attr``.

Temporal rule
-------------
For a target event at timestamp ``T`` a snapshot is built with
``cutoff_timestamp=T``. Every event whose timestamp is **not** strictly less
than ``T`` is dropped before any node or edge is created, so no future edge and
no future statistic can reach the representation.
:meth:`TemporalGraphSnapshot.validate` re-checks that invariant on the
materialized tensors.

Redteam ground truth is never imported, read or referenced here. Labels stay
outside the graph, exactly as in M3.5/M4.0.

Memory
------
Construction is CPU-only and bounded. ``max_events`` defaults to
:data:`DEFAULT_MAX_SNAPSHOT_EVENTS` and raises :class:`SnapshotBoundError`
rather than silently growing, so the full 38M-event dataset cannot be turned
into one graph by accident. Nothing is moved to a GPU by this module;
mini-batch and neighbourhood-sampling concerns belong to the later training
stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ml.graph.temporal_graph import TemporalGraph, host_node_id, user_node_id
from ml.preprocessing.features import DEFAULT_RECENT_WINDOW_SECONDS
from ml.preprocessing.schema import CANONICAL_EVENT_TYPES, CanonicalEvent

#: Node type codes. Mirrors :class:`~ml.graph.schema.EntityType`.
NODE_TYPE_USER = 0
NODE_TYPE_HOST = 1

#: Relation (edge type) codes.
RELATION_IDENTITY = 0
RELATION_MOVEMENT = 1

#: Canonical event types in a fixed order, so the event-type one-hot block in
#: ``edge_attr`` is stable across runs and machines.
EVENT_TYPE_ORDER: tuple[str, ...] = tuple(sorted(CANONICAL_EVENT_TYPES))

#: Node feature layout. Every value is a count or a type flag derived only from
#: events admitted into the snapshot, i.e. from timestamps strictly before the
#: cutoff. No attack-specific quantity appears here.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "is_user",
    "is_host",
    "event_count",
    "out_degree",
    "in_degree",
    "recent_event_count",
    "success_count",
    "failure_count",
)

#: Edge feature layout. ``time_delta`` is seconds since the first admitted
#: event, kept in raw units so no information is destroyed; feature scaling is
#: a modelling decision for the training stage, not for this representation.
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "success",
    "is_identity",
    "is_movement",
    "self_loop",
    *(f"event_type_{name}" for name in EVENT_TYPE_ORDER),
    "time_delta",
)

#: Largest number of events one snapshot may consume without an explicit
#: opt-in. Two edges are produced per event, so this bound also caps the edge
#: tensors.
DEFAULT_MAX_SNAPSHOT_EVENTS = 1_000_000


class GraphDataError(ValueError):
    """Raised when a snapshot cannot be built from the supplied events."""


class SnapshotBoundError(GraphDataError):
    """Raised when a build would exceed the bounded-snapshot guard."""


class TemporalLeakageError(GraphDataError):
    """Raised when a materialized snapshot violates the temporal rule."""


def _feature_index(names: Sequence[str], name: str) -> int:
    return names.index(name)


_NODE_IS_USER = _feature_index(NODE_FEATURE_NAMES, "is_user")
_NODE_IS_HOST = _feature_index(NODE_FEATURE_NAMES, "is_host")
_NODE_EVENT_COUNT = _feature_index(NODE_FEATURE_NAMES, "event_count")
_NODE_OUT_DEGREE = _feature_index(NODE_FEATURE_NAMES, "out_degree")
_NODE_IN_DEGREE = _feature_index(NODE_FEATURE_NAMES, "in_degree")
_NODE_RECENT = _feature_index(NODE_FEATURE_NAMES, "recent_event_count")
_NODE_SUCCESS = _feature_index(NODE_FEATURE_NAMES, "success_count")
_NODE_FAILURE = _feature_index(NODE_FEATURE_NAMES, "failure_count")

_EDGE_SUCCESS = _feature_index(EDGE_FEATURE_NAMES, "success")
_EDGE_IS_IDENTITY = _feature_index(EDGE_FEATURE_NAMES, "is_identity")
_EDGE_IS_MOVEMENT = _feature_index(EDGE_FEATURE_NAMES, "is_movement")
_EDGE_SELF_LOOP = _feature_index(EDGE_FEATURE_NAMES, "self_loop")
_EDGE_TYPE_BASE = _feature_index(
    EDGE_FEATURE_NAMES, f"event_type_{EVENT_TYPE_ORDER[0]}"
)
_EDGE_TIME_DELTA = _feature_index(EDGE_FEATURE_NAMES, "time_delta")


@dataclass(frozen=True, slots=True)
class TemporalGraphSnapshot:
    """A deterministic, PyG-compatible view of one bounded temporal window.

    Array contract (``N`` nodes, ``E`` edges, ``M`` events):

    ====================  =======  ====================================
    attribute             dtype    shape
    ====================  =======  ====================================
    ``x``                 float32  ``(N, len(NODE_FEATURE_NAMES))``
    ``node_type``         int8     ``(N,)``
    ``edge_index``        int64    ``(2, E)``
    ``edge_attr``         float32  ``(E, len(EDGE_FEATURE_NAMES))``
    ``edge_type``         int8     ``(E,)``
    ``edge_time``         float64  ``(E,)``
    ``edge_event_index``  int64    ``(E,)``
    ``event_timestamps``  float64  ``(M,)``
    ====================  =======  ====================================

    ``edge_time`` is non-decreasing: edges are emitted in ``(timestamp,
    event_id)`` order, the same deterministic ordering the M2.1 temporal graph
    uses, so a temporal model can consume them as a stream. Raw LANL timestamps
    are preserved exactly in float64.
    """

    node_ids: tuple[str, ...]
    node_type: np.ndarray
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_type: np.ndarray
    edge_time: np.ndarray
    edge_event_index: np.ndarray
    event_ids: tuple[str, ...]
    event_timestamps: np.ndarray
    cutoff_timestamp: float | None
    window_start: float | None
    window_end: float | None
    recent_window_seconds: float
    excluded_future_events: int

    # ------------------------------------------------------------------
    # Shape / identity
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_events(self) -> int:
        return len(self.event_ids)

    @property
    def node_feature_dimension(self) -> int:
        return len(NODE_FEATURE_NAMES)

    @property
    def edge_feature_dimension(self) -> int:
        return len(EDGE_FEATURE_NAMES)

    def node_index(self, node_id: str) -> int:
        """Row of ``node_id`` (namespaced, e.g. ``"user:U1"``) in ``x``."""
        try:
            return self.node_ids.index(node_id)
        except ValueError:
            raise KeyError(f"no node {node_id!r} in this snapshot") from None

    def user_index(self, user: str) -> int:
        return self.node_index(user_node_id(user))

    def host_index(self, host: str) -> int:
        return self.node_index(host_node_id(host))

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Re-check every structural and temporal invariant on the tensors.

        Raises :class:`GraphDataError` for a shape, dtype, index or finiteness
        violation and :class:`TemporalLeakageError` for a temporal one. Called
        automatically at the end of every build.
        """
        n, e, m = self.num_nodes, self.num_edges, self.num_events

        expected = {
            "x": ((n, len(NODE_FEATURE_NAMES)), np.float32),
            "node_type": ((n,), np.int8),
            "edge_index": ((2, e), np.int64),
            "edge_attr": ((e, len(EDGE_FEATURE_NAMES)), np.float32),
            "edge_type": ((e,), np.int8),
            "edge_time": ((e,), np.float64),
            "edge_event_index": ((e,), np.int64),
            "event_timestamps": ((m,), np.float64),
        }
        for name, (shape, dtype) in expected.items():
            array = getattr(self, name)
            if array.shape != shape:
                raise GraphDataError(
                    f"{name}: expected shape {shape}, got {array.shape}"
                )
            if array.dtype != dtype:
                raise GraphDataError(
                    f"{name}: expected dtype {np.dtype(dtype)}, got {array.dtype}"
                )

        for name in ("x", "edge_attr", "edge_time", "event_timestamps"):
            array = getattr(self, name)
            if array.size and not np.isfinite(array).all():
                raise GraphDataError(f"{name}: contains NaN or Inf")

        if e:
            if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n:
                raise GraphDataError(
                    "edge_index: references a node outside [0, num_nodes)"
                )
            if (
                int(self.edge_event_index.min()) < 0
                or int(self.edge_event_index.max()) >= m
            ):
                raise GraphDataError(
                    "edge_event_index: references an event outside "
                    "[0, num_events)"
                )
            if np.any(np.diff(self.edge_time) < 0):
                raise TemporalLeakageError(
                    "edge_time: edges are not in non-decreasing time order"
                )
            if not np.array_equal(
                self.edge_time, self.event_timestamps[self.edge_event_index]
            ):
                raise GraphDataError(
                    "edge_time: disagrees with the referenced event timestamp"
                )

        if self.cutoff_timestamp is not None:
            if e and float(self.edge_time.max()) >= self.cutoff_timestamp:
                raise TemporalLeakageError(
                    f"future edge: an edge at or after the cutoff "
                    f"{self.cutoff_timestamp!r} is present"
                )
            if m and float(self.event_timestamps.max()) >= self.cutoff_timestamp:
                raise TemporalLeakageError(
                    f"future event: an event at or after the cutoff "
                    f"{self.cutoff_timestamp!r} is present"
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """Small machine-readable description; no per-node data included."""
        user_nodes = int((self.node_type == NODE_TYPE_USER).sum())
        return {
            "num_nodes": self.num_nodes,
            "num_user_nodes": user_nodes,
            "num_host_nodes": self.num_nodes - user_nodes,
            "num_edges": self.num_edges,
            "num_identity_edges": int(
                (self.edge_type == RELATION_IDENTITY).sum()
            ),
            "num_movement_edges": int(
                (self.edge_type == RELATION_MOVEMENT).sum()
            ),
            "num_events": self.num_events,
            "node_feature_dimension": self.node_feature_dimension,
            "edge_feature_dimension": self.edge_feature_dimension,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "cutoff_timestamp": self.cutoff_timestamp,
            "recent_window_seconds": self.recent_window_seconds,
            "excluded_future_events": self.excluded_future_events,
        }

    def nbytes(self) -> int:
        """Total bytes held by the numeric arrays (CPU footprint estimate)."""
        return int(
            sum(
                getattr(self, name).nbytes
                for name in (
                    "x",
                    "node_type",
                    "edge_index",
                    "edge_attr",
                    "edge_type",
                    "edge_time",
                    "edge_event_index",
                    "event_timestamps",
                )
            )
        )

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------
    def to_torch(self, device: str = "cpu") -> dict[str, Any]:
        """Return the arrays as ``torch`` tensors on ``device``.

        Requires PyTorch but not PyTorch Geometric. ``device`` defaults to CPU;
        moving a snapshot to a GPU is the caller's explicit choice, because the
        development machine has only 4 GB of VRAM.
        """
        torch = _require_torch()
        return {
            "x": torch.from_numpy(self.x).to(device),
            "node_type": torch.from_numpy(self.node_type).to(device),
            "edge_index": torch.from_numpy(self.edge_index).to(device),
            "edge_attr": torch.from_numpy(self.edge_attr).to(device),
            "edge_type": torch.from_numpy(self.edge_type).to(device),
            "edge_time": torch.from_numpy(self.edge_time).to(device),
            "edge_event_index": torch.from_numpy(self.edge_event_index).to(
                device
            ),
        }

    def to_pyg_data(self, device: str = "cpu") -> Any:
        """Return a :class:`torch_geometric.data.Data` object on ``device``.

        ``num_nodes`` is set explicitly so an isolated node (a host that only
        ever appears as a source, or an empty snapshot) is not silently lost.
        """
        Data = _require_pyg_data()
        tensors = self.to_torch(device=device)
        data = Data(
            x=tensors["x"],
            edge_index=tensors["edge_index"],
            edge_attr=tensors["edge_attr"],
        )
        data.num_nodes = self.num_nodes
        data.node_type = tensors["node_type"]
        data.edge_type = tensors["edge_type"]
        data.edge_time = tensors["edge_time"]
        data.edge_event_index = tensors["edge_event_index"]
        return data


def torch_available() -> bool:
    """Whether PyTorch can be imported in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def pyg_available() -> bool:
    """Whether PyTorch Geometric can be imported in this environment."""
    try:
        from torch_geometric.data import Data  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is not installed in this environment; the numpy arrays on "
            "TemporalGraphSnapshot are usable without it"
        ) from exc
    return torch


def _require_pyg_data() -> Any:
    _require_torch()
    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch Geometric is not installed in this environment; use "
            "TemporalGraphSnapshot.to_torch() for plain torch tensors"
        ) from exc
    return Data


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def build_graph_snapshot(
    events: Iterable[CanonicalEvent],
    *,
    cutoff_timestamp: float | None = None,
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
    max_events: int | None = DEFAULT_MAX_SNAPSHOT_EVENTS,
) -> TemporalGraphSnapshot:
    """Build a snapshot from a bounded, non-decreasing event sequence.

    Args:
        events: canonical events in non-decreasing timestamp order. Duplicate
            ``event_id`` values are rejected, matching
            :meth:`~ml.graph.temporal_graph.TemporalGraph.add_event`.
        cutoff_timestamp: when given, every event with ``timestamp >= cutoff``
            is excluded before construction. This is the temporal rule for a
            target event at time ``T``: pass ``cutoff_timestamp=T``.
        recent_window_seconds: span of the ``recent_event_count`` node feature,
            measured back from the snapshot reference time. Defaults to the
            same 3600 s window the M3.4 feature extractor uses.
        max_events: bounded-snapshot guard. ``None`` disables it and must be an
            explicit, deliberate choice.

    Raises:
        GraphDataError: on a non-event input, a decreasing timestamp or a
            duplicate ``event_id``.
        SnapshotBoundError: when more than ``max_events`` events are admitted.
    """
    if recent_window_seconds < 0:
        raise GraphDataError("recent_window_seconds must be non-negative")
    if max_events is not None and max_events <= 0:
        raise GraphDataError("max_events must be positive or None")

    admitted, excluded_future = _admit_events(
        events, cutoff_timestamp=cutoff_timestamp, max_events=max_events
    )
    # (timestamp, event_id) is the same deterministic ordering the M2.1 graph
    # uses, so any tie permutation of the input yields identical tensors.
    admitted.sort(key=lambda event: (event.timestamp, event.event_id))

    return _materialize(
        admitted,
        cutoff_timestamp=cutoff_timestamp,
        recent_window_seconds=float(recent_window_seconds),
        excluded_future_events=excluded_future,
    )


def snapshot_from_temporal_graph(
    graph: TemporalGraph,
    *,
    start_time: float = 0.0,
    cutoff_timestamp: float | None = None,
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
    max_events: int | None = DEFAULT_MAX_SNAPSHOT_EVENTS,
) -> TemporalGraphSnapshot:
    """Build a snapshot from an existing M2.1 :class:`TemporalGraph`.

    Events in ``[start_time, cutoff_timestamp)`` are used. The graph's own
    inclusive-bound query is asked for a range ending at the cutoff and the
    cutoff itself is then excluded by :func:`build_graph_snapshot`, so the
    strict "before ``T``" rule holds either way.
    """
    if not isinstance(graph, TemporalGraph):
        raise GraphDataError(
            f"expected a TemporalGraph, got {type(graph).__name__}"
        )
    end_time = float("inf") if cutoff_timestamp is None else cutoff_timestamp
    events = graph.get_events_between(start_time, end_time)
    return build_graph_snapshot(
        events,
        cutoff_timestamp=cutoff_timestamp,
        recent_window_seconds=recent_window_seconds,
        max_events=max_events,
    )


def _admit_events(
    events: Iterable[CanonicalEvent],
    *,
    cutoff_timestamp: float | None,
    max_events: int | None,
) -> tuple[list[CanonicalEvent], int]:
    """Filter out future events, validate order and enforce the guard."""
    admitted: list[CanonicalEvent] = []
    seen: set[str] = set()
    excluded_future = 0
    previous_timestamp: float | None = None

    for event in events:
        if not isinstance(event, CanonicalEvent):
            raise GraphDataError(
                f"expected CanonicalEvent instances, got {type(event).__name__}"
            )
        if (
            previous_timestamp is not None
            and event.timestamp < previous_timestamp
        ):
            raise GraphDataError(
                "events must be supplied in non-decreasing timestamp order"
            )
        previous_timestamp = event.timestamp

        if cutoff_timestamp is not None and event.timestamp >= cutoff_timestamp:
            excluded_future += 1
            continue
        if event.event_id in seen:
            raise GraphDataError(
                f"event_id {event.event_id!r} appears more than once"
            )
        seen.add(event.event_id)
        admitted.append(event)

        if max_events is not None and len(admitted) > max_events:
            raise SnapshotBoundError(
                f"snapshot would exceed the bound of {max_events} events; "
                "narrow the window or pass an explicit max_events"
            )

    return admitted, excluded_future


def _materialize(
    events: list[CanonicalEvent],
    *,
    cutoff_timestamp: float | None,
    recent_window_seconds: float,
    excluded_future_events: int,
) -> TemporalGraphSnapshot:
    node_ids: list[str] = []
    node_index: dict[str, int] = {}
    node_types: list[int] = []

    def intern(node_id: str, node_type: int) -> int:
        index = node_index.get(node_id)
        if index is None:
            index = len(node_ids)
            node_index[node_id] = index
            node_ids.append(node_id)
            node_types.append(node_type)
        return index

    event_count = len(events)
    edge_count = 2 * event_count

    event_ids: list[str] = []
    event_timestamps = np.empty(event_count, dtype=np.float64)
    edge_index = np.empty((2, edge_count), dtype=np.int64)
    edge_attr = np.zeros((edge_count, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    edge_type = np.empty(edge_count, dtype=np.int8)
    edge_time = np.empty(edge_count, dtype=np.float64)
    edge_event_index = np.empty(edge_count, dtype=np.int64)

    window_start = events[0].timestamp if events else None
    window_end = events[-1].timestamp if events else None

    # Reference time for "recent" activity: the cutoff when one is supplied
    # (the target event's own time), otherwise the last admitted timestamp.
    reference_time = (
        cutoff_timestamp
        if cutoff_timestamp is not None
        else (window_end if window_end is not None else 0.0)
    )
    recent_start = reference_time - recent_window_seconds

    # Per-node accumulators, in node-index order.
    participation: list[int] = []
    recent: list[int] = []
    successes: list[int] = []
    failures: list[int] = []

    def grow(index: int) -> None:
        while len(participation) <= index:
            participation.append(0)
            recent.append(0)
            successes.append(0)
            failures.append(0)

    for position, event in enumerate(events):
        timestamp = float(event.timestamp)
        event_ids.append(event.event_id)
        event_timestamps[position] = timestamp

        user_node = intern(user_node_id(event.user), NODE_TYPE_USER)
        source_node = intern(host_node_id(event.source_host), NODE_TYPE_HOST)
        dest_node = intern(host_node_id(event.destination_host), NODE_TYPE_HOST)
        grow(max(user_node, source_node, dest_node))

        # A self-referencing event must count once per distinct node, the same
        # rule the M2.1 per-node event index applies.
        is_recent = timestamp > recent_start
        for node in {user_node, source_node, dest_node}:
            participation[node] += 1
            if is_recent:
                recent[node] += 1
            if event.success:
                successes[node] += 1
            else:
                failures[node] += 1

        success_value = 1.0 if event.success else 0.0
        self_loop_value = (
            1.0 if event.source_host == event.destination_host else 0.0
        )
        type_offset = _EDGE_TYPE_BASE + EVENT_TYPE_ORDER.index(event.event_type)
        time_delta = timestamp - float(window_start)

        identity_row = 2 * position
        movement_row = identity_row + 1

        edge_index[0, identity_row] = user_node
        edge_index[1, identity_row] = dest_node
        edge_type[identity_row] = RELATION_IDENTITY
        edge_attr[identity_row, _EDGE_IS_IDENTITY] = 1.0

        edge_index[0, movement_row] = source_node
        edge_index[1, movement_row] = dest_node
        edge_type[movement_row] = RELATION_MOVEMENT
        edge_attr[movement_row, _EDGE_IS_MOVEMENT] = 1.0

        for row in (identity_row, movement_row):
            edge_attr[row, _EDGE_SUCCESS] = success_value
            edge_attr[row, _EDGE_SELF_LOOP] = self_loop_value
            edge_attr[row, type_offset] = 1.0
            edge_attr[row, _EDGE_TIME_DELTA] = time_delta
            edge_time[row] = timestamp
            edge_event_index[row] = position

    num_nodes = len(node_ids)
    x = np.zeros((num_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node_type = np.asarray(node_types, dtype=np.int8)
    if num_nodes:
        x[:, _NODE_IS_USER] = (node_type == NODE_TYPE_USER).astype(np.float32)
        x[:, _NODE_IS_HOST] = (node_type == NODE_TYPE_HOST).astype(np.float32)
        x[:, _NODE_EVENT_COUNT] = np.asarray(participation, dtype=np.float32)
        x[:, _NODE_RECENT] = np.asarray(recent, dtype=np.float32)
        x[:, _NODE_SUCCESS] = np.asarray(successes, dtype=np.float32)
        x[:, _NODE_FAILURE] = np.asarray(failures, dtype=np.float32)
        # Degrees are read back off the edge list so they can never disagree
        # with the structure a message-passing layer actually sees.
        x[:, _NODE_OUT_DEGREE] = np.bincount(
            edge_index[0], minlength=num_nodes
        ).astype(np.float32)
        x[:, _NODE_IN_DEGREE] = np.bincount(
            edge_index[1], minlength=num_nodes
        ).astype(np.float32)

    snapshot = TemporalGraphSnapshot(
        node_ids=tuple(node_ids),
        node_type=node_type,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_time=edge_time,
        edge_event_index=edge_event_index,
        event_ids=tuple(event_ids),
        event_timestamps=event_timestamps,
        cutoff_timestamp=(
            None if cutoff_timestamp is None else float(cutoff_timestamp)
        ),
        window_start=None if window_start is None else float(window_start),
        window_end=None if window_end is None else float(window_end),
        recent_window_seconds=recent_window_seconds,
        excluded_future_events=excluded_future_events,
    )
    snapshot.validate()
    return snapshot
