"""Real-LANL temporal event classifiers: GAT, behavioural and hybrid (M4.3/M4.4).

What this is
------------
A small GAT-style edge/event classifier built directly on the M4.2 tensor
representation. For a target authentication event at time ``T`` the model
embeds the surrounding graph as it existed strictly *before* ``T``, reads the
embeddings of the three entities the event touches (user, source host,
destination host), and scores the event.

Three variants share this one implementation:

``graph_only`` (M4.3)
    Consumes only M4.2 inputs -- the 8 node features, the 6 edge features, the
    temporal information and the graph relationships. Comparing it against the
    M4.0 XGBoost baseline compares representations, not one model ablated.

``behavioral_only`` (M4.4 control)
    Drops the GAT encoder and classifies from the 17 M3.4 features alone. It
    exists because a hybrid-versus-XGBoost comparison confounds the
    representation with the model family; only a no-graph network with the
    same head isolates what the graph branch contributes.

``hybrid`` (M4.4)
    Concatenates the graph embeddings with the 17 M3.4 features. The GAT
    encoder is identical to ``graph_only``, so a difference between ``hybrid``
    and ``behavioral_only`` is attributable to the graph branch alone.

Two further variants exist only for the M4.7 ablation. Every M4.4 variant also
feeds the head the 7 ``TARGET_FEATURE_NAMES`` attributes, three of which are
graph-derived "seen before" flags, so ``behavioral_only`` is not purely
behavioural. ``target_only`` (those 7 attributes alone) and
``behavioral_no_target`` (the 17 features alone) complete the design needed to
separate the information sources. See :data:`VARIANT_INPUTS`.

The 17 behavioural features are produced by the existing M3.4 streaming
extractor inside the same single pass that builds the batches, so the hybrid
classifier and the M4.0 baseline provably consume the same numbers rather than
two independently reimplemented feature sets.

Research rules enforced here
----------------------------
* **Labels are targets only.** Redteam information enters through ``y`` and
  through nothing else. Node features, edge features, target-event features and
  the 17 behavioural features are built by :mod:`ml.models.graph_data`, by
  :mod:`ml.preprocessing.features` and by this module's ``TARGET_FEATURE_NAMES``
  block, none of which can see a label.
* **Train-only preprocessing.** The behavioural block is standardised by
  :class:`BehavioralScaler`, whose mean and scale are fitted on TRAIN batches
  only and then frozen for validation, test and any later window.
* **Strict temporal cutoff.** Each batch's history graph is built with
  ``cutoff_timestamp`` equal to the timestamp of the batch's first target
  event, so no edge at or after that instant reaches the representation.
  Events sharing a timestamp cannot see each other, matching M3.4/M3.5.
* **Chronological order only.** Batches are produced in time order, consumed in
  time order, and never shuffled. Splits come from the existing
  :class:`~ml.preprocessing.ml_dataset.ChronologicalSplitConfig`; a batch never
  straddles a split boundary.
* **Validation decides, test confirms.** The epoch and the decision threshold
  are both selected on validation, then frozen. Test labels are read once, to
  compute the final metrics.

Memory design
-------------
Preprocessing is CPU-only and streaming: an event stream passes through once
while a bounded deque of at most ``max_history_events`` events supplies the
history. Batches hold forward edges only; the reverse edges a symmetric
message-passing layer needs are materialized on the device at forward time and
freed with the batch. :data:`MEMORY_SAFE_MAX_HISTORY_EVENTS` and
``GNNExperimentConfig.max_batches`` are hard guards so a mis-specified run
cannot grow without bound, and no snapshot of the whole dataset is ever built.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from ml.models.graph_data import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    NODE_TYPE_HOST,
    NODE_TYPE_USER,
    TemporalGraphSnapshot,
    build_graph_snapshot,
)
from ml.graph.temporal_graph import host_node_id, user_node_id
from ml.preprocessing.features import (
    DEFAULT_RECENT_WINDOW_SECONDS,
    EventFeatures,
    TemporalGraphFeatureExtractor,
)
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    LabelKey,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    TimestampRange,
    label_for_event,
)
from ml.preprocessing.schema import CanonicalEvent

#: Attributes of the target event itself, known at prediction time. ``success``,
#: ``self_loop`` and the event type come from the event record; the three
#: ``*_seen_before`` flags and ``time_since_history_end`` are read off the
#: pre-cutoff history graph. Nothing here consults a label.
TARGET_FEATURE_NAMES: tuple[str, ...] = (
    "success",
    "self_loop",
    "event_type_authentication",
    "time_since_history_end",
    "user_seen_before",
    "source_seen_before",
    "destination_seen_before",
)

#: The 17 causal temporal/behavioural features of M3.4, in their canonical
#: order. Taken from :class:`~ml.preprocessing.features.EventFeatures` rather
#: than restated, so the hybrid classifier and the M4.0 XGBoost baseline are
#: guaranteed to consume the same feature set in the same order.
BEHAVIORAL_FEATURE_NAMES: tuple[str, ...] = EventFeatures.feature_names()

#: Hard ceiling on the events one batch's history graph may contain. 20k events
#: produce 40k forward edges, roughly 2 MB on the CPU and well under 100 MB of
#: activations on a 4 GB card.
MEMORY_SAFE_MAX_HISTORY_EVENTS = 50_000

#: Hard ceiling on target events per batch.
MEMORY_SAFE_MAX_BLOCK_SIZE = 32_768

#: Model variants. ``graph_only`` reproduces M4.3 exactly; ``behavioral_only``
#: is the no-graph control that makes the hybrid comparison interpretable;
#: ``hybrid`` is the M4.4 model under test.
MODEL_GRAPH_ONLY = "graph_only"
MODEL_BEHAVIORAL_ONLY = "behavioral_only"
MODEL_HYBRID = "hybrid"
MODEL_VARIANTS: tuple[str, ...] = (
    MODEL_GRAPH_ONLY,
    MODEL_BEHAVIORAL_ONLY,
    MODEL_HYBRID,
)

#: M4.7 ablation variants. They complete the input design without changing any
#: M4.4 variant: ``target_only`` feeds the head the 7 target-event attributes
#: alone, and ``behavioral_no_target`` the 17 behavioural features alone.
MODEL_TARGET_ONLY = "target_only"
MODEL_BEHAVIORAL_NO_TARGET = "behavioral_no_target"
ABLATION_VARIANTS: tuple[str, ...] = (
    MODEL_TARGET_ONLY,
    MODEL_BEHAVIORAL_NO_TARGET,
    MODEL_BEHAVIORAL_ONLY,
    MODEL_GRAPH_ONLY,
    MODEL_HYBRID,
)

#: Input blocks each variant's classifier head receives:
#: ``(graph embeddings, 17 behavioural features, 7 target attributes)``.
VARIANT_INPUTS: dict[str, tuple[bool, bool, bool]] = {
    MODEL_TARGET_ONLY: (False, False, True),
    MODEL_BEHAVIORAL_NO_TARGET: (False, True, False),
    MODEL_BEHAVIORAL_ONLY: (False, True, True),
    MODEL_GRAPH_ONLY: (True, False, True),
    MODEL_HYBRID: (True, True, True),
}


class GNNError(RuntimeError):
    """Raised when a GNN experiment is mis-configured or unsafe to run."""


@dataclass(frozen=True, slots=True)
class GNNExperimentConfig:
    """Deterministic configuration for one bounded experiment."""

    block_size: int = 8_192
    max_history_events: int = 20_000
    hidden_dim: int = 32
    heads: int = 4
    dropout: float = 0.0
    epochs: int = 20
    learning_rate: float = 0.005
    weight_decay: float = 0.0
    seed: int = 0
    max_batches: int = 400
    device: str = "cpu"
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS

    def __post_init__(self) -> None:
        if not 0 < self.block_size <= MEMORY_SAFE_MAX_BLOCK_SIZE:
            raise GNNError(
                f"block_size must be in (0, {MEMORY_SAFE_MAX_BLOCK_SIZE}]"
            )
        if not 0 < self.max_history_events <= MEMORY_SAFE_MAX_HISTORY_EVENTS:
            raise GNNError(
                "max_history_events must be in "
                f"(0, {MEMORY_SAFE_MAX_HISTORY_EVENTS}]"
            )
        if self.max_batches <= 0:
            raise GNNError("max_batches must be positive")
        if self.epochs <= 0:
            raise GNNError("epochs must be positive")


# ----------------------------------------------------------------------
# Batch construction (CPU, numpy, label-free except for the target vector)
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventGraphBatch:
    """One temporal block: a history graph plus the events to classify.

    ``x`` / ``edge_index`` / ``edge_attr`` describe the graph strictly before
    ``cutoff_timestamp``. The three ``target_*_index`` arrays point at rows of
    ``x``; entities never seen before the cutoff are appended as isolated nodes
    carrying their entity-type flags and zero history, which is exactly what is
    known about them at prediction time.

    ``behavioral`` holds the 17 causal M3.4 features for each target event,
    produced by the same streaming extractor the M4.0 baseline uses. It is
    carried alongside the graph rather than in a second pipeline, so the hybrid
    model and XGBoost provably see the same numbers.
    """

    split: str
    cutoff_timestamp: float
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    target_user_index: np.ndarray
    target_source_index: np.ndarray
    target_dest_index: np.ndarray
    target_attr: np.ndarray
    behavioral: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    history_events: int

    @property
    def num_targets(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def positive_count(self) -> int:
        return int(np.count_nonzero(self.labels))

    def nbytes(self) -> int:
        return int(
            self.x.nbytes
            + self.edge_index.nbytes
            + self.edge_attr.nbytes
            + self.target_user_index.nbytes
            + self.target_source_index.nbytes
            + self.target_dest_index.nbytes
            + self.target_attr.nbytes
            + self.behavioral.nbytes
            + self.labels.nbytes
            + self.timestamps.nbytes
        )

    def validate(self) -> None:
        """Structural checks; raises :class:`GNNError` on any violation."""
        b, n = self.num_targets, self.num_nodes
        if self.x.shape != (n, len(NODE_FEATURE_NAMES)):
            raise GNNError(f"x has shape {self.x.shape}")
        if self.edge_attr.shape != (self.num_edges, len(EDGE_FEATURE_NAMES)):
            raise GNNError(f"edge_attr has shape {self.edge_attr.shape}")
        if self.target_attr.shape != (b, len(TARGET_FEATURE_NAMES)):
            raise GNNError(f"target_attr has shape {self.target_attr.shape}")
        if self.behavioral.shape != (b, len(BEHAVIORAL_FEATURE_NAMES)):
            raise GNNError(f"behavioral has shape {self.behavioral.shape}")
        for name in (
            "target_user_index",
            "target_source_index",
            "target_dest_index",
        ):
            index = getattr(self, name)
            if index.shape != (b,):
                raise GNNError(f"{name} has shape {index.shape}")
            if b and (int(index.min()) < 0 or int(index.max()) >= n):
                raise GNNError(f"{name} points outside the node range")
        if self.num_edges and int(self.edge_index.max()) >= n:
            raise GNNError("edge_index points outside the node range")
        if self.split not in SPLIT_NAMES:
            raise GNNError(f"unknown split {self.split!r}")
        for name in ("x", "edge_attr", "target_attr", "behavioral"):
            array = getattr(self, name)
            if array.size and not np.isfinite(array).all():
                raise GNNError(f"{name} contains NaN or Inf")
        if self.timestamps.size and float(self.timestamps.min()) < (
            self.cutoff_timestamp
        ):
            raise GNNError("a target event precedes the history cutoff")

    def to_device(
        self, device: str = "cpu", scaler: "BehavioralScaler | None" = None
    ) -> dict[str, object]:
        """Move the batch to ``device``, adding reverse edges on the way.

        Reverse edges are built here rather than stored so the CPU cache holds
        one copy of the graph, not two. The extra ``is_reverse`` column keeps
        the two directions distinguishable to the attention layer.

        ``scaler``, when supplied, applies the frozen TRAIN-only standardisation
        to the behavioural block. Scaling happens here rather than in the cached
        batch so the raw M3.4 values stay on the batch and cannot be rescaled
        twice.
        """
        import torch

        behavioral = (
            self.behavioral if scaler is None else scaler.transform(self.behavioral)
        )

        edge_index = torch.from_numpy(self.edge_index)
        edge_attr = torch.from_numpy(self.edge_attr)
        forward_flag = torch.zeros((edge_attr.shape[0], 1), dtype=torch.float32)
        reverse_flag = torch.ones_like(forward_flag)
        both_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        both_attr = torch.cat(
            [
                torch.cat([edge_attr, forward_flag], dim=1),
                torch.cat([edge_attr, reverse_flag], dim=1),
            ],
            dim=0,
        )
        return {
            "x": torch.from_numpy(self.x).to(device),
            "edge_index": both_index.to(device),
            "edge_attr": both_attr.to(device),
            "user_index": torch.from_numpy(self.target_user_index).to(device),
            "source_index": torch.from_numpy(self.target_source_index).to(device),
            "dest_index": torch.from_numpy(self.target_dest_index).to(device),
            "target_attr": torch.from_numpy(self.target_attr).to(device),
            "behavioral": torch.from_numpy(behavioral).to(device),
            "y": torch.from_numpy(self.labels.astype(np.float32)).to(device),
        }


def _extend_with_target_nodes(
    snapshot: TemporalGraphSnapshot, required: Sequence[tuple[str, int]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Append isolated nodes for entities absent from the history graph.

    ``required`` is an ordered sequence of ``(node_id, node_type)`` pairs; the
    order is the order of first appearance inside the block, so the mapping is
    deterministic. Appended rows carry only the entity-type flags: their
    history is genuinely empty, and zero is the honest value.
    """
    index = {node_id: position for position, node_id in enumerate(snapshot.node_ids)}
    extra_types: list[int] = []
    for node_id, node_type in required:
        if node_id not in index:
            index[node_id] = len(index)
            extra_types.append(node_type)

    if not extra_types:
        return snapshot.x, index

    extra = np.zeros((len(extra_types), len(NODE_FEATURE_NAMES)), dtype=np.float32)
    types = np.asarray(extra_types, dtype=np.int8)
    extra[:, NODE_FEATURE_NAMES.index("is_user")] = types == NODE_TYPE_USER
    extra[:, NODE_FEATURE_NAMES.index("is_host")] = types == NODE_TYPE_HOST
    return np.vstack([snapshot.x, extra]), index


def _target_features(
    event: CanonicalEvent,
    *,
    history_end: float | None,
    seen: dict[str, int],
    history_node_count: int,
) -> list[float]:
    """Attributes of one target event, all knowable strictly before ``T``."""

    def seen_before(node_id: str) -> float:
        position = seen.get(node_id)
        return 1.0 if position is not None and position < history_node_count else 0.0

    return [
        1.0 if event.success else 0.0,
        1.0 if event.source_host == event.destination_host else 0.0,
        1.0,  # event_type_authentication: the only canonical type in schema v1
        0.0 if history_end is None else max(0.0, event.timestamp - history_end),
        seen_before(user_node_id(event.user)),
        seen_before(host_node_id(event.source_host)),
        seen_before(host_node_id(event.destination_host)),
    ]


def _make_batch(
    history: Sequence[CanonicalEvent],
    targets: Sequence[CanonicalEvent],
    labels: Sequence[int],
    split: str,
    behavioral: Sequence[Sequence[float]],
) -> EventGraphBatch:
    cutoff = float(targets[0].timestamp)
    snapshot = build_graph_snapshot(
        list(history),
        cutoff_timestamp=cutoff,
        max_events=MEMORY_SAFE_MAX_HISTORY_EVENTS,
    )
    history_node_count = snapshot.num_nodes
    history_end = (
        None if snapshot.window_end is None else float(snapshot.window_end)
    )

    required: list[tuple[str, int]] = []
    for event in targets:
        required.append((user_node_id(event.user), NODE_TYPE_USER))
        required.append((host_node_id(event.source_host), NODE_TYPE_HOST))
        required.append((host_node_id(event.destination_host), NODE_TYPE_HOST))
    x, index = _extend_with_target_nodes(snapshot, required)

    user_index = np.empty(len(targets), dtype=np.int64)
    source_index = np.empty(len(targets), dtype=np.int64)
    dest_index = np.empty(len(targets), dtype=np.int64)
    target_attr = np.empty(
        (len(targets), len(TARGET_FEATURE_NAMES)), dtype=np.float32
    )
    timestamps = np.empty(len(targets), dtype=np.float64)

    for position, event in enumerate(targets):
        user_index[position] = index[user_node_id(event.user)]
        source_index[position] = index[host_node_id(event.source_host)]
        dest_index[position] = index[host_node_id(event.destination_host)]
        target_attr[position] = _target_features(
            event,
            history_end=history_end,
            seen=index,
            history_node_count=history_node_count,
        )
        timestamps[position] = event.timestamp

    batch = EventGraphBatch(
        split=split,
        cutoff_timestamp=cutoff,
        x=x,
        edge_index=snapshot.edge_index,
        edge_attr=snapshot.edge_attr,
        target_user_index=user_index,
        target_source_index=source_index,
        target_dest_index=dest_index,
        target_attr=target_attr,
        behavioral=np.asarray(behavioral, dtype=np.float32).reshape(
            len(targets), len(BEHAVIORAL_FEATURE_NAMES)
        ),
        labels=np.asarray(labels, dtype=np.int8),
        timestamps=timestamps,
        history_events=snapshot.num_events,
    )
    batch.validate()
    return batch


def iter_event_batches(
    events: Iterable[CanonicalEvent],
    label_index: frozenset[LabelKey],
    *,
    timestamp_range: TimestampRange,
    emit_from_timestamp: float,
    config: GNNExperimentConfig | None = None,
    split_config: ChronologicalSplitConfig | None = None,
) -> Iterator[EventGraphBatch]:
    """Stream chronological batches from a bounded real-LANL event source.

    Events earlier than ``emit_from_timestamp`` build history only and are
    never classified, exactly as in the M3.6 window. A block is closed when it
    reaches ``block_size`` targets or when the chronological split changes, so
    no batch straddles a split boundary.

    Every event -- context events included -- is also pushed through the M3.4
    :class:`~ml.preprocessing.features.TemporalGraphFeatureExtractor` in stream
    order, so the 17 behavioural features attached to each target event are the
    same ones the M4.0 baseline consumes. The extractor is never reset at the
    emission boundary, matching :meth:`MLDataset.from_event_source`.
    """
    settings = config or GNNExperimentConfig()
    splits = split_config or ChronologicalSplitConfig()

    extractor = TemporalGraphFeatureExtractor(
        recent_window_seconds=settings.recent_window_seconds
    )
    history: deque[CanonicalEvent] = deque(maxlen=settings.max_history_events)
    pending: list[CanonicalEvent] = []
    pending_labels: list[int] = []
    pending_behavioral: list[tuple[float, ...]] = []
    pending_split: str | None = None
    produced = 0

    def flush() -> Iterator[EventGraphBatch]:
        nonlocal pending, pending_labels, pending_behavioral
        nonlocal pending_split, produced
        if not pending:
            return
        if produced >= settings.max_batches:
            raise GNNError(
                f"batch guard hit: more than {settings.max_batches} batches; "
                "widen block_size or narrow the window"
            )
        yield _make_batch(
            history, pending, pending_labels, pending_split, pending_behavioral
        )
        produced += 1
        history.extend(pending)
        pending = []
        pending_labels = []
        pending_behavioral = []
        pending_split = None

    previous_timestamp: float | None = None
    for event in events:
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise GNNError("events must arrive in non-decreasing timestamp order")
        previous_timestamp = event.timestamp

        # Causal by construction: the extractor exposes only events with a
        # strictly earlier timestamp, and same-timestamp events are held in a
        # pending batch so they cannot see each other.
        features = extractor.process_event(event)

        if event.timestamp < emit_from_timestamp:
            # Context region: real history, never a training or test target.
            yield from flush()
            history.append(event)
            continue

        split = splits.split_for_timestamp(
            event.timestamp, timestamp_range.start, timestamp_range.end
        )
        if pending_split is not None and split != pending_split:
            yield from flush()
        if len(pending) >= settings.block_size:
            yield from flush()

        pending_split = split
        pending.append(event)
        pending_labels.append(label_for_event(event, label_index))
        pending_behavioral.append(features.to_vector())

    yield from flush()


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress magnitude while preserving sign.

    The M3.4 time deltas use ``-1.0`` to mean "no prior event", so the plain
    ``log1p(clamp(min=0))`` used on the graph tensors would collapse that
    sentinel onto a genuine zero-second gap. ``sign(v) * log1p(|v|)`` keeps the
    two distinct and is a fixed transform that learns nothing from the data.
    """
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(slots=True)
class BehavioralScaler:
    """Standardisation of the 17 behavioural features, fitted on TRAIN only.

    RESEARCH_CONSTRAINTS section 13 requires learned preprocessing statistics
    to come from training data alone and to be frozen before validation, test
    or W2 are touched. This object is therefore fitted once, from training
    batches, and then applied unchanged everywhere else. It refuses to be
    fitted twice and refuses to transform before it is fitted.
    """

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    fitted_on_events: int = 0

    #: Standard deviations below this are treated as constant columns and are
    #: divided by 1.0, so a feature that never varies in training becomes zero
    #: instead of exploding.
    minimum_scale: float = 1e-6

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.scale is not None

    def fit(self, batches: Sequence[EventGraphBatch]) -> "BehavioralScaler":
        """Fit from training batches only; ``batches`` must all be TRAIN."""
        if self.is_fitted:
            raise GNNError(
                "BehavioralScaler is already fitted; refitting would silently "
                "change the frozen preprocessing statistics"
            )
        offending = sorted({b.split for b in batches} - {SPLIT_TRAIN})
        if offending:
            raise GNNError(
                f"BehavioralScaler must be fitted on {SPLIT_TRAIN} batches "
                f"only, got {offending}"
            )
        if not batches:
            raise GNNError("cannot fit BehavioralScaler on zero batches")

        stacked = signed_log1p(
            np.concatenate([batch.behavioral for batch in batches])
        )
        self.mean = stacked.mean(axis=0).astype(np.float32)
        scale = stacked.std(axis=0).astype(np.float32)
        scale[scale < self.minimum_scale] = 1.0
        self.scale = scale
        self.fitted_on_events = int(stacked.shape[0])
        return self

    def transform(self, behavioral: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise GNNError("BehavioralScaler must be fitted before use")
        return ((signed_log1p(behavioral) - self.mean) / self.scale).astype(
            np.float32
        )


@dataclass(slots=True)
class BatchedEventDataset:
    """All batches of one bounded experiment, kept on the CPU in time order."""

    batches: list[EventGraphBatch] = field(default_factory=list)

    @classmethod
    def from_batches(cls, batches: Iterable[EventGraphBatch]) -> "BatchedEventDataset":
        return cls(batches=list(batches))

    def split(self, name: str) -> list[EventGraphBatch]:
        return [batch for batch in self.batches if batch.split == name]

    def counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for name in SPLIT_NAMES:
            batches = self.split(name)
            positives = sum(batch.positive_count() for batch in batches)
            targets = sum(batch.num_targets for batch in batches)
            result[name] = {
                "batches": len(batches),
                "events": targets,
                "positive": positives,
                "negative": targets - positives,
            }
        return result

    def labels(self, name: str) -> np.ndarray:
        batches = self.split(name)
        if not batches:
            return np.empty(0, dtype=np.int8)
        return np.concatenate([batch.labels for batch in batches])

    def nbytes(self) -> int:
        return sum(batch.nbytes() for batch in self.batches)

    def peak_batch_edges(self) -> int:
        return max((batch.num_edges for batch in self.batches), default=0)

    def summary(self) -> dict[str, object]:
        return {
            "batches": len(self.batches),
            "cpu_bytes": self.nbytes(),
            "peak_batch_edges": self.peak_batch_edges(),
            "peak_batch_nodes": max(
                (batch.num_nodes for batch in self.batches), default=0
            ),
            "counts": self.counts(),
        }


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(
    config: GNNExperimentConfig | None = None,
    *,
    variant: str = MODEL_GRAPH_ONLY,
    node_dim: int = len(NODE_FEATURE_NAMES),
    edge_dim: int = len(EDGE_FEATURE_NAMES) + 1,
    target_dim: int = len(TARGET_FEATURE_NAMES),
    behavioral_dim: int = len(BEHAVIORAL_FEATURE_NAMES),
):
    """Construct one variant of the event classifier.

    ``graph_only`` is the M4.3 model unchanged, ``behavioral_only`` drops the
    GAT encoder, and ``hybrid`` concatenates both. The M4.7 ablation variants
    ``target_only`` and ``behavioral_no_target`` are also accepted; see
    :data:`VARIANT_INPUTS`. Requires torch and torch_geometric.
    """
    if variant not in VARIANT_INPUTS:
        raise GNNError(
            f"unknown variant {variant!r}; expected {ABLATION_VARIANTS}"
        )
    use_graph, use_behavioral, use_target = VARIANT_INPUTS[variant]
    settings = config or GNNExperimentConfig()
    module = _model_class()
    return module(
        node_dim=node_dim,
        edge_dim=edge_dim,
        target_dim=target_dim,
        behavioral_dim=behavioral_dim,
        hidden_dim=settings.hidden_dim,
        heads=settings.heads,
        dropout=settings.dropout,
        use_graph=use_graph,
        use_behavioral=use_behavioral,
        use_target=use_target,
    )


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _model_class():
    """Define the module lazily so importing this file never needs torch."""
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv

    class TemporalEventGAT(nn.Module):
        """Two GAT layers over the pre-cutoff graph, then an event head.

        Counts and time deltas span several orders of magnitude, so every
        input block passes through a fixed ``log1p`` compression followed by a
        learned :class:`~torch.nn.LayerNorm`. The transform is deterministic,
        label-free and identical at train and test time.

        ``use_graph``, ``use_behavioral`` and ``use_target`` select the
        variant. The GAT
        encoder is byte-identical across variants that keep it, so a difference
        between ``hybrid`` and ``behavioral_only`` is attributable to the graph
        branch and nothing else. The behavioural block arrives already
        standardised by the frozen TRAIN-only :class:`BehavioralScaler`, so it
        is fed straight to the head without further compression.
        """

        def __init__(
            self,
            *,
            node_dim: int,
            edge_dim: int,
            target_dim: int,
            behavioral_dim: int,
            hidden_dim: int,
            heads: int,
            dropout: float,
            use_graph: bool = True,
            use_behavioral: bool = False,
            use_target: bool = True,
        ) -> None:
            super().__init__()
            if not (use_graph or use_behavioral or use_target):
                raise GNNError(
                    "at least one of use_graph / use_behavioral / use_target "
                    "is required"
                )
            self.use_graph = use_graph
            self.use_behavioral = use_behavioral
            self.use_target = use_target

            # LayerNorm initialisation draws no random numbers, so creating the
            # target block conditionally leaves every M4.4 variant's weight
            # initialisation -- and therefore its recorded results -- unchanged.
            head_dim = 0
            if use_target:
                self.target_norm = nn.LayerNorm(target_dim)
                head_dim += target_dim
            if use_graph:
                self.node_norm = nn.LayerNorm(node_dim)
                self.edge_norm = nn.LayerNorm(edge_dim)
                self.conv1 = GATConv(
                    node_dim,
                    hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
                self.conv2 = GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=1,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
                head_dim += 3 * hidden_dim
            if use_behavioral:
                self.behavioral_norm = nn.LayerNorm(behavioral_dim)
                head_dim += behavioral_dim

            self.head = nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        @staticmethod
        def compress(values):
            return torch.log1p(values.clamp(min=0.0))

        def encode(self, x, edge_index, edge_attr):
            x = self.node_norm(self.compress(x))
            edge_attr = self.edge_norm(self.compress(edge_attr))
            hidden = torch.nn.functional.elu(
                self.conv1(x, edge_index, edge_attr=edge_attr)
            )
            return self.conv2(hidden, edge_index, edge_attr=edge_attr)

        def forward(
            self,
            x,
            edge_index,
            edge_attr,
            user_index,
            source_index,
            dest_index,
            target_attr,
            behavioral=None,
        ):
            parts = []
            if self.use_graph:
                embeddings = self.encode(x, edge_index, edge_attr)
                parts.extend(
                    [
                        embeddings[user_index],
                        embeddings[source_index],
                        embeddings[dest_index],
                    ]
                )
            if self.use_target:
                parts.append(self.target_norm(self.compress(target_attr)))
            if self.use_behavioral:
                if behavioral is None:
                    raise GNNError(
                        "this variant requires the behavioural feature block"
                    )
                parts.append(self.behavioral_norm(behavioral))
            return self.head(torch.cat(parts, dim=1)).squeeze(-1)

    return TemporalEventGAT


def forward_batch(
    model,
    batch: EventGraphBatch,
    device: str = "cpu",
    scaler: BehavioralScaler | None = None,
):
    """Run one batch through ``model`` and return the raw logits."""
    tensors = batch.to_device(device, scaler=scaler)
    return model(
        tensors["x"],
        tensors["edge_index"],
        tensors["edge_attr"],
        tensors["user_index"],
        tensors["source_index"],
        tensors["dest_index"],
        tensors["target_attr"],
        tensors["behavioral"],
    )


def predict_scores(
    model,
    batches: Sequence[EventGraphBatch],
    device: str = "cpu",
    scaler: BehavioralScaler | None = None,
) -> np.ndarray:
    """Sigmoid scores for every target event, in chronological order."""
    import torch

    if not batches:
        return np.empty(0, dtype=np.float64)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            logits = forward_batch(model, batch, device=device, scaler=scaler)
            scores.append(
                torch.sigmoid(logits).detach().to("cpu").numpy().astype(np.float64)
            )
    return np.concatenate(scores)


def compute_pos_weight(labels: np.ndarray) -> float:
    """``negatives / positives`` from TRAIN labels only, mirroring M4.0.

    Returns 1.0 when training has no positives, which leaves the loss
    unweighted rather than dividing by zero.
    """
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return 1.0
    return float(labels.shape[0] - positives) / float(positives)


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch training loss and validation ranking quality."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    validation_average_precision: list[float | None] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_average_precision: float | None = None
    seconds: float = 0.0


def train_model(
    model,
    dataset: BatchedEventDataset,
    *,
    config: GNNExperimentConfig | None = None,
    device: str = "cpu",
    scaler: BehavioralScaler | None = None,
    verbose: bool = False,
) -> TrainingHistory:
    """Train on TRAIN only; select the epoch on VALIDATION only.

    Batches are consumed in chronological order and never shuffled. The best
    epoch is the one maximising validation average precision; its weights are
    restored before the function returns. Test data is not touched anywhere in
    this function.
    """
    import torch
    from sklearn.metrics import average_precision_score

    settings = config or GNNExperimentConfig()
    torch.manual_seed(settings.seed)

    train_batches = dataset.split(SPLIT_TRAIN)
    validation_batches = dataset.split(SPLIT_VALIDATION)
    if not train_batches:
        raise GNNError("the training split is empty")

    y_train = dataset.labels(SPLIT_TRAIN)
    y_validation = dataset.labels(SPLIT_VALIDATION)
    pos_weight = torch.tensor(
        [compute_pos_weight(y_train)], dtype=torch.float32, device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    history = TrainingHistory()
    best_state: dict[str, object] | None = None
    started = time.time()

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = 0.0
        total_targets = 0
        for batch in train_batches:
            tensors = batch.to_device(device, scaler=scaler)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                tensors["x"],
                tensors["edge_index"],
                tensors["edge_attr"],
                tensors["user_index"],
                tensors["source_index"],
                tensors["dest_index"],
                tensors["target_attr"],
                tensors["behavioral"],
            )
            loss = criterion(logits, tensors["y"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_targets
            total_targets += batch.num_targets

        epoch_loss = total_loss / max(1, total_targets)
        validation_ap: float | None = None
        if validation_batches and 0 < int(np.count_nonzero(y_validation)) < y_validation.size:
            scores = predict_scores(
                model, validation_batches, device=device, scaler=scaler
            )
            validation_ap = float(average_precision_score(y_validation, scores))

        history.epochs.append(epoch)
        history.train_loss.append(epoch_loss)
        history.validation_average_precision.append(validation_ap)
        if validation_ap is not None and (
            history.best_validation_average_precision is None
            or validation_ap > history.best_validation_average_precision
        ):
            history.best_validation_average_precision = validation_ap
            history.best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        if verbose:
            shown = "n/a" if validation_ap is None else f"{validation_ap:.6f}"
            print(
                f"    epoch {epoch:3d}  train_loss={epoch_loss:.5f}  "
                f"validation_pr_auc={shown}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    elif history.epochs:
        # No usable validation signal: keep the final epoch and say so.
        history.best_epoch = history.epochs[-1]

    history.seconds = time.time() - started
    return history


def enable_deterministic_cuda() -> None:
    """Force bitwise-reproducible CUDA kernels (PROJECT_STATE section 29).

    ``GATConv`` accumulates its scatter reduction in a non-deterministic order on
    GPU. Differences of order 1e-9 per epoch compound, and because graph-bearing
    variants have unstable validation PR-AUC across epochs, validation-based epoch
    selection can land on a different epoch and change the reported metrics.
    Must be called before torch initialises cuBLAS.
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> str:
    """Return the device actually usable, falling back to CPU with no error."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def estimate_device_bytes(dataset: BatchedEventDataset, hidden_dim: int, heads: int) -> int:
    """Rough peak bytes one batch needs on the device.

    Counts the graph tensors (doubled for reverse edges) plus the first
    attention layer's per-edge messages, which dominate everything else.
    """
    edges = 2 * dataset.peak_batch_edges()
    nodes = max((batch.num_nodes for batch in dataset.batches), default=0)
    graph_bytes = edges * (2 * 8 + (len(EDGE_FEATURE_NAMES) + 1) * 4)
    message_bytes = edges * hidden_dim * heads * 4 * 3  # message, grad, workspace
    node_bytes = nodes * hidden_dim * heads * 4 * 3
    return int(graph_bytes + message_bytes + node_bytes)
