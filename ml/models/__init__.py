"""Model-facing data structures and models for the graph learning stage.

M4.2 provides the tensor representation (:mod:`ml.models.graph_data`); M4.3 and
M4.4 provide the temporal GAT, the behavioural control and the hybrid
classifier (:mod:`ml.models.gnn`).
"""

from ml.models.graph_data import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    NODE_TYPE_HOST,
    NODE_TYPE_USER,
    RELATION_IDENTITY,
    RELATION_MOVEMENT,
    GraphDataError,
    SnapshotBoundError,
    TemporalGraphSnapshot,
    TemporalLeakageError,
    build_graph_snapshot,
    pyg_available,
    snapshot_from_temporal_graph,
    torch_available,
)

__all__ = [
    "EDGE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "NODE_TYPE_HOST",
    "NODE_TYPE_USER",
    "RELATION_IDENTITY",
    "RELATION_MOVEMENT",
    "GraphDataError",
    "SnapshotBoundError",
    "TemporalGraphSnapshot",
    "TemporalLeakageError",
    "build_graph_snapshot",
    "pyg_available",
    "snapshot_from_temporal_graph",
    "torch_available",
]
