"""Graph construction package: schema, temporal graph engine, and
deterministic synthetic scenarios for software testing."""

from ml.graph.synthetic_scenarios import (
    create_benign_scenario,
    create_mixed_scenario,
    create_multihop_scenario,
)
from ml.graph.temporal_graph import (
    DuplicateEventError,
    NodeEventRecord,
    TemporalGraph,
    TemporalGraphError,
    UnknownEventError,
    UnknownNodeError,
    host_node_id,
    user_node_id,
)

__all__ = [
    "DuplicateEventError",
    "NodeEventRecord",
    "TemporalGraph",
    "TemporalGraphError",
    "UnknownEventError",
    "UnknownNodeError",
    "create_benign_scenario",
    "create_mixed_scenario",
    "create_multihop_scenario",
    "host_node_id",
    "user_node_id",
]
