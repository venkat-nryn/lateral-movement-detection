"""Static visualisation utilities for the temporal graph engine.

Produces deterministic matplotlib figures from
:class:`~ml.graph.temporal_graph.TemporalGraph` instances:

- :func:`plot_temporal_graph` — directed graph drawing with USER and
  HOST nodes styled distinctly and event context on edges.
- :func:`plot_event_timeline` — chronological timeline of events with
  timestamp, user, source/destination hosts, and outcome.
- :func:`compute_deterministic_layout` — fixed two-column node
  positions (no random layouts).

All figures are labelled as synthetic software fixtures; this module
must not be used to present synthetic activity as real attack data.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ml.graph.temporal_graph import TemporalGraph

DISCLAIMER = "Synthetic software fixture - not LANL data"

USER_COLOR = "#8ecae6"
HOST_COLOR = "#b5e48c"
SUCCESS_MARKER = "#2a9d8f"
FAILURE_MARKER = "#d1495b"

_DEFAULT_GRAPH_TITLE = f"Temporal security interaction graph ({DISCLAIMER})"
_DEFAULT_TIMELINE_TITLE = f"Event timeline ({DISCLAIMER})"


def compute_deterministic_layout(
    graph: TemporalGraph,
) -> dict[str, tuple[float, float]]:
    """Return fixed node positions: users in the left column, hosts in
    the right column, each sorted by identifier. Contains no randomness
    so identical graphs always render identically."""
    users: list[str] = []
    hosts: list[str] = []
    for node_id, attrs in graph.iter_nodes():
        (users if attrs.get("entity_type") == "user" else hosts).append(node_id)
    users.sort()
    hosts.sort()

    def place(column: list[str], x: float) -> None:
        if not column:
            return
        if len(column) == 1:
            positions[column[0]] = (x, 0.5)
            return
        step = 1.0 / (len(column) - 1)
        for index, node in enumerate(column):
            positions[node] = (x, index * step)

    positions: dict[str, tuple[float, float]] = {}
    place(users, 0.0)
    place(hosts, 1.0)
    return positions


def _short_label(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def _new_axes(
    ax: Optional[Axes], figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_temporal_graph(
    graph: TemporalGraph,
    *,
    ax: Optional[Axes] = None,
    title: str = _DEFAULT_GRAPH_TITLE,
    with_edge_labels: bool = True,
) -> Figure:
    """Draw the temporal graph as a static directed figure.

    USER nodes are blue squares, HOST nodes are green circles, arrows
    show ``USER -> DESTINATION_HOST`` interactions, and edge labels
    carry the event id and outcome. Layout is deterministic.
    """
    fig, ax = _new_axes(ax, (11.0, 6.5))

    display = nx.MultiDiGraph()
    for node_id, attrs in graph.iter_nodes():
        display.add_node(node_id, **attrs)
    for source, target, key, attrs in graph.iter_edges():
        display.add_edge(source, target, key=key, **attrs)

    pos = compute_deterministic_layout(graph)

    user_nodes = [n for n, d in display.nodes(data=True) if d.get("entity_type") == "user"]
    host_nodes = [n for n in display.nodes if n not in set(user_nodes)]

    nx.draw_networkx_edges(
        display,
        pos,
        ax=ax,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        edge_color="#404040",
        width=1.6,
        node_size=1400,
    )
    if user_nodes:
        nx.draw_networkx_nodes(
            display,
            pos,
            nodelist=user_nodes,
            node_shape="s",
            node_size=1400,
            node_color=USER_COLOR,
            edgecolors="#1d3557",
            ax=ax,
            label="USER",
        )
    if host_nodes:
        nx.draw_networkx_nodes(
            display,
            pos,
            nodelist=host_nodes,
            node_shape="o",
            node_size=1500,
            node_color=HOST_COLOR,
            edgecolors="#386641",
            ax=ax,
            label="HOST",
        )
    nx.draw_networkx_labels(
        display,
        pos,
        labels={n: _short_label(n) for n in display.nodes},
        font_size=9,
        ax=ax,
    )

    if with_edge_labels and display.number_of_edges() > 0:
        edge_labels = {
            (u, v, key): (
                f"{attrs['event_id']}"
                + ("" if attrs["success"] else " [fail]")
            )
            for u, v, key, attrs in graph.iter_edges()
        }
        nx.draw_networkx_edge_labels(
            display,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            ax=ax,
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if display.number_of_nodes() > 0:
        ax.legend(loc="lower left", scatterpoints=1)
    fig.tight_layout()
    return fig


def plot_event_timeline(
    graph: TemporalGraph,
    *,
    ax: Optional[Axes] = None,
    title: str = _DEFAULT_TIMELINE_TITLE,
) -> Figure:
    """Draw a chronological event timeline.

    Each marker is one event ordered chronologically (earliest at the
    top) and annotated with timestamp, user, source host, destination
    host, and success/failure.
    """
    fig, ax = _new_axes(ax, (13.0, max(3.0, 0.7 * graph.number_of_events() + 2.0)))

    events = graph.get_events_between(float("-inf"), float("inf"))
    if not events:
        ax.text(
            0.5,
            0.5,
            "No events in graph.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        return fig

    stamps = [event.timestamp for event in events]
    ranks = list(range(len(events)))
    colors = [
        SUCCESS_MARKER if event.success else FAILURE_MARKER for event in events
    ]
    ax.scatter(stamps, ranks, c=colors, s=70, zorder=3)
    for rank, event in enumerate(events):
        outcome = "success" if event.success else "FAILED"
        label = (
            f"{event.timestamp:.1f} | {event.user}: "
            f"{event.source_host} -> {event.destination_host} [{outcome}]"
        )
        ax.annotate(label, (stamps[rank], rank), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9)

    ax.set_yticks([])
    ax.invert_yaxis()
    if len(set(stamps)) == 1:
        margin = max(abs(stamps[0]) * 0.05, 10.0)
        ax.set_xlim(stamps[0] - margin, stamps[0] + margin)
    ax.set_xlabel("timestamp (Unix epoch seconds)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title, fontsize=11)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SUCCESS_MARKER,
                   label="success"),
        plt.Line2D([], [], marker="o", linestyle="", color=FAILURE_MARKER,
                   label="failed"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def format_event_lines(graph: TemporalGraph) -> list[str]:
    """Plain-text chronological timeline lines for terminal output."""
    lines = []
    for position, event in enumerate(
        graph.get_events_between(float("-inf"), float("inf")), start=1
    ):
        outcome = "success" if event.success else "FAILED"
        lines.append(
            f"[{position:>2}] t={event.timestamp:.1f}  {event.user}: "
            f"{event.source_host} -> {event.destination_host}  [{outcome}]"
        )
    return lines
