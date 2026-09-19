"""Demonstration of the temporal graph engine on a deterministic
synthetic software scenario.

Run from the project root:

    python scripts/demo_temporal_graph.py

Prints an event timeline to the terminal and saves two figures under
``experiments/figures/``.

The scenario used here is a deterministic software fixture. It is not
derived from LANL, does not represent real attacks, and must not be
presented as evidence of detection performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.graph.synthetic_scenarios import create_multihop_scenario  # noqa: E402
from ml.graph.temporal_graph import TemporalGraph  # noqa: E402
from ml.graph.visualization import (  # noqa: E402
    DISCLAIMER,
    format_event_lines,
    plot_event_timeline,
    plot_temporal_graph,
)

FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
GRAPH_PNG = FIGURE_DIR / "synthetic_multihop_graph.png"
TIMELINE_PNG = FIGURE_DIR / "synthetic_multihop_timeline.png"


def main() -> None:
    scenario = create_multihop_scenario()
    graph = TemporalGraph()
    graph.add_events(scenario)

    print("=" * 64)
    print("Temporal graph demo - multi-step movement pattern")
    print(f"Data: {DISCLAIMER}")
    print(f"Scenario: {len(scenario)} synthetic events "
          "(hypothetical software scenario, not a real attack)")
    print("=" * 64)
    print(f"nodes={graph.number_of_nodes()} "
          f"edges={graph.number_of_edges()} "
          f"events={graph.number_of_events()}")
    print()
    print("Event timeline (chronological):")
    for line in format_event_lines(graph):
        print(" ", line)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    graph_fig = plot_temporal_graph(graph)
    graph_fig.savefig(GRAPH_PNG, dpi=150)
    plt_close(graph_fig)

    timeline_fig = plot_event_timeline(graph)
    timeline_fig.savefig(TIMELINE_PNG, dpi=150)
    plt_close(timeline_fig)

    print()
    print(f"Saved graph figure   : {GRAPH_PNG}")
    print(f"Saved timeline figure: {TIMELINE_PNG}")


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
