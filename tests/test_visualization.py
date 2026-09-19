"""Unit tests for M2.3 temporal graph visualisation.

Tests use matplotlib's non-interactive Agg backend and verify
figure/axis/file creation only - no visual pixel comparison.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from ml.graph.synthetic_scenarios import (  # noqa: E402
    create_benign_scenario,
    create_multihop_scenario,
)
from ml.graph.temporal_graph import TemporalGraph  # noqa: E402
from ml.graph.visualization import (  # noqa: E402
    DISCLAIMER,
    compute_deterministic_layout,
    format_event_lines,
    plot_event_timeline,
    plot_temporal_graph,
)


def build_graph():
    graph = TemporalGraph()
    graph.add_events(create_benign_scenario())
    return graph


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# 1. Visualisation accepts a valid TemporalGraph


def test_plot_accepts_valid_temporal_graph():
    fig = plot_temporal_graph(build_graph())
    assert fig is not None
    assert len(fig.axes) == 1


def test_timeline_accepts_valid_temporal_graph():
    fig = plot_event_timeline(build_graph())
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_into_existing_axes_reuses_figure():
    outer_fig, ax = plt.subplots()
    returned = plot_temporal_graph(build_graph(), ax=ax)
    assert returned is outer_fig


# 2. Graph figure is created


def test_graph_figure_has_expected_content_and_disclaimer():
    graph = build_graph()
    fig = plot_temporal_graph(graph)
    ax = fig.axes[0]
    assert DISCLAIMER in ax.get_title()
    # one text label per node plus a title; legend present for USER/HOST
    assert ax.get_legend() is not None
    assert ax.texts, "node labels missing"


# 3. Timeline figure is created


def test_timeline_figure_lists_all_events_chronologically():
    graph = build_graph()
    fig = plot_event_timeline(graph)
    ax = fig.axes[0]
    annotations = [t.get_text() for t in ax.texts]
    assert len(annotations) == graph.number_of_events()
    stamps_in_annotations = [
        float(text.split("|")[0].strip().lstrip("t=")) for text in annotations
    ]
    assert stamps_in_annotations == sorted(stamps_in_annotations)


def test_timeline_marks_failed_events():
    from ml.preprocessing.schema import CanonicalEvent

    tg = TemporalGraph()
    tg.add_event(
        CanonicalEvent(
            event_id="f-1",
            timestamp=10.0,
            user="U",
            source_host="A",
            destination_host="B",
            event_type="authentication",
            success=False,
        )
    )
    fig = plot_event_timeline(tg)
    assert any("[FAILED]" in t.get_text() for t in fig.axes[0].texts)


# 4. Figures can be saved


@pytest.mark.parametrize("plotter", [plot_temporal_graph, plot_event_timeline])
def test_figures_can_be_saved(plotter, tmp_path):
    fig = plotter(build_graph())
    target = tmp_path / "figure.png"
    fig.savefig(target, dpi=100)
    assert target.exists()
    assert target.stat().st_size > 0


# 5. Empty graph handled cleanly


def test_empty_graph_handled_cleanly():
    empty = TemporalGraph()
    graph_fig = plot_temporal_graph(empty)
    timeline_fig = plot_event_timeline(empty)
    assert "No events" in timeline_fig.axes[0].texts[0].get_text()
    assert compute_deterministic_layout(empty) == {}


# 6. Synthetic scenario passes through the pipeline end to end


def test_multihop_scenario_visualisation_pipeline(tmp_path):
    graph = TemporalGraph()
    graph.add_events(create_multihop_scenario())

    layout = compute_deterministic_layout(graph)
    assert layout == compute_deterministic_layout(graph)

    graph_png = tmp_path / "synthetic_multihop_graph.png"
    timeline_png = tmp_path / "synthetic_multihop_timeline.png"

    plot_temporal_graph(graph).savefig(graph_png, dpi=100)
    plot_event_timeline(graph).savefig(timeline_png, dpi=100)

    assert graph_png.exists() and graph_png.stat().st_size > 0
    assert timeline_png.exists() and timeline_png.stat().st_size > 0

    lines = format_event_lines(graph)
    assert len(lines) == graph.number_of_events() == 4


def test_deterministic_layout_is_stable_for_benign_graph():
    first = compute_deterministic_layout(build_graph())
    second = compute_deterministic_layout(build_graph())
    assert first == second
    users_x = {x for node, (x, _) in first.items() if node.startswith("user:")}
    hosts_x = {x for node, (x, _) in first.items() if node.startswith("host:")}
    assert users_x == {0.0}
    assert hosts_x == {1.0}
