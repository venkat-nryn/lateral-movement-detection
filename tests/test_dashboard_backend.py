"""Tests for the SOC dashboard backend.

Every fixture is a small synthetic export built in a temporary directory: a
tiny XGBoost model scored over random 17-feature rows. These tests exercise the
serving layer's mechanics -- integrity checks, replay gating, filters, graph
aggregation, the lab's isolation and the API -- and derive no research claim.
The real export is produced by ``scripts/build_dashboard_data.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from xgboost import XGBClassifier

from backend.api.server import create_app
from backend.services import context, graph
from backend.services.experiments import computed_results, step_curves, threshold_sweep
from backend.services.lab import LabError, LabSession
from backend.services.replay import ReplayClock
from backend.services.store import DashboardStore, StoreError, severity_for
from ml.baselines.xgboost_baseline import evaluate_at_threshold, predict_scores
from ml.evaluation.cross_window import model_digest
from ml.preprocessing.features import EventFeatures, TemporalGraphFeatureExtractor
from ml.preprocessing.schema import CanonicalEvent

FEATURES = EventFeatures.feature_names()
THRESHOLD = 0.3
START, END = 1000.0, 3000.0


def build_export(directory, rows: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((rows, len(FEATURES))) * 50.0
    y = (X[:, 0] > 42.0).astype(np.int8)
    model = XGBClassifier(
        n_estimators=12, max_depth=3, learning_rate=0.3, n_jobs=1, random_state=0, tree_method="hist"
    )
    model.fit(X, y)
    scores = predict_scores(model, X)
    timestamps = START + np.arange(rows) * 5.0
    users, hosts = [f"U{i}" for i in range(5)], [f"H{i}" for i in range(10)]
    np.savez_compressed(
        directory / "w2.npz",
        timestamps=timestamps,
        event_ids=np.asarray([f"lanl_{i:016x}" for i in range(rows)], dtype="S32"),
        features=X,
        labels=y,
        scores=scores,
        user=rng.integers(0, len(users), rows).astype(np.int32),
        source=rng.integers(0, len(hosts), rows).astype(np.int32),
        destination=rng.integers(0, len(hosts), rows).astype(np.int32),
        success=rng.random(rows) > 0.1,
        users=np.asarray(users),
        hosts=np.asarray(hosts),
    )
    np.savez_compressed(
        directory / "w1.npz",
        validation_scores=scores[:150],
        validation_labels=y[:150],
        validation_timestamps=timestamps[:150],
        test_scores=scores[150:300],
        test_labels=y[150:300],
        test_timestamps=timestamps[150:300],
    )
    model.save_model(str(directory / "model.ubj"))
    metrics = evaluate_at_threshold(y, scores, THRESHOLD).as_dict()
    manifest = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": {"auth": {"bytes": 1}, "redteam": {"bytes": 1}, "redteam_records": 3},
        "windows": {
            "w1": {"emit_end": 500.0, "boundaries": {"train_end": 300.0, "validation_end": 400.0}},
            "w2": {"emit_start": START, "emit_end": END, "context_events": 10, "emitted_events": rows},
        },
        "model": {
            "name": "test model",
            "selection": "test",
            "digest": model_digest(model),
            "threshold": THRESHOLD,
            "threshold_rule": "test",
            "feature_names": list(FEATURES),
            "train_rows": 1,
            "train_positives": 1,
            "validation_rows": 1,
            "validation_positives": 1,
        },
        "metrics": {"w1_validation": metrics, "w1_test": metrics, "w2": metrics},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.fixture()
def export_dir(tmp_path):
    return build_export(tmp_path)


@pytest.fixture()
def store(export_dir):
    return DashboardStore(export_dir, verify_rows=64)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ----------------------------------------------------------------------
# Store integrity and gating
# ----------------------------------------------------------------------


def test_store_verifies_scores_against_the_loaded_model(store):
    assert store.integrity.scores_identical
    assert store.integrity.rows_checked >= store.alert_rows.size
    assert isinstance(store.integrity.digest_matches, bool)


def test_tampered_scores_are_refused(export_dir):
    with np.load(export_dir / "w2.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["scores"] = arrays["scores"].copy()
    arrays["scores"][int(np.argmax(arrays["scores"]))] -= 0.25
    np.savez_compressed(export_dir / "w2.npz", **arrays)
    with pytest.raises(StoreError, match="differ"):
        DashboardStore(export_dir)


def test_missing_export_is_a_clear_error(tmp_path):
    with pytest.raises(StoreError, match="build_dashboard_data"):
        DashboardStore(tmp_path / "absent")


def test_alerts_are_exactly_the_rows_at_or_above_the_threshold(store):
    assert np.array_equal(store.alert_rows, np.flatnonzero(store.scores >= THRESHOLD))
    assert store.alert_rows.size > 0


def test_counters_only_cover_the_replayed_prefix(store):
    until = START + 500.0
    cursor = store.cursor(until)
    counters = store.counters(until)
    revealed = store.alert_rows[store.alert_rows < cursor]
    tp = int(store.labels[revealed].sum())

    assert counters["events"] == cursor == int(np.count_nonzero(store.timestamps <= until))
    assert counters["alerts"] == revealed.size
    assert counters["evaluation"]["true_positives"] == tp
    assert counters["evaluation"]["positives_seen"] == int(store.labels[:cursor].sum())


def test_alert_listing_never_reveals_the_future(store):
    until = START + 700.0
    listing = store.alerts(until=until, limit=1000)
    assert listing["items"]
    assert all(item["timestamp"] <= until for item in listing["items"])
    assert listing["revealed"] == int(np.count_nonzero(store.timestamps[store.alert_rows] <= until))

    future = int(store.alert_rows[-1])
    assert store.alert_row(store.event_id(future), until=store.timestamps[future] - 1) is None
    assert store.row_for_event(store.event_id(future), until=store.timestamps[future]) == future


def test_alert_filters_compose(store):
    everything = store.alerts(until=END, limit=1000)
    gt_only = store.alerts(until=END, ground_truth=True, limit=1000)
    by_user = store.alerts(until=END, user="u1", limit=1000)

    assert all(item["ground_truth"] for item in gt_only["items"])
    assert all(item["user"] == "U1" for item in by_user["items"])
    assert sum(everything["severity_counts"].values()) == everything["total"]
    severity = next(k for k, v in everything["severity_counts"].items() if v)
    only = store.alerts(until=END, severity=severity, limit=1000)
    assert only["total"] == everything["severity_counts"][severity]


def test_severity_bands():
    assert severity_for(0.01, 0.015) is None
    assert severity_for(0.02, 0.015) == "low"
    assert severity_for(0.2, 0.015) == "medium"
    assert severity_for(0.6, 0.015) == "high"
    assert severity_for(0.95, 0.015) == "critical"


def test_triage_persists_and_validates(export_dir):
    store = DashboardStore(export_dir)
    event_id = store.event_id(int(store.alert_rows[0]))
    store.set_triage(event_id, status="investigating", note="checking host")

    reloaded = DashboardStore(export_dir)
    assert reloaded.alert_record(int(store.alert_rows[0]))["status"] == "investigating"
    with pytest.raises(ValueError):
        store.set_triage(event_id, status="deleted")
    with pytest.raises(KeyError):
        store.set_triage("lanl_notanalert", status="resolved")


def test_explanation_is_locally_accurate(store):
    row = int(store.alert_rows[0])
    explanation = store.explain(row)
    total = sum(c["contribution"] for c in explanation["contributions"]) + explanation["bias"]

    assert total == pytest.approx(explanation["margin"], abs=1e-4)
    assert 1.0 / (1.0 + np.exp(-explanation["margin"])) == pytest.approx(store.scores[row], abs=1e-6)
    assert len(explanation["contributions"]) == len(FEATURES)


# ----------------------------------------------------------------------
# Replay clock
# ----------------------------------------------------------------------


def test_replay_clock_advances_pauses_seeks_and_finishes():
    fake = FakeTime()
    clock = ReplayClock(START, END, speed=5.0, now=fake)
    fake.now = 10.0
    assert clock.position() == START + 50.0

    clock.pause()
    fake.now = 100.0
    assert clock.position() == START + 50.0

    clock.seek(START + 400.0)
    assert clock.generation == 1
    clock.play()
    clock.set_speed(20.0)
    fake.now = 110.0
    assert clock.position() == START + 400.0 + 200.0

    fake.now = 10_000.0
    assert clock.state()["finished"] and clock.position() == END
    clock.seek(-5.0)
    assert clock.position() == START
    with pytest.raises(ValueError):
        clock.set_speed(7.0)


# ----------------------------------------------------------------------
# Graph and context
# ----------------------------------------------------------------------


def test_graph_slice_aggregates_movement_and_identity_edges(store):
    until = START + 1500.0
    view = graph.graph_slice(store, t_end=until, window=600.0, until=until, max_nodes=400)
    lo = int(np.searchsorted(store.timestamps, until - 600.0, side="left"))
    hi = int(np.searchsorted(store.timestamps, until, side="right"))

    assert view["events"] == hi - lo
    movement = [e for e in view["edges"] if e["relation"] == "movement"]
    assert all(e["source"] != e["target"] for e in movement)
    expected_moves = int(np.count_nonzero(store.source[lo:hi] != store.destination[lo:hi]))
    assert sum(e["events"] for e in movement) == expected_moves
    identity = [e for e in view["edges"] if e["relation"] == "identity"]
    assert sum(e["events"] for e in identity) == hi - lo
    ids = {n["id"] for n in view["nodes"]}
    assert all(n.split(":", 1)[0] in ("user", "host") for n in ids)


def test_graph_keeps_alert_endpoints_under_a_tight_node_cap(store):
    view = graph.graph_slice(store, t_end=END, window=3600.0, until=END, max_nodes=10)
    alert_edges = [e for e in view["edges"] if e["alerts"]]
    ids = {n["id"] for n in view["nodes"]}
    for edge in alert_edges:
        assert edge["source"] in ids and edge["target"] in ids


def test_graph_is_clamped_to_the_replay_clock(store):
    until = START + 200.0
    view = graph.graph_slice(store, t_end=END, window=3600.0, until=until)
    assert view["t_end"] == until
    assert view["events"] == store.cursor(until)


def test_node_and_edge_details_match_their_events(store):
    view = graph.graph_slice(store, t_end=END, window=3600.0, until=END, max_nodes=400)
    edge = next(e for e in view["edges"] if e["relation"] == "movement")
    detail = graph.edge_detail(store, edge["id"], t_end=END, window=3600.0, until=END)
    assert detail["events"] == edge["events"]
    node = graph.node_detail(store, edge["source"], t_end=END, window=3600.0, until=END)
    assert node["events"] >= edge["events"]
    with pytest.raises(KeyError):
        graph.node_detail(store, "host:NOPE", t_end=END, window=60.0, until=END)


def test_context_is_related_and_gated(store):
    row = int(store.alert_rows[len(store.alert_rows) // 2])
    until = float(store.timestamps[row]) + 50.0
    related = context.related_events(store, row, until=until, window_seconds=600.0)
    assert related["events"]
    for item in related["events"]:
        assert item["timestamp"] <= until
        assert item["relation"]

    path = context.movement_path(store, row, until=until)
    times = [hop["timestamp"] for hop in path["hops"]]
    assert times == sorted(times)
    assert sum(1 for hop in path["hops"] if hop.get("anchor")) == 1
    assert "Heuristic" in path["note"]
    assert all(hop["timestamp"] <= until for hop in path["hops"])
    # The trace follows the alert's own account and nobody else.
    assert {hop["user"] for hop in path["hops"]} == {path["account"]} == {store.event(row)["user"]}


def test_source_fanout_lists_real_first_contacts_up_to_the_alert(store):
    row = int(store.alert_rows[len(store.alert_rows) // 2])
    fanout = context.source_fanout(store, row, lookback_seconds=1000.0)
    t = float(store.timestamps[row])
    source = store.event(row)["source"]

    lo = int(np.searchsorted(store.timestamps, t - 1000.0, side="left"))
    mask = (store.source[lo : row + 1] == store.source[row]) & (
        store.destination[lo : row + 1] != store.source[row]
    )
    expected = len(set(store.destination[lo : row + 1][mask].tolist()))

    assert fanout["distinct_destinations"] == expected
    destinations = [e["destination"] for e in fanout["first_contacts"]]
    assert len(destinations) == len(set(destinations))
    for event in fanout["first_contacts"]:
        assert event["source"] == source and event["destination"] != source
        assert event["timestamp"] <= t
    assert "Real events" in fanout["note"]


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------


def test_step_curves_are_exact_corners():
    labels = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    curves = step_curves(labels, scores)
    assert [round(p["precision"], 4) for p in curves["pr"][1:]] == [1.0, 0.6667, 0.5]
    assert [p["recall"] for p in curves["pr"][1:]] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert curves["roc"][0] == {"fpr": 0.0, "tpr": 0.0}
    assert curves["roc"][-1] == {"fpr": 1.0, "tpr": 1.0}


def test_sweep_agrees_with_the_m40_evaluator(store):
    sweep = threshold_sweep(store.labels, store.scores, 2.0, THRESHOLD)
    selected = next(row for row in sweep if row["selected"])
    reference = evaluate_at_threshold(store.labels, store.scores, THRESHOLD)
    assert (selected["tp"], selected["fp"]) == (reference.tp, reference.fp)
    assert selected["predicted_positive"] == reference.predicted_positive


def test_computed_results_cover_three_splits(store):
    results = computed_results(store)
    assert set(results["splits"]) == {"w1_validation", "w1_test", "w2"}
    assert results["kind"] == "computed"


# ----------------------------------------------------------------------
# Demo Lab
# ----------------------------------------------------------------------


def fingerprint(store) -> str:
    digest = hashlib.sha256()
    for name in ("timestamps", "features", "scores", "labels", "user", "source", "destination"):
        digest.update(getattr(store, name).tobytes())
    return digest.hexdigest()


def test_lab_runs_the_real_feature_pipeline(store):
    lab = LabSession(store.model, store.threshold, store.feature_names)
    records = lab.run_scenario("fanout_sweep", {"targets": 12, "advance": 10})

    reference = TemporalGraphFeatureExtractor()
    for record in records:
        expected = reference.process_event(
            CanonicalEvent(
                event_id=record["event_id"],
                timestamp=record["timestamp"],
                user=record["user"],
                source_host=record["source"],
                destination_host=record["destination"],
                event_type="authentication",
                success=record["success"],
            )
        ).to_vector()
        assert tuple(f["value"] for f in record["features"]) == pytest.approx(expected)
        vector = np.asarray([[f["value"] for f in record["features"]]])
        assert record["score"] == float(predict_scores(store.model, vector)[0])
        assert record["alert"] == (record["score"] >= store.threshold)
        assert record["synthetic"] is True

    assert lab.graph.number_of_events() == 12
    assert lab.graph.number_of_edges() == 24


def test_lab_never_touches_the_replay_store(store):
    before = fingerprint(store)
    lab = LabSession(store.model, store.threshold, store.feature_names)
    lab.run_scenario("benign_activity", {"users": 4, "rounds": 2})
    lab.run_scenario("failed_logons", {"attempts": 5})
    lab.explain(0)
    assert fingerprint(store) == before


def test_lab_validates_input_and_limits(store):
    lab = LabSession(store.model, store.threshold, store.feature_names)
    with pytest.raises(LabError):
        lab.add_event(user="bad user", source="H", destination="H")
    with pytest.raises(LabError):
        lab.add_event(user="u", source="H", destination="D", success="yes")
    with pytest.raises(LabError):
        lab.run_scenario("exploit", {})
    with pytest.raises(LabError):
        lab.run_scenario("fanout_sweep", {"targets": 10_000})
    lab.add_event(user="u", source="A", destination="B")
    lab.reset()
    assert lab.state()["events"] == 0


def test_lab_scenarios_are_deterministic(store):
    first = LabSession(store.model, store.threshold, store.feature_names).run_scenario("fanout_sweep", {"targets": 5})
    second = LabSession(store.model, store.threshold, store.feature_names).run_scenario("fanout_sweep", {"targets": 5})
    assert [r["score"] for r in first] == [r["score"] for r in second]


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------


def run_client(store, fake, scenario):
    async def runner():
        clock = ReplayClock(store.start, store.end, speed=1.0, playing=False, now=fake)
        clock.seek(START + 800.0)
        async with TestClient(TestServer(create_app(store, clock=clock))) as client:
            return await scenario(client)

    return asyncio.run(runner())


def test_api_status_cors_and_preflight(store):
    async def scenario(client):
        origin = {"Origin": "http://localhost:3000"}
        status = await client.get("/api/status", headers=origin)
        body = await status.json()
        preflight = await client.options("/api/alerts/x", headers=origin)
        foreign = await client.get("/api/status", headers={"Origin": "http://evil.example"})
        return status, body, preflight, foreign

    status, body, preflight, foreign = run_client(store, FakeTime(), scenario)
    assert status.status == 200
    assert status.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    assert body["mode"] == "replay" and body["model"]["integrity"]["scores_identical"]
    assert preflight.status == 204
    assert "Access-Control-Allow-Origin" not in foreign.headers


def test_api_alerts_are_gated_and_triage_works(store):
    until = START + 800.0
    revealed = int(store.alert_rows[store.timestamps[store.alert_rows] <= until][0])
    future = int(store.alert_rows[-1])

    async def scenario(client):
        listing = await (await client.get("/api/alerts?limit=1000")).json()
        hidden = await client.get(f"/api/alerts/{store.event_id(future)}")
        detail = await (await client.get(f"/api/alerts/{store.event_id(revealed)}")).json()
        explain = await (await client.get(f"/api/alerts/{store.event_id(revealed)}/explain")).json()
        ctx = await (await client.get(f"/api/alerts/{store.event_id(revealed)}/context")).json()
        triage = await client.patch(
            f"/api/alerts/{store.event_id(revealed)}", json={"status": "escalated", "note": "x"}
        )
        bad = await client.patch(f"/api/alerts/{store.event_id(revealed)}", json={"status": "nope"})
        return listing, hidden, detail, explain, ctx, triage, bad

    listing, hidden, detail, explain, ctx, triage, bad = run_client(store, FakeTime(), scenario)
    assert all(item["timestamp"] <= until for item in listing["items"])
    assert hidden.status == 404
    assert len(detail["features"]) == len(FEATURES)
    assert explain["contributions"]
    assert ctx["path"]["hops"]
    assert triage.status == 200 and bad.status == 400


def test_api_replay_control_and_stream(store):
    async def scenario(client):
        bad = await client.post("/api/replay", json={"action": "explode"})
        speed = await (await client.post("/api/replay", json={"action": "speed", "value": 60})).json()
        response = await client.get("/api/stream")
        chunk = await response.content.readuntil(b"\n\n")
        response.close()
        return bad, speed, chunk

    bad, speed, chunk = run_client(store, FakeTime(), scenario)
    assert bad.status == 400
    assert speed["speed"] == 60.0
    text = chunk.decode()
    assert text.startswith("event: tick")
    tick = json.loads(text.split("data: ", 1)[1])
    assert tick["counters"]["events"] == store.cursor(START + 800.0)
    assert all(event["timestamp"] <= START + 800.0 for event in tick["feed"])


def test_api_lab_round_trip_and_isolation(store):
    before = fingerprint(store)

    async def scenario(client):
        scenario_response = await (
            await client.post("/api/lab/scenario", json={"kind": "fanout_sweep", "params": {"targets": 6}})
        ).json()
        single = await client.post("/api/lab/event", json={"user": "a", "source": "B", "destination": "C"})
        invalid = await client.post("/api/lab/event", json={"user": "a b", "source": "B", "destination": "C"})
        explain = await (await client.get("/api/lab/events/0/explain")).json()
        lab_graph = await (await client.get("/api/lab/graph")).json()
        overview = await (await client.get("/api/overview")).json()
        return scenario_response, single, invalid, explain, lab_graph, overview

    scenario_response, single, invalid, explain, lab_graph, overview = run_client(store, FakeTime(), scenario)
    assert len(scenario_response["events"]) == 6
    assert scenario_response["state"]["synthetic"] is True
    assert single.status == 200 and invalid.status == 400
    assert explain["synthetic"] is True
    assert lab_graph["nodes"]
    assert overview["counters"]["events"] == store.cursor(START + 800.0)
    assert fingerprint(store) == before
