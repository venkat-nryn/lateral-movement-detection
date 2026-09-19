"""aiohttp API for the SOC dashboard.

Every route reads from the verified :class:`DashboardStore` or the isolated
:class:`LabSession`. Replay-backed routes are gated by the replay clock, so
nothing the replay has not yet reached is ever served. ``/api/lab/*`` is the
only route family that touches synthetic data, and it never touches the store.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from aiohttp import web

from backend.config.settings import ALLOWED_ORIGINS
from backend.services import context, graph
from backend.services.experiments import computed_results
from backend.services.lab import LabError, LabSession
from backend.services.recorded_results import recorded_results
from backend.services.replay import ReplayClock
from backend.services.store import SEVERITY_BANDS, TRIAGE_STATUSES, DashboardStore

TICK_SECONDS = 0.5
FEED_SAMPLE = 14
TICK_ALERT_CAP = 100

STORE = web.AppKey("store", DashboardStore)
CLOCK = web.AppKey("clock", ReplayClock)
LAB = web.AppKey("lab", LabSession)
STARTED = web.AppKey("started", float)
CACHE = web.AppKey("cache", dict)


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def dumps(payload) -> str:
    return json.dumps(payload, default=_default, allow_nan=False)


def ok(payload, status: int = 200) -> web.Response:
    return web.Response(text=dumps(payload), status=status, content_type="application/json")


def error(status: int, message: str) -> web.Response:
    return ok({"error": message}, status=status)


def _cors_headers(request: web.Request) -> dict[str, str]:
    origin = request.headers.get("Origin")
    if origin not in ALLOWED_ORIGINS:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Vary": "Origin",
    }


@web.middleware
async def cors_and_errors(request: web.Request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_cors_headers(request))
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        response = error(exc.status, exc.reason)
    except (ValueError, LabError) as exc:
        response = error(400, str(exc))
    except KeyError as exc:
        response = error(404, f"not found: {exc.args[0] if exc.args else ''}")
    if not response.prepared:
        response.headers.update(_cors_headers(request))
    return response


def _float(request: web.Request, name: str, default: float | None = None) -> float | None:
    raw = request.query.get(name)
    if raw in (None, ""):
        return default
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int(request: web.Request, name: str, default: int, low: int, high: int) -> int:
    raw = request.query.get(name)
    value = default if raw in (None, "") else int(raw)
    return max(low, min(high, value))


async def _json_body(request: web.Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("request body must be JSON")
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _position(request: web.Request) -> float:
    return request.app[CLOCK].position()


# ----------------------------------------------------------------------
# Status, overview and replay
# ----------------------------------------------------------------------
async def health(request: web.Request) -> web.Response:
    return ok({"ok": True})


async def status(request: web.Request) -> web.Response:
    store = request.app[STORE]
    manifest = store.manifest
    return ok(
        {
            "mode": "replay",
            "data_source": "real LANL W2 authentication events (historical, replayed)",
            "uptime_seconds": time.time() - request.app[STARTED],
            "replay": request.app[CLOCK].state(),
            "model": {
                **{k: manifest["model"][k] for k in ("name", "selection", "digest", "threshold", "threshold_rule", "train_rows", "train_positives", "validation_rows", "validation_positives")},
                "features": len(store.feature_names),
                "integrity": {
                    "rows_checked": store.integrity.rows_checked,
                    "scores_identical": store.integrity.scores_identical,
                    "digest_matches": store.integrity.digest_matches,
                },
            },
            "export": {
                "created_at": manifest["created_at"],
                "rows": store.rows,
                "alerts_total": int(store.alert_rows.size),
                "users": len(store.users),
                "hosts": len(store.hosts),
                "window": manifest["windows"]["w2"],
            },
            "severity_bands": [{"name": n, "min_score": c} for n, c in SEVERITY_BANDS]
            + [{"name": "low", "min_score": store.threshold}],
            "severity_note": "Severity is a fixed presentation band on the model score, not a model output.",
            "triage_statuses": list(TRIAGE_STATUSES),
        }
    )


async def overview(request: web.Request) -> web.Response:
    store = request.app[STORE]
    position = _position(request)
    metrics = store.manifest["metrics"]
    return ok(
        {
            "position": position,
            "counters": store.counters(position),
            "activity": store.activity(position),
            "recent_alerts": store.alerts(until=position, limit=8)["items"],
            "reference_metrics": {
                "w1_test": metrics["w1_test"],
                "w2_full": metrics["w2"],
                "note": "Computed by the export over complete splits; W2 is the full window being replayed.",
            },
        }
    )


async def replay_control(request: web.Request) -> web.Response:
    clock = request.app[CLOCK]
    body = await _json_body(request)
    action = body.get("action")
    if action == "play":
        clock.play()
    elif action == "pause":
        clock.pause()
    elif action == "restart":
        clock.seek(clock.start)
        clock.play()
    elif action == "seek":
        clock.seek(float(body["value"]))
    elif action == "speed":
        clock.set_speed(float(body["value"]))
    else:
        raise ValueError("action must be play, pause, restart, seek or speed")
    return ok(clock.state())


def _tick(store: DashboardStore, start: int, stop: int, state: dict, reset: bool) -> dict:
    lo = int(np.searchsorted(store.alert_rows, start, side="left"))
    hi = int(np.searchsorted(store.alert_rows, stop, side="left"))
    alert_rows = store.alert_rows[lo:hi][-TICK_ALERT_CAP:]
    feed_rows = range(max(start, stop - FEED_SAMPLE), stop)
    return {
        "replay": state,
        "reset": reset,
        "new_events": stop - start,
        "new_alerts": hi - lo,
        "alerts": [store.alert_record(int(r)) for r in alert_rows],
        "feed": [store.event(r) for r in feed_rows],
        "counters": store.counters(state["position"]),
    }


async def stream(request: web.Request) -> web.StreamResponse:
    """Server-sent events: one tick of replay progress every half second."""
    store, clock = request.app[STORE], request.app[CLOCK]
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            **_cors_headers(request),
        }
    )
    await response.prepare(request)
    cursor: int | None = None
    generation: int | None = None
    try:
        while True:
            state = clock.state()
            new_cursor = store.cursor(state["position"])
            reset = cursor is None or generation != state["generation"] or new_cursor < cursor
            start = max(0, new_cursor - FEED_SAMPLE) if reset else cursor
            payload = _tick(store, start, new_cursor, state, reset)
            if reset:
                payload["new_events"] = 0
                payload["new_alerts"] = 0
                payload["alerts"] = []
            await response.write(f"event: tick\ndata: {dumps(payload)}\n\n".encode("utf-8"))
            cursor, generation = new_cursor, state["generation"]
            await asyncio.sleep(TICK_SECONDS)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def recent_events(request: web.Request) -> web.Response:
    store = request.app[STORE]
    return ok({"items": store.recent_events(_position(request), _int(request, "limit", 50, 1, 500))})


# ----------------------------------------------------------------------
# Alerts and investigation
# ----------------------------------------------------------------------
async def alerts(request: web.Request) -> web.Response:
    store = request.app[STORE]
    q = request.query
    gt = q.get("ground_truth")
    return ok(
        store.alerts(
            until=_position(request),
            user=q.get("user") or None,
            source=q.get("source") or None,
            destination=q.get("destination") or None,
            severity=q.get("severity") or None,
            status=q.get("status") or None,
            ground_truth=None if gt in (None, "") else gt == "true",
            t_from=_float(request, "from"),
            t_to=_float(request, "to"),
            query=q.get("q") or None,
            limit=_int(request, "limit", 100, 1, 1000),
            offset=_int(request, "offset", 0, 0, 10_000_000),
            newest_first=q.get("order", "desc") != "asc",
        )
    )


def _revealed_row(request: web.Request) -> int:
    store = request.app[STORE]
    event_id = request.match_info["event_id"]
    row = store.row_for_event(event_id, _position(request))
    if row is None:
        raise KeyError(f"{event_id} (unknown, or not yet reached by the replay)")
    return row


async def alert_detail(request: web.Request) -> web.Response:
    store = request.app[STORE]
    row = _revealed_row(request)
    record = store.alert_record(row) if store.scores[row] >= store.threshold else store.event(row)
    return ok(
        {
            "event": record,
            "features": store.feature_rows(row),
            "threshold": store.threshold,
            "ground_truth_note": "Exact 4-field redteam match (research ground truth; unavailable to a production SOC).",
        }
    )


async def alert_explain(request: web.Request) -> web.Response:
    store = request.app[STORE]
    return ok(store.explain(_revealed_row(request)))


async def alert_context(request: web.Request) -> web.Response:
    store = request.app[STORE]
    row = _revealed_row(request)
    position = _position(request)
    return ok(
        {
            "related": context.related_events(
                store, row, until=position, window_seconds=_float(request, "window", 1800.0) or 1800.0
            ),
            "path": context.movement_path(store, row, until=position),
            "fanout": context.source_fanout(store, row),
        }
    )


async def alert_triage(request: web.Request) -> web.Response:
    store = request.app[STORE]
    row = _revealed_row(request)
    body = await _json_body(request)
    entry = store.set_triage(
        store.event_id(row), status=str(body.get("status", "")), note=str(body.get("note", ""))
    )
    return ok({"event_id": store.event_id(row), **entry})


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------
async def graph_view(request: web.Request) -> web.Response:
    store = request.app[STORE]
    position = _position(request)
    return ok(
        graph.graph_slice(
            store,
            t_end=_float(request, "t_end", position),
            window=_float(request, "window", 300.0),
            until=position,
            max_nodes=_int(request, "max_nodes", 150, 10, graph.MAX_NODES),
            focus=request.query.get("focus") or None,
        )
    )


async def graph_node(request: web.Request) -> web.Response:
    store = request.app[STORE]
    position = _position(request)
    return ok(
        graph.node_detail(
            store,
            request.query["id"],
            t_end=_float(request, "t_end", position),
            window=_float(request, "window", 300.0),
            until=position,
        )
    )


async def graph_edge(request: web.Request) -> web.Response:
    store = request.app[STORE]
    position = _position(request)
    return ok(
        graph.edge_detail(
            store,
            request.query["id"],
            t_end=_float(request, "t_end", position),
            window=_float(request, "window", 300.0),
            until=position,
        )
    )


# ----------------------------------------------------------------------
# Experiments and pipeline
# ----------------------------------------------------------------------
async def experiments(request: web.Request) -> web.Response:
    cache = request.app[CACHE]
    if "experiments" not in cache:
        cache["experiments"] = {
            "computed": computed_results(request.app[STORE]),
            "recorded": recorded_results(),
        }
    return ok(cache["experiments"])


async def pipeline(request: web.Request) -> web.Response:
    store, lab = request.app[STORE], request.app[LAB]
    manifest = store.manifest
    counters = store.counters(_position(request))
    w2 = manifest["windows"]["w2"]
    return ok(
        {
            "stages": [
                {"key": "lanl", "title": "LANL dataset", "detail": "auth.txt.gz + redteam.txt.gz (read-only)",
                 "stats": {"auth_bytes": manifest["source"]["auth"]["bytes"], "redteam_records": manifest["source"]["redteam_records"]}},
                {"key": "ingestion", "title": "Ingestion", "detail": "Streaming gzip adapter, bounded windows (M3.1, M3.6)",
                 "stats": {"w2_context_events": w2["context_events"], "w2_emitted_events": w2["emitted_events"]}},
                {"key": "canonical", "title": "Canonical events", "detail": "Validated CanonicalEvent records (M2.0)",
                 "stats": {"replayed": counters["events"], "users": len(store.users), "hosts": len(store.hosts)}},
                {"key": "features", "title": "Temporal features", "detail": "17 causal behavioural features (M3.4)",
                 "stats": {"features": len(store.feature_names)}},
                {"key": "graph", "title": "Temporal graph", "detail": "User/host multigraph (M2.1/M4.2). Drives the graph view and the GNN research; not an input to the served detector, because graph embeddings did not help (section 36).",
                 "stats": {"node_types": 2, "edges_per_event": 2}},
                {"key": "model", "title": "Detection model", "detail": manifest["model"]["name"] + ", frozen after W1 (M4.0/M4.6)",
                 "stats": {"threshold": store.threshold, "integrity_rows": store.integrity.rows_checked}},
                {"key": "explain", "title": "Explainability", "detail": "Exact TreeSHAP per event (M4.5)", "stats": {}},
                {"key": "alert", "title": "Alert", "detail": "Score at or above the validation threshold",
                 "stats": {"alerts_so_far": counters["alerts"], "alerts_total": int(store.alert_rows.size)}},
            ],
            "lab": {"events": len(lab.events), "alerts": sum(1 for e in lab.events if e["alert"])},
        }
    )


# ----------------------------------------------------------------------
# Demo Lab (synthetic, isolated)
# ----------------------------------------------------------------------
async def lab_state(request: web.Request) -> web.Response:
    return ok(request.app[LAB].state())


async def lab_reset(request: web.Request) -> web.Response:
    request.app[LAB].reset()
    return ok(request.app[LAB].state())


async def lab_event(request: web.Request) -> web.Response:
    body = await _json_body(request)
    record = request.app[LAB].add_event(
        user=body.get("user"),
        source=body.get("source"),
        destination=body.get("destination"),
        success=body.get("success", True),
        advance=body.get("advance", 1.0),
    )
    return ok({"events": [record], "state": request.app[LAB].state(limit=50)})


async def lab_scenario(request: web.Request) -> web.Response:
    body = await _json_body(request)
    records = request.app[LAB].run_scenario(str(body.get("kind", "")), body.get("params") or {})
    return ok({"events": records, "state": request.app[LAB].state(limit=50)})


async def lab_explain(request: web.Request) -> web.Response:
    return ok(request.app[LAB].explain(int(request.match_info["index"])))


async def lab_graph(request: web.Request) -> web.Response:
    return ok(request.app[LAB].graph_view())


def create_app(store: DashboardStore, *, clock: ReplayClock | None = None) -> web.Application:
    app = web.Application(middlewares=[cors_and_errors])
    app[STORE] = store
    app[CLOCK] = clock or ReplayClock(store.start, store.end)
    app[LAB] = LabSession(store.model, store.threshold, store.feature_names)
    app[STARTED] = time.time()
    app[CACHE] = {}
    app.add_routes(
        [
            web.get("/api/health", health),
            web.get("/api/status", status),
            web.get("/api/overview", overview),
            web.post("/api/replay", replay_control),
            web.get("/api/stream", stream),
            web.get("/api/events/recent", recent_events),
            web.get("/api/alerts", alerts),
            web.get("/api/alerts/{event_id}", alert_detail),
            web.patch("/api/alerts/{event_id}", alert_triage),
            web.get("/api/alerts/{event_id}/explain", alert_explain),
            web.get("/api/alerts/{event_id}/context", alert_context),
            web.get("/api/graph", graph_view),
            web.get("/api/graph/node", graph_node),
            web.get("/api/graph/edge", graph_edge),
            web.get("/api/experiments", experiments),
            web.get("/api/pipeline", pipeline),
            web.get("/api/lab/state", lab_state),
            web.post("/api/lab/reset", lab_reset),
            web.post("/api/lab/event", lab_event),
            web.post("/api/lab/scenario", lab_scenario),
            web.get("/api/lab/events/{index}/explain", lab_explain),
            web.get("/api/lab/graph", lab_graph),
        ]
    )
    return app
