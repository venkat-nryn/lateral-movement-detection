"""Bounded temporal graph views over the replayed W2 events.

Nodes and edges follow the M2.1 temporal-graph model: typed, namespaced nodes
(``user:<id>``, ``host:<id>``) and, per event, an identity edge
``user -> destination`` plus a movement edge ``source -> destination``. Views
aggregate the events of one time window into weighted edges so the browser
receives a readable graph instead of hundreds of thousands of raw edges.
Movement self-loops (local logons) count toward node activity but are not drawn.
"""

from __future__ import annotations

import numpy as np

from backend.services.store import DashboardStore
from ml.graph.temporal_graph import host_node_id, user_node_id

MAX_WINDOW_SECONDS = 3600.0
MAX_NODES = 400
MAX_EDGES = 1200


def _window(store: DashboardStore, t_end: float, window: float, until: float) -> tuple[int, int, float, float]:
    t_end = float(min(max(t_end, store.start), until, store.end))
    window = float(min(max(window, 1.0), MAX_WINDOW_SECONDS))
    lo = int(np.searchsorted(store.timestamps, t_end - window, side="left"))
    hi = int(np.searchsorted(store.timestamps, t_end, side="right"))
    return lo, hi, t_end, window


def _parse_node(store: DashboardStore, node_id: str) -> tuple[str, int]:
    kind, _, name = node_id.partition(":")
    table = store.user_code if kind == "user" else store.host_code if kind == "host" else None
    if table is None or name not in table:
        raise KeyError(node_id)
    return kind, table[name]


def _aggregate(first: np.ndarray, second: np.ndarray, width: int, scores, alerts, labels, times):
    keys = first.astype(np.int64) * width + second.astype(np.int64)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    max_score = np.full(unique.size, -1.0)
    np.maximum.at(max_score, inverse, scores)
    last = np.full(unique.size, -np.inf)
    np.maximum.at(last, inverse, times)
    return (
        unique // width,
        unique % width,
        counts,
        max_score,
        np.bincount(inverse, weights=alerts, minlength=unique.size).astype(np.int64),
        np.bincount(inverse, weights=labels, minlength=unique.size).astype(np.int64),
        last,
    )


def graph_slice(
    store: DashboardStore,
    *,
    t_end: float,
    window: float,
    until: float,
    max_nodes: int = 150,
    focus: str | None = None,
) -> dict:
    """Aggregated user/host graph for ``(t_end - window, t_end]``."""
    lo, hi, t_end, window = _window(store, t_end, window, until)
    max_nodes = int(min(max(max_nodes, 10), MAX_NODES))
    sl = slice(lo, hi)
    user, src, dst = store.user[sl], store.source[sl], store.destination[sl]
    scores, labels, times = store.scores[sl], store.labels[sl], store.timestamps[sl]
    alerts = (scores >= store.threshold).astype(np.int64)

    if focus:
        kind, code = _parse_node(store, focus)
        mask = (user == code) if kind == "user" else ((src == code) | (dst == code))
        user, src, dst, scores, labels, times, alerts = (
            a[mask] for a in (user, src, dst, scores, labels, times, alerts)
        )

    width = max(len(store.hosts), len(store.users), 1)
    moving = src != dst
    movement = _aggregate(src[moving], dst[moving], width, scores[moving], alerts[moving], labels[moving], times[moving])
    identity = _aggregate(user, dst, width, scores, alerts, labels, times)

    # Node activity counts every event a node takes part in, local logons included.
    host_events = np.bincount(src, minlength=len(store.hosts)) + np.bincount(
        dst[moving], minlength=len(store.hosts)
    )
    host_alerts = np.bincount(src, weights=alerts, minlength=len(store.hosts)) + np.bincount(
        dst[moving], weights=alerts[moving], minlength=len(store.hosts)
    )
    user_events = np.bincount(user, minlength=len(store.users))
    user_alerts = np.bincount(user, weights=alerts, minlength=len(store.users))

    must: set[tuple[str, int]] = set()
    for first_kind, table in (("host", movement), ("user", identity)):
        a, b, _, _, alert_counts, _, _ = table
        for x, y in zip(a[alert_counts > 0], b[alert_counts > 0]):
            must.add((first_kind, int(x)))
            must.add(("host", int(y)))
    if focus:
        must.add(_parse_node(store, focus))

    ranked: list[tuple[int, str, int]] = [
        (int(host_events[h]), "host", int(h)) for h in np.flatnonzero(host_events)
    ] + [(int(user_events[u]), "user", int(u)) for u in np.flatnonzero(user_events)]
    ranked.sort(key=lambda item: -item[0])
    selected = list(must)[:max_nodes]
    chosen = set(selected)
    for _, kind, code in ranked:
        if len(chosen) >= max_nodes:
            break
        if (kind, code) not in chosen:
            chosen.add((kind, code))

    def node_id(kind: str, code: int) -> str:
        return user_node_id(store.users[code]) if kind == "user" else host_node_id(store.hosts[code])

    nodes = [
        {
            "id": node_id(kind, code),
            "kind": kind,
            "label": store.users[code] if kind == "user" else store.hosts[code],
            "events": int(user_events[code] if kind == "user" else host_events[code]),
            "alerts": int(user_alerts[code] if kind == "user" else host_alerts[code]),
        }
        for kind, code in chosen
    ]

    edges: list[dict] = []
    for relation, first_kind, table in (("movement", "host", movement), ("identity", "user", identity)):
        a, b, counts, max_score, alert_counts, gt_counts, last = table
        for i in range(a.size):
            if (first_kind, int(a[i])) not in chosen or ("host", int(b[i])) not in chosen:
                continue
            source_id = node_id(first_kind, int(a[i]))
            target_id = node_id("host", int(b[i]))
            edges.append(
                {
                    "id": f"{relation}|{source_id}|{target_id}",
                    "relation": relation,
                    "source": source_id,
                    "target": target_id,
                    "events": int(counts[i]),
                    "alerts": int(alert_counts[i]),
                    "ground_truth": int(gt_counts[i]),
                    "max_score": float(max_score[i]),
                    "last_seen": float(last[i]),
                }
            )
    edges.sort(key=lambda e: (-e["alerts"], -e["events"]))
    truncated_edges = len(edges) > MAX_EDGES
    edges = edges[:MAX_EDGES]

    return {
        "t_start": t_end - window,
        "t_end": t_end,
        "window": window,
        "events": int(user.size),
        "focus": focus,
        "nodes": nodes,
        "edges": edges,
        "totals": {
            "active_hosts": int(np.count_nonzero(host_events)),
            "active_users": int(np.count_nonzero(user_events)),
            "alerts": int(alerts.sum()),
        },
        "truncated": len(ranked) > len(chosen) or truncated_edges,
    }


def node_detail(store: DashboardStore, node_id: str, *, t_end: float, window: float, until: float, limit: int = 100) -> dict:
    kind, code = _parse_node(store, node_id)
    lo, hi, t_end, window = _window(store, t_end, window, until)
    if kind == "user":
        mask = store.user[lo:hi] == code
    else:
        mask = (store.source[lo:hi] == code) | (store.destination[lo:hi] == code)
    rows = np.flatnonzero(mask) + lo
    scores = store.scores[rows]
    peers: dict[str, int] = {}
    for r in rows:
        if kind == "user":
            peer = store.hosts[store.destination[r]]
        else:
            other = store.destination[r] if store.source[r] == code else store.source[r]
            peer = store.hosts[other]
        peers[peer] = peers.get(peer, 0) + 1
    top_peers = sorted(peers.items(), key=lambda kv: -kv[1])[:10]
    return {
        "id": node_id,
        "kind": kind,
        "label": store.users[code] if kind == "user" else store.hosts[code],
        "t_start": t_end - window,
        "t_end": t_end,
        "events": int(rows.size),
        "alerts": int(np.count_nonzero(scores >= store.threshold)),
        "failures": int(np.count_nonzero(~store.success[rows])),
        "outbound": int(np.count_nonzero(store.source[rows] == code)) if kind == "host" else None,
        "inbound": int(np.count_nonzero(store.destination[rows] == code)) if kind == "host" else None,
        "distinct_peers": len(peers),
        "top_peers": [{"name": name, "events": count} for name, count in top_peers],
        "recent": [store.event(int(r)) for r in rows[::-1][:limit]],
    }


def edge_detail(store: DashboardStore, edge_id: str, *, t_end: float, window: float, until: float, limit: int = 100) -> dict:
    relation, source_id, target_id = (edge_id.split("|") + ["", ""])[:3]
    if relation not in ("movement", "identity"):
        raise KeyError(edge_id)
    first_kind, first = _parse_node(store, source_id)
    _, second = _parse_node(store, target_id)
    lo, hi, t_end, window = _window(store, t_end, window, until)
    if relation == "movement":
        mask = (store.source[lo:hi] == first) & (store.destination[lo:hi] == second)
    else:
        mask = (store.user[lo:hi] == first) & (store.destination[lo:hi] == second)
    rows = np.flatnonzero(mask) + lo
    return {
        "id": edge_id,
        "relation": relation,
        "source": source_id,
        "target": target_id,
        "t_start": t_end - window,
        "t_end": t_end,
        "events": int(rows.size),
        "alerts": int(np.count_nonzero(store.scores[rows] >= store.threshold)),
        "recent": [store.event(int(r)) for r in rows[::-1][:limit]],
    }
