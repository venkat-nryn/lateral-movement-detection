"""Authorized local Demo Lab: synthetic events through the real pipeline.

What it is
----------
An isolated sandbox that lets an analyst *type in* authentication records and
watch them flow through the project's own components:

    CanonicalEvent (M2.0) -> TemporalGraphFeatureExtractor, 17 features (M3.4)
    -> TemporalGraph update (M2.1) -> frozen XGBoost score (M4.0/M4.6)
    -> alert at the validation threshold -> exact TreeSHAP (M4.5)

What it is not
--------------
* It performs no network activity, authentication, exploitation or system
  change of any kind. An "event" is a Python record the session creates.
* Lab events are synthetic. RESEARCH_CONSTRAINTS section 3 forbids using them
  as research evidence, so the session shares nothing mutable with the replay
  store: it has its own extractor, graph, clock and alert list, and every
  response carries ``synthetic: true``.
* The frozen model was trained on real LANL W1 traffic. Lab events are out of
  distribution, so a lab score demonstrates the mechanics of the pipeline, not
  its detection performance. Nothing is tuned to make lab alerts fire.
"""

from __future__ import annotations

import re
import time

import numpy as np

from backend.services.store import severity_for
from ml.baselines.xgboost_baseline import predict_scores
from ml.evaluation.explainability import explain_tree_model
from ml.graph.temporal_graph import TemporalGraph, host_node_id, user_node_id
from ml.preprocessing.features import TemporalGraphFeatureExtractor
from ml.preprocessing.schema import CanonicalEvent

MAX_LAB_EVENTS = 20_000
MAX_BATCH = 300
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.@$\-]{1,40}$")

SCENARIOS = {
    "benign_activity": "Users log on to their own workstations and a few shared servers.",
    "lateral_hop": "One account moves from one host to another.",
    "fanout_sweep": "One source host authenticates to many hosts it has never reached before.",
    "failed_logons": "Repeated failed authentications from one account to one host.",
}


class LabError(ValueError):
    """Raised for invalid lab input or when the lab's safety limits are hit."""


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise LabError(
            f"{field} must be 1-40 characters of letters, digits, '_', '.', '@', '$' or '-'"
        )
    return value


def _bounded_int(value: object, field: str, low: int, high: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LabError(f"{field} must be an integer in [{low}, {high}]")
    return value


def _bounded_float(value: object, field: str, low: float, high: float, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise LabError(f"{field} must be a number in [{low}, {high}]")
    return float(value)


class LabSession:
    """One isolated lab run over the frozen detector."""

    def __init__(self, model, threshold: float, feature_names: tuple[str, ...]) -> None:
        self._model = model
        self.threshold = float(threshold)
        self.feature_names = tuple(feature_names)
        self.reset()

    def reset(self) -> None:
        self.extractor = TemporalGraphFeatureExtractor()
        self.graph = TemporalGraph()
        self.clock = 0.0
        self.events: list[dict] = []
        self._vectors: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def add_event(
        self,
        *,
        user: str,
        source: str,
        destination: str,
        success: bool = True,
        advance: float = 1.0,
    ) -> dict:
        user = _identifier(user, "user")
        source = _identifier(source, "source")
        destination = _identifier(destination, "destination")
        if not isinstance(success, bool):
            raise LabError("success must be true or false")
        advance = _bounded_float(advance, "advance", 0.0, 86_400.0, 1.0)
        if len(self.events) >= MAX_LAB_EVENTS:
            raise LabError(f"the lab holds at most {MAX_LAB_EVENTS} events; reset it first")

        self.clock += advance
        index = len(self.events)
        event = CanonicalEvent(
            event_id=f"lab-{index:06d}",
            timestamp=self.clock,
            user=user,
            source_host=source,
            destination_host=destination,
            event_type="authentication",
            success=success,
        )

        started = time.perf_counter()
        features = self.extractor.process_event(event)
        extracted = time.perf_counter()
        self.graph.add_event(event)
        graphed = time.perf_counter()
        vector = np.asarray([features.to_vector()], dtype=np.float64)
        score = float(predict_scores(self._model, vector)[0])
        scored = time.perf_counter()

        record = {
            "index": index,
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "user": user,
            "source": source,
            "destination": destination,
            "success": success,
            "score": score,
            "alert": score >= self.threshold,
            "severity": severity_for(score, self.threshold),
            "features": [
                {"name": name, "value": float(value)}
                for name, value in zip(self.feature_names, vector[0])
            ],
            "stage_ms": {
                "features": round((extracted - started) * 1000, 3),
                "graph": round((graphed - extracted) * 1000, 3),
                "score": round((scored - graphed) * 1000, 3),
            },
            "synthetic": True,
        }
        self.events.append(record)
        self._vectors.append(vector[0])
        return record

    def run_scenario(self, kind: str, params: dict | None = None) -> list[dict]:
        params = dict(params or {})
        if kind not in SCENARIOS:
            raise LabError(f"unknown scenario {kind!r}; expected one of {sorted(SCENARIOS)}")
        plan = getattr(self, f"_plan_{kind}")(params)
        if len(plan) > MAX_BATCH:
            raise LabError(f"a scenario may create at most {MAX_BATCH} events")
        if len(self.events) + len(plan) > MAX_LAB_EVENTS:
            raise LabError(f"the lab holds at most {MAX_LAB_EVENTS} events; reset it first")
        return [self.add_event(**step) for step in plan]

    # Scenario plans are deterministic lists of add_event keyword arguments.
    def _plan_benign_activity(self, params: dict) -> list[dict]:
        users = _bounded_int(params.get("users"), "users", 1, 50, 8)
        rounds = _bounded_int(params.get("rounds"), "rounds", 1, 30, 4)
        advance = _bounded_float(params.get("advance"), "advance", 0.0, 3600.0, 15.0)
        plan = []
        for r in range(rounds):
            for u in range(users):
                workstation = f"LAB-WS-{u:02d}"
                plan.append(dict(user=f"lab.user{u:02d}", source=workstation, destination=workstation, advance=advance))
                server = f"LAB-SRV-{(u + r) % 3}"
                plan.append(dict(user=f"lab.user{u:02d}", source=workstation, destination=server, advance=advance))
        return plan

    def _plan_lateral_hop(self, params: dict) -> list[dict]:
        return [
            dict(
                user=_identifier(params.get("user", "lab.admin"), "user"),
                source=_identifier(params.get("source", "LAB-WS-00"), "source"),
                destination=_identifier(params.get("destination", "LAB-SRV-9"), "destination"),
                success=bool(params.get("success", True)),
                advance=_bounded_float(params.get("advance"), "advance", 0.0, 3600.0, 30.0),
            )
        ]

    def _plan_fanout_sweep(self, params: dict) -> list[dict]:
        source = _identifier(params.get("source", "LAB-ATTACK-01"), "source")
        user = _identifier(params.get("user", "lab.svc"), "user")
        targets = _bounded_int(params.get("targets"), "targets", 1, 200, 40)
        advance = _bounded_float(params.get("advance"), "advance", 0.0, 3600.0, 20.0)
        prefix = _identifier(params.get("target_prefix", "LAB-TGT"), "target_prefix")
        return [
            dict(user=user, source=source, destination=f"{prefix}-{k:03d}", advance=advance)
            for k in range(targets)
        ]

    def _plan_failed_logons(self, params: dict) -> list[dict]:
        attempts = _bounded_int(params.get("attempts"), "attempts", 1, 50, 10)
        return [
            dict(
                user=_identifier(params.get("user", "lab.user00"), "user"),
                source=_identifier(params.get("source", "LAB-WS-00"), "source"),
                destination=_identifier(params.get("destination", "LAB-SRV-0"), "destination"),
                success=False,
                advance=_bounded_float(params.get("advance"), "advance", 0.0, 3600.0, 5.0),
            )
            for _ in range(attempts)
        ]

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def explain(self, index: int) -> dict:
        if not 0 <= index < len(self.events):
            raise KeyError(index)
        vector = self._vectors[index][None, :]
        explanation = explain_tree_model(self._model, vector, self.feature_names)
        contributions = sorted(
            (
                {"name": name, "value": float(value), "contribution": float(phi)}
                for name, value, phi in zip(self.feature_names, vector[0], explanation.contributions[0])
            ),
            key=lambda item: (-abs(item["contribution"]), item["name"]),
        )
        return {
            "method": "exact TreeSHAP (XGBoost pred_contribs), log-odds space",
            "bias": float(explanation.bias[0]),
            "margin": float(explanation.margins[0]),
            "score": self.events[index]["score"],
            "contributions": contributions,
            "synthetic": True,
        }

    def graph_view(self, max_nodes: int = 300) -> dict:
        alert_ids = {e["event_id"] for e in self.events if e["alert"]}
        edges: dict[tuple[str, str, str], dict] = {}
        for u, v, key, attrs in self.graph.iter_edges():
            relation = "identity" if u.startswith("user:") else "movement"
            if relation == "movement" and u == v:
                continue
            entry = edges.setdefault(
                (relation, u, v),
                {"id": f"{relation}|{u}|{v}", "relation": relation, "source": u, "target": v, "events": 0, "alerts": 0},
            )
            entry["events"] += 1
            entry["alerts"] += int(key in alert_ids)
        nodes = []
        for node_id, attrs in self.graph.iter_nodes():
            nodes.append({"id": node_id, "kind": attrs["entity_type"], "label": attrs["identifier"]})
        return {
            "nodes": nodes[:max_nodes],
            "edges": [e for e in edges.values() if e["source"] in {n["id"] for n in nodes[:max_nodes]}
                      and e["target"] in {n["id"] for n in nodes[:max_nodes]}],
            "synthetic": True,
        }

    def state(self, limit: int = 200) -> dict:
        alerts = [e for e in self.events if e["alert"]]
        return {
            "synthetic": True,
            "clock": self.clock,
            "events": len(self.events),
            "alerts": len(alerts),
            "threshold": self.threshold,
            "graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
            },
            "recent": self.events[::-1][:limit],
            "alert_events": alerts[::-1][:limit],
            "scenarios": SCENARIOS,
            "limits": {"events": MAX_LAB_EVENTS, "batch": MAX_BATCH},
        }


__all__ = ["LabError", "LabSession", "SCENARIOS", "host_node_id", "user_node_id"]
