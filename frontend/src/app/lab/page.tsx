"use client";

import { useCallback, useEffect, useState } from "react";
import { api, post, type Explanation, type LabEvent, type LabGraph, type LabState } from "@/lib/api";
import { pct } from "@/lib/format";
import { LineChart, ShapBars } from "@/components/charts";
import { GraphCanvas } from "@/components/GraphCanvas";
import { Card, Empty, ErrorNote, Hosts, KeyValue, Provenance, ScoreBar, SeverityBadge, Spinner } from "@/components/ui";
import { PageTitle } from "@/components/ui";

type Kind = "benign_activity" | "lateral_hop" | "fanout_sweep" | "failed_logons";

const FIELDS: Record<Kind, { key: string; label: string; type: "text" | "number"; initial: string | number }[]> = {
  benign_activity: [
    { key: "users", label: "Users", type: "number", initial: 8 },
    { key: "rounds", label: "Rounds", type: "number", initial: 4 },
    { key: "advance", label: "Seconds between events", type: "number", initial: 15 },
  ],
  lateral_hop: [
    { key: "user", label: "User", type: "text", initial: "lab.admin" },
    { key: "source", label: "Source host", type: "text", initial: "LAB-WS-00" },
    { key: "destination", label: "Destination host", type: "text", initial: "LAB-SRV-9" },
    { key: "advance", label: "Seconds after previous event", type: "number", initial: 30 },
  ],
  fanout_sweep: [
    { key: "source", label: "Source host", type: "text", initial: "LAB-ATTACK-01" },
    { key: "user", label: "User", type: "text", initial: "lab.svc" },
    { key: "targets", label: "New destinations", type: "number", initial: 40 },
    { key: "advance", label: "Seconds between hops", type: "number", initial: 20 },
  ],
  failed_logons: [
    { key: "user", label: "User", type: "text", initial: "lab.user00" },
    { key: "source", label: "Source host", type: "text", initial: "LAB-WS-00" },
    { key: "destination", label: "Destination host", type: "text", initial: "LAB-SRV-0" },
    { key: "attempts", label: "Attempts", type: "number", initial: 10 },
  ],
};

const TITLES: Record<Kind, string> = {
  benign_activity: "Benign activity",
  lateral_hop: "Single lateral hop",
  fanout_sweep: "Fan-out sweep",
  failed_logons: "Failed logons",
};

const STAGES = ["Event", "Features", "Graph", "Score", "Alert"] as const;

export default function LabPage() {
  const [state, setState] = useState<LabState | null>(null);
  const [graph, setGraph] = useState<LabGraph | null>(null);
  const [kind, setKind] = useState<Kind>("fanout_sweep");
  const [params, setParams] = useState<Record<string, string | number>>({});
  const [manual, setManual] = useState({ user: "lab.analyst", source: "LAB-WS-01", destination: "LAB-SRV-1", success: true, advance: 10 });
  const [lastRun, setLastRun] = useState<LabEvent[]>([]);
  const [stage, setStage] = useState(-1);
  const [selected, setSelected] = useState<LabEvent | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [s, g] = await Promise.all([api<LabState>("/api/lab/state"), api<LabGraph>("/api/lab/graph")]);
    setState(s);
    setGraph(g);
  }, []);

  useEffect(() => {
    refresh().catch((err) => setError((err as Error).message));
  }, [refresh]);

  useEffect(() => {
    setParams(Object.fromEntries(FIELDS[kind].map((f) => [f.key, f.initial])));
  }, [kind]);

  // Walk the stage indicator through the pipeline for the newest event.
  const animate = useCallback((events: LabEvent[]) => {
    setLastRun(events);
    setStage(0);
    STAGES.forEach((_, i) => setTimeout(() => setStage(i), 180 * i));
    const last = events[events.length - 1];
    if (last) setSelected(last);
  }, []);

  const run = async (action: () => Promise<{ events: LabEvent[] }>) => {
    setBusy(true);
    setError(null);
    try {
      const result = await action();
      animate(result.events);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!selected) {
      setExplanation(null);
      return;
    }
    api<Explanation>(`/api/lab/events/${selected.index}/explain`).then(setExplanation).catch(() => setExplanation(null));
  }, [selected]);

  const typed = (field: { key: string; type: string }) => {
    const value = params[field.key];
    return field.type === "number" ? Number(value) : value;
  };

  const newest = lastRun[lastRun.length - 1];

  return (
    <div className="space-y-4">
      <PageTitle
        title="Demo Lab"
        subtitle="Push controlled test events through the project's real pipeline and watch each stage"
        actions={
          <button className="btn btn-danger" disabled={busy} onClick={() => run(async () => { await post("/api/lab/reset", {}); setSelected(null); setLastRun([]); return { events: [] }; })}>
            Reset lab
          </button>
        }
      />

      <div className="flex items-start gap-3 rounded-md border border-soc-warn/40 bg-soc-warn/[0.07] px-4 py-3">
        <svg viewBox="0 0 24 24" className="mt-0.5 h-4 w-4 shrink-0 text-soc-warn" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3 2 20h20L12 3zm0 6v5m0 3v.01" /></svg>
        <div className="text-[12.5px] leading-relaxed text-soc-muted">
          <span className="font-semibold text-soc-warn">AUTHORIZED LOCAL LAB — synthetic events only.</span> An “event” here is a
          record created inside the backend process: nothing touches a network, host, account or credential. Lab data is
          isolated from the real LANL replay and is never counted in alerts or metrics. The detector was trained on real W1
          traffic, so lab scores show how the pipeline works, not how well it detects.
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[340px_1fr]">
        <div className="space-y-4">
          <Card title="Scenario" subtitle="Deterministic templates; you set the parameters">
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-1.5">
                {(Object.keys(FIELDS) as Kind[]).map((k) => (
                  <button
                    key={k}
                    onClick={() => setKind(k)}
                    className={`rounded border px-2 py-1.5 text-left text-[12px] transition-colors ${
                      kind === k ? "border-soc-accent/70 bg-soc-accent/10 text-soc-text" : "border-soc-line text-soc-muted hover:text-soc-text"
                    }`}
                  >
                    {TITLES[k]}
                  </button>
                ))}
              </div>
              <p className="text-[11.5px] text-soc-faint">{state?.scenarios[kind]}</p>
              {FIELDS[kind].map((field) => (
                <label key={field.key} className="block">
                  <span className="label">{field.label}</span>
                  <input
                    className="input mt-1 w-full"
                    type={field.type}
                    value={params[field.key] ?? ""}
                    onChange={(e) => setParams((p) => ({ ...p, [field.key]: e.target.value }))}
                  />
                </label>
              ))}
              <button
                className="btn btn-primary w-full justify-center"
                disabled={busy}
                onClick={() =>
                  run(() =>
                    post<{ events: LabEvent[] }>("/api/lab/scenario", {
                      kind,
                      params: Object.fromEntries(FIELDS[kind].map((f) => [f.key, typed(f)])),
                    }),
                  )
                }
              >
                {busy ? "Running…" : `Run ${TITLES[kind].toLowerCase()}`}
              </button>
            </div>
          </Card>

          <Card title="Single event" subtitle="Hand-crafted authentication record">
            <form
              className="space-y-2"
              onSubmit={(e) => {
                e.preventDefault();
                void run(() => post<{ events: LabEvent[] }>("/api/lab/event", { ...manual, advance: Number(manual.advance) }));
              }}
            >
              {(["user", "source", "destination"] as const).map((key) => (
                <input key={key} className="input w-full" placeholder={key} value={manual[key]} onChange={(e) => setManual({ ...manual, [key]: e.target.value })} />
              ))}
              <div className="flex items-center gap-2">
                <input className="input w-24" type="number" value={manual.advance} onChange={(e) => setManual({ ...manual, advance: Number(e.target.value) })} />
                <span className="text-[11.5px] text-soc-faint">s later</span>
                <label className="ml-auto flex items-center gap-1.5 text-[12px] text-soc-muted">
                  <input type="checkbox" checked={!manual.success} onChange={(e) => setManual({ ...manual, success: !e.target.checked })} />
                  failed
                </label>
              </div>
              <button className="btn w-full justify-center" disabled={busy} type="submit">Send event</button>
            </form>
          </Card>
          <ErrorNote error={error} />
        </div>

        <div className="space-y-4">
          <Card title="Pipeline trace" subtitle={newest ? `Newest event ${newest.event_id}: ${newest.source} → ${newest.destination}` : "Run a scenario to trace an event"}>
            <div className="grid grid-cols-5 gap-2">
              {STAGES.map((name, i) => {
                const reached = stage >= i;
                const alert = name === "Alert" && newest?.alert;
                const value =
                  !newest ? "" :
                  name === "Event" ? newest.event_id :
                  name === "Features" ? `${newest.features.length} values · ${newest.stage_ms.features} ms` :
                  name === "Graph" ? `${state?.graph.nodes ?? 0} nodes · ${newest.stage_ms.graph} ms` :
                  name === "Score" ? `${pct(newest.score, 2)} · ${newest.stage_ms.score} ms` :
                  newest.alert ? `ALERT (${newest.severity})` : "below threshold";
                return (
                  <div
                    key={name}
                    className={`rounded-md border p-3 transition-all duration-300 ${
                      !reached ? "border-soc-line bg-soc-raised opacity-40" :
                      alert ? "border-soc-danger/70 bg-soc-danger/10" :
                      "border-soc-accent/50 bg-soc-accent/[0.07]"
                    }`}
                  >
                    <div className="label">{String(i + 1).padStart(2, "0")} · {name}</div>
                    <div className={`mono mt-1.5 truncate text-[11.5px] ${alert ? "text-soc-danger" : "text-soc-text"}`}>{value || "—"}</div>
                  </div>
                );
              })}
            </div>
            {lastRun.length > 1 && (
              <div className="mt-4">
                <LineChart
                  height={170}
                  xLabel="Event in this run"
                  yLabel="Score"
                  xDomain={[1, lastRun.length]}
                  series={[
                    { name: "detection score", color: "#38bdf8", points: lastRun.map((e, i) => ({ x: i + 1, y: e.score })) },
                    { name: `threshold ${state?.threshold.toFixed(4)}`, color: "#f59e0b", dashed: true, points: [{ x: 1, y: state?.threshold ?? 0 }, { x: lastRun.length, y: state?.threshold ?? 0 }] },
                  ]}
                />
              </div>
            )}
          </Card>

          <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
            <Card title="Lab graph" subtitle={`${state?.graph.nodes ?? 0} nodes · ${state?.graph.edges ?? 0} event edges (M2.1 TemporalGraph)`} bodyClass="p-0">
              <div className="relative h-[340px]">
                {graph && graph.nodes.length ? (
                  <GraphCanvas
                    nodes={graph.nodes}
                    edges={graph.edges}
                    showIdentity
                    highlightSuspicious={false}
                    selectedId={null}
                    fitKey={String(state?.events ?? 0)}
                    onSelectNode={() => undefined}
                    onSelectEdge={() => undefined}
                    onClear={() => undefined}
                  />
                ) : (
                  <Empty>Graph is empty</Empty>
                )}
              </div>
            </Card>
            <Card title="Explanation" subtitle={selected ? `${selected.event_id} · exact TreeSHAP` : "Select a lab event"}>
              {explanation ? (
                <ShapBars contributions={explanation.contributions} bias={explanation.bias} margin={explanation.margin} limit={8} />
              ) : selected ? (
                <Spinner />
              ) : (
                <Empty>No event selected</Empty>
              )}
            </Card>
          </div>

          <Card
            title="Lab event stream"
            subtitle={state ? `${state.events} events · ${state.alerts} alerts · clock +${state.clock.toFixed(0)} s` : ""}
            bodyClass="p-0"
          >
            <div className="max-h-[320px] overflow-y-auto">
              {state && state.recent.length ? (
                <table className="w-full text-[12px]">
                  <tbody>
                    {state.recent.map((event) => (
                      <tr
                        key={event.event_id}
                        onClick={() => setSelected(event)}
                        className={`table-row cursor-pointer ${selected?.event_id === event.event_id ? "bg-soc-accent/10" : ""}`}
                      >
                        <td className="mono px-3 py-1.5 text-soc-faint">{event.event_id}</td>
                        <td className="px-2 text-soc-muted">{event.user}</td>
                        <td className="px-2"><Hosts source={event.source} destination={event.destination} /></td>
                        <td className="px-2"><ScoreBar value={event.score} threshold={state.threshold} /></td>
                        <td className="px-2"><SeverityBadge severity={event.severity} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Empty>No lab events yet</Empty>
              )}
            </div>
          </Card>

          {selected && (
            <Card title="Features of the selected event" subtitle="Computed by the same M3.4 extractor as the real pipeline">
              <KeyValue items={selected.features.map((f) => [f.name, <span key={f.name} className="mono">{Number.isInteger(f.value) ? f.value : f.value.toFixed(2)}</span>])} />
            </Card>
          )}
          <Provenance>
            Scenarios never execute anything. They only generate CanonicalEvent records from the parameters shown, capped at{" "}
            {state?.limits.batch ?? 300} per run and {state?.limits.events.toLocaleString() ?? "20,000"} per lab session.
          </Provenance>
        </div>
      </div>
    </div>
  );
}
