"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type EdgeDetail, type GraphSlice, type NodeDetail } from "@/lib/api";
import { useReplay } from "@/lib/replay";
import { DEFAULT_THRESHOLD, useStatus } from "@/lib/status";
import { lanlTime, num } from "@/lib/format";
import { GraphCanvas } from "@/components/GraphCanvas";
import { EventTable } from "@/components/EventTable";
import { Card, Empty, ErrorNote, KeyValue, Provenance, Segmented, Spinner } from "@/components/ui";
import { PageTitle } from "@/components/ui";

const WINDOWS = [
  { value: 60, label: "1m" },
  { value: 300, label: "5m" },
  { value: 900, label: "15m" },
  { value: 1800, label: "30m" },
  { value: 3600, label: "60m" },
];
const NODE_CAPS = [
  { value: 80, label: "80" },
  { value: 150, label: "150" },
  { value: 250, label: "250" },
  { value: 400, label: "400" },
];

type Selection = { kind: "node"; id: string } | { kind: "edge"; id: string } | null;

export default function GraphPage() {
  const { replay, tickCount } = useReplay();
  const status = useStatus();
  const threshold = status?.model.threshold ?? DEFAULT_THRESHOLD;

  const [windowSeconds, setWindowSeconds] = useState(300);
  const [maxNodes, setMaxNodes] = useState(150);
  const [showIdentity, setShowIdentity] = useState(false);
  const [highlight, setHighlight] = useState(false);
  const [showTruth, setShowTruth] = useState(false);
  const [focus, setFocus] = useState("");
  const [focusInput, setFocusInput] = useState("");
  const [followLive, setFollowLive] = useState(true);
  const [tEnd, setTEnd] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [data, setData] = useState<GraphSlice | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [detail, setDetail] = useState<NodeDetail | EdgeDetail | null>(null);

  const position = replay?.position ?? null;
  const effectiveEnd = followLive ? position : tEnd;
  const requestId = useRef(0);

  const load = useCallback(async () => {
    if (effectiveEnd === null) return;
    const id = ++requestId.current;
    setLoading(true);
    try {
      const params = new URLSearchParams({
        t_end: String(effectiveEnd),
        window: String(windowSeconds),
        max_nodes: String(maxNodes),
      });
      if (focus) params.set("focus", focus);
      const slice = await api<GraphSlice>(`/api/graph?${params}`);
      if (id === requestId.current) {
        setData(slice);
        setError(null);
      }
    } catch (err) {
      if (id === requestId.current) setError((err as Error).message);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, [effectiveEnd, windowSeconds, maxNodes, focus]);

  // Follow-live refreshes every ~5 s; manual scrubbing loads on change.
  const liveBucket = followLive ? Math.floor(tickCount / 10) : 0;
  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveBucket, windowSeconds, maxNodes, focus, followLive, followLive ? null : tEnd]);

  // Timeline playback: advance t_end in steps up to the replay clock.
  useEffect(() => {
    if (!playing || followLive || position === null) return;
    const timer = setInterval(() => {
      setTEnd((current) => {
        const next = (current ?? replay!.start) + windowSeconds / 5;
        if (next >= position) {
          setPlaying(false);
          return position;
        }
        return next;
      });
    }, 1100);
    return () => clearInterval(timer);
  }, [playing, followLive, position, windowSeconds, replay]);

  // Details for the selected node or edge.
  useEffect(() => {
    if (!selection || effectiveEnd === null) {
      setDetail(null);
      return;
    }
    const params = new URLSearchParams({ id: selection.id, t_end: String(effectiveEnd), window: String(windowSeconds) });
    api<NodeDetail | EdgeDetail>(`/api/graph/${selection.kind}?${params}`)
      .then(setDetail)
      .catch((err) => setError((err as Error).message));
    // Refetch when the slice changes so the panel stays in step with the canvas.
  }, [selection, data?.t_end, windowSeconds]); // eslint-disable-line react-hooks/exhaustive-deps

  const fitKey = `${windowSeconds}|${maxNodes}|${focus}|${showIdentity}`;
  const alertEdges = useMemo(() => (data ? data.edges.filter((e) => e.alerts > 0) : []), [data]);

  const applyFocus = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) {
      setFocus("");
      return;
    }
    setFocus(trimmed.includes(":") ? trimmed : `host:${trimmed}`);
    setSelection(null);
  };

  return (
    <div className="flex h-[calc(100vh-88px)] flex-col gap-3">
      <PageTitle
        title="Network Graph"
        subtitle="User and host temporal graph over a sliding window of replayed W2 events (M2.1 model: identity and movement edges)"
      />

      <div className="panel flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="label">Window</span>
          <Segmented value={windowSeconds} options={WINDOWS} onChange={setWindowSeconds} />
        </div>
        <div className="flex items-center gap-2">
          <span className="label">Nodes</span>
          <Segmented value={maxNodes} options={NODE_CAPS} onChange={setMaxNodes} />
        </div>
        <Toggle label="Identity edges" value={showIdentity} onChange={setShowIdentity} />
        <Toggle label="Suspicious paths" value={highlight} onChange={setHighlight} tone="danger" />
        <Toggle label="Ground truth" value={showTruth} onChange={setShowTruth} tone="violet" />
        <form
          className="ml-auto flex items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            applyFocus(focusInput);
          }}
        >
          <input
            className="input w-56"
            placeholder="Focus: host C17693 or user:U66@DOM1"
            value={focusInput}
            onChange={(event) => setFocusInput(event.target.value)}
          />
          <button className="btn" type="submit">Focus</button>
          {focus && (
            <button className="btn" type="button" onClick={() => { setFocus(""); setFocusInput(""); }}>
              Clear
            </button>
          )}
        </form>
      </div>

      <div className="panel flex items-center gap-3 px-3 py-2">
        <button
          className={`btn h-7 ${followLive ? "btn-primary" : ""}`}
          onClick={() => {
            setFollowLive(!followLive);
            setPlaying(false);
            setTEnd(position);
          }}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${followLive ? "animate-pulseDot bg-soc-ok" : "bg-soc-faint"}`} />
          Follow live
        </button>
        <button
          className="btn h-7"
          disabled={followLive}
          onClick={() => {
            if (tEnd !== null && position !== null && tEnd >= position - 1 && replay) setTEnd(replay.start + windowSeconds);
            setPlaying(!playing);
          }}
        >
          {playing ? "Pause playback" : "Play timeline"}
        </button>
        <input
          type="range"
          className="flex-1 accent-sky-400"
          min={replay?.start ?? 0}
          max={position ?? 1}
          step={10}
          value={effectiveEnd ?? 0}
          disabled={!replay}
          onChange={(event) => {
            setFollowLive(false);
            setPlaying(false);
            setTEnd(Number(event.target.value));
          }}
        />
        <span className="mono w-[210px] text-right text-soc-muted">
          {data ? `${lanlTime(data.t_start)} → ${lanlTime(data.t_end, false)}` : "—"}
        </span>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[1fr_380px]">
        <div className="panel relative min-h-[420px] overflow-hidden">
          {data && (
            <GraphCanvas
              nodes={data.nodes}
              edges={data.edges}
              showIdentity={showIdentity}
              highlightSuspicious={highlight}
              showTruth={showTruth}
              selectedId={selection?.id ?? null}
              fitKey={fitKey}
              onSelectNode={(id) => setSelection({ kind: "node", id })}
              onSelectEdge={(id) => setSelection({ kind: "edge", id })}
              onClear={() => setSelection(null)}
            />
          )}
          <div className="pointer-events-none absolute left-3 top-3 space-y-1.5">
            <div className="panel pointer-events-auto flex items-center gap-3 px-2.5 py-1.5 text-[11px] text-soc-muted">
              {loading ? <Spinner /> : <span className="h-1.5 w-1.5 rounded-full bg-soc-ok" />}
              {data ? (
                <span>
                  {num(data.events)} events · {data.totals.active_hosts.toLocaleString()} hosts ·{" "}
                  {data.totals.active_users.toLocaleString()} users · <span className="text-soc-danger">{data.totals.alerts} alerts</span>
                  {data.truncated && <span className="text-soc-faint"> · showing top {data.nodes.length} nodes</span>}
                </span>
              ) : (
                "Loading graph…"
              )}
            </div>
            <Legend />
          </div>
          {error && <div className="absolute right-3 top-3 w-72"><ErrorNote error={error} /></div>}
        </div>

        <Card
          title={selection ? (selection.kind === "node" ? "Node activity" : "Edge events") : "Suspicious movement"}
          subtitle={selection ? selection.id.replaceAll("|", " · ") : "Alerted edges in this window"}
          bodyClass="p-0"
          className="min-h-0"
          actions={
            selection && (
              <button className="btn h-7" onClick={() => setSelection(null)}>Close</button>
            )
          }
        >
          <div className="h-full max-h-[calc(100vh-270px)] overflow-y-auto">
            {!selection ? (
              alertEdges.length ? (
                alertEdges.map((edge) => (
                  <button
                    key={edge.id}
                    onClick={() => setSelection({ kind: "edge", id: edge.id })}
                    className="block w-full border-b border-soc-line/70 px-4 py-2.5 text-left transition-colors hover:bg-soc-raised"
                  >
                    <div className="mono flex items-center justify-between text-[12px]">
                      <span>
                        {edge.source.split(":")[1]} <span className="text-soc-faint">→</span> {edge.target.split(":")[1]}
                      </span>
                      <span className="text-soc-danger">{edge.alerts} alert{edge.alerts > 1 ? "s" : ""}</span>
                    </div>
                    <div className="mt-0.5 text-[11px] text-soc-faint">
                      {edge.relation} · {edge.events} events · max score {edge.max_score?.toFixed(3)}
                      {edge.ground_truth ? <span className="text-soc-violet"> · {edge.ground_truth} redteam</span> : null}
                    </div>
                  </button>
                ))
              ) : (
                <Empty>No alerted edges in this window</Empty>
              )
            ) : detail ? (
              <DetailPanel
                detail={detail}
                threshold={threshold}
                onFocus={(id) => {
                  setFocus(id);
                  setFocusInput(id);
                }}
              />
            ) : (
              <Empty><Spinner /></Empty>
            )}
          </div>
        </Card>
      </div>
      <Provenance>
        Edges aggregate every event in the window. Movement self-loops (local logons) count toward node activity but are
        not drawn. The graph is a view of the replayed data; the served detector uses the 17 behavioural features, since
        graph embeddings did not improve detection (PROJECT_STATE §36).
      </Provenance>
    </div>
  );
}

function DetailPanel({
  detail,
  threshold,
  onFocus,
}: {
  detail: NodeDetail | EdgeDetail;
  threshold: number;
  onFocus: (id: string) => void;
}) {
  const isNode = "kind" in detail;
  return (
    <div className="space-y-3">
      <div className="space-y-3 px-4 pt-3">
        {isNode ? (
          <>
            <div className="flex items-center justify-between">
              <span className="mono text-[14px] text-soc-text">{detail.label}</span>
              <span className="text-[11px] uppercase text-soc-faint">{detail.kind}</span>
            </div>
            <KeyValue
              items={[
                ["Events in window", num(detail.events)],
                ["Alerts", <span key="a" className={detail.alerts ? "text-soc-danger" : ""}>{detail.alerts}</span>],
                ["Failed authentications", num(detail.failures)],
                ...(detail.kind === "host"
                  ? ([["Outbound / inbound", `${num(detail.outbound)} / ${num(detail.inbound)}`]] as [string, string][])
                  : []),
                ["Distinct peers", num(detail.distinct_peers)],
              ]}
            />
            {detail.top_peers.length > 0 && (
              <div>
                <div className="label mb-1.5">Top peers</div>
                <div className="flex flex-wrap gap-1.5">
                  {detail.top_peers.map((peer) => (
                    <span key={peer.name} className="mono rounded border border-soc-line px-1.5 py-0.5 text-[11px] text-soc-muted">
                      {peer.name} <span className="text-soc-faint">{peer.events}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
            <button className="btn btn-primary w-full justify-center" onClick={() => onFocus(detail.id)}>
              Focus graph on this {detail.kind}
            </button>
          </>
        ) : (
          <KeyValue
            items={[
              ["Relation", detail.relation],
              ["Events in window", num(detail.events)],
              ["Alerts", <span key="a" className={detail.alerts ? "text-soc-danger" : ""}>{detail.alerts}</span>],
            ]}
          />
        )}
      </div>
      <div className="label px-4">Recent events · click to investigate</div>
      <EventTable
        events={detail.recent.slice(0, 60)}
        threshold={threshold}
        dense
        onSelect={(event) => {
          window.location.href = `/investigation?id=${encodeURIComponent(event.event_id)}`;
        }}
      />
    </div>
  );
}

function Toggle({
  label,
  value,
  onChange,
  tone = "accent",
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  tone?: "accent" | "danger" | "violet";
}) {
  const on = { accent: "bg-soc-accent", danger: "bg-soc-danger", violet: "bg-soc-violet" }[tone];
  return (
    <button onClick={() => onChange(!value)} className="flex items-center gap-2 text-[12px] text-soc-muted transition-colors hover:text-soc-text">
      <span className={`relative h-4 w-7 rounded-full transition-colors ${value ? on : "bg-soc-line"}`}>
        <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${value ? "left-3.5" : "left-0.5"}`} />
      </span>
      {label}
    </button>
  );
}

function Legend() {
  return (
    <div className="panel pointer-events-auto flex flex-wrap items-center gap-3 px-2.5 py-1.5 text-[10.5px] text-soc-muted">
      <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-full border border-soc-accent bg-[#16324a]" />host</span>
      <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rotate-45 border border-soc-violet bg-[#2a2150]" />user</span>
      <span className="flex items-center gap-1.5"><i className="h-[2px] w-4 bg-[#2e3d50]" />movement</span>
      <span className="flex items-center gap-1.5"><i className="h-[2px] w-4 border-t border-dashed border-soc-violet" />identity</span>
      <span className="flex items-center gap-1.5"><i className="h-[2px] w-4 bg-soc-danger" />alerted</span>
    </div>
  );
}
