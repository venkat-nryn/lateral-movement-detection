"use client";

import { useEffect, useState } from "react";
import { api, type Pipeline } from "@/lib/api";
import { useReplay } from "@/lib/replay";
import { num } from "@/lib/format";
import { Card, Empty, KeyValue, Provenance, Spinner } from "@/components/ui";
import { PageTitle } from "@/components/ui";

const MODULES: Record<string, { modules: string[]; explain: string }> = {
  lanl: {
    modules: ["data/raw/lanl/auth.txt.gz", "data/raw/lanl/redteam.txt.gz"],
    explain: "The original LANL Comprehensive Multi-Source Cyber-Security Events data. Read-only; never modified.",
  },
  ingestion: {
    modules: ["ml/preprocessing/lanl_adapter.py", "ml/preprocessing/ml_window.py"],
    explain: "Streams the 7.6 GB gzip without decompressing it to disk, and selects bounded windows (W1, W2) by a fixed rule on the redteam file alone. A window build fails loudly rather than truncating.",
  },
  canonical: {
    modules: ["ml/preprocessing/schema.py"],
    explain: "Each record becomes a validated CanonicalEvent with a deterministic event id. Malformed input is rejected, not repaired.",
  },
  features: {
    modules: ["ml/preprocessing/features.py"],
    explain: "17 causal temporal/behavioural features. An event at time T only sees events strictly before T; same-timestamp events cannot see each other.",
  },
  graph: {
    modules: ["ml/graph/temporal_graph.py", "ml/models/graph_data.py"],
    explain: "User and host nodes; every event adds a user→destination identity edge and a source→destination movement edge.",
  },
  model: {
    modules: ["ml/baselines/xgboost_baseline.py", "ml/evaluation/cross_window.py"],
    explain: "Standard XGBoost fitted on W1 TRAIN, threshold chosen on W1 VALIDATION, then frozen with a SHA-256 digest. The dashboard re-verifies its scores at start-up.",
  },
  explain: {
    modules: ["ml/evaluation/explainability.py"],
    explain: "Exact TreeSHAP: per-feature contributions that sum, with the bias, to the model's log-odds output.",
  },
  alert: {
    modules: ["backend/services/store.py"],
    explain: "An event scoring at or above the frozen threshold becomes an alert, surfaced when the replay clock reaches it.",
  },
};

const STAT_LABELS: Record<string, string> = {
  auth_bytes: "auth.txt.gz",
  redteam_records: "Redteam records",
  w2_context_events: "W2 context events",
  w2_emitted_events: "W2 emitted events",
  replayed: "Replayed so far",
  users: "Users in W2",
  hosts: "Hosts in W2",
  features: "Features",
  node_types: "Node types",
  edges_per_event: "Edges per event",
  threshold: "Threshold",
  integrity_rows: "Rows re-verified",
  alerts_so_far: "Alerts so far",
  alerts_total: "Alerts in W2",
};

function formatStat(key: string, value: number) {
  if (key === "auth_bytes") return `${(value / 1e9).toFixed(2)} GB`;
  if (key === "threshold") return value.toFixed(6);
  return num(value);
}

export default function PipelinePage() {
  const { replay, tickCount } = useReplay();
  const [data, setData] = useState<Pipeline | null>(null);
  const [active, setActive] = useState("model");

  useEffect(() => {
    if (tickCount % 4 !== 0 && data) return;
    api<Pipeline>("/api/pipeline").then(setData).catch(() => undefined);
  }, [tickCount, data]);

  if (!data) return <Empty><Spinner /></Empty>;
  const flowing = Boolean(replay?.playing);
  const selected = data.stages.find((s) => s.key === active) ?? data.stages[0];

  return (
    <div className="space-y-4">
      <PageTitle title="Pipeline" subtitle="How a real LANL record becomes an analyst alert, with live counts from the running system" />

      <div className="panel overflow-x-auto p-5">
        <div className="flex min-w-[1100px] items-stretch">
          {data.stages.map((stage, i) => (
            <div key={stage.key} className="flex flex-1 items-center">
              <button
                onClick={() => setActive(stage.key)}
                className={`group relative flex h-full w-full flex-col rounded-md border p-3 text-left transition-all duration-200 hover:-translate-y-0.5 ${
                  active === stage.key
                    ? "border-soc-accent/70 bg-soc-accent/10 shadow-[0_0_0_1px_rgba(56,189,248,0.2)]"
                    : stage.key === "alert"
                      ? "border-soc-danger/40 bg-soc-danger/5 hover:border-soc-danger/70"
                      : "border-soc-line bg-soc-raised hover:border-soc-lineStrong"
                }`}
              >
                <span className="mono text-[10px] text-soc-faint">{String(i + 1).padStart(2, "0")}</span>
                <span className={`mt-0.5 text-[13px] font-semibold ${stage.key === "alert" ? "text-soc-danger" : "text-soc-text"}`}>{stage.title}</span>
                <span className="mt-2 space-y-0.5">
                  {Object.entries(stage.stats).slice(0, 2).map(([key, value]) => (
                    <span key={key} className="block text-[11px] text-soc-muted">
                      <span className="mono text-soc-text tabular-nums">{formatStat(key, value)}</span>{" "}
                      <span className="text-soc-faint">{STAT_LABELS[key] ?? key}</span>
                    </span>
                  ))}
                </span>
              </button>
              {i < data.stages.length - 1 && <Connector flowing={flowing} />}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_360px]">
        <Card title={selected.title} subtitle={selected.detail}>
          <p className="text-[12.5px] leading-relaxed text-soc-muted">{MODULES[selected.key]?.explain}</p>
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <div className="label mb-2">Live statistics</div>
              {Object.keys(selected.stats).length ? (
                <KeyValue items={Object.entries(selected.stats).map(([k, v]) => [STAT_LABELS[k] ?? k, <span key={k} className="mono">{formatStat(k, v)}</span>])} />
              ) : (
                <span className="text-[12px] text-soc-faint">Computed on demand per event.</span>
              )}
            </div>
            <div>
              <div className="label mb-2">Implemented in</div>
              <ul className="space-y-1">
                {MODULES[selected.key]?.modules.map((m) => (
                  <li key={m} className="mono text-[11.5px] text-soc-accent">{m}</li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
        <Card title="Demo Lab pipeline" subtitle="Isolated synthetic runs through the same components">
          <KeyValue
            items={[
              ["Lab events", num(data.lab.events)],
              ["Lab alerts", num(data.lab.alerts)],
            ]}
          />
          <div className="mt-3">
            <Provenance>
              Lab events are synthetic and isolated. They never enter the replay, the alert queue or any metric
              (RESEARCH_CONSTRAINTS §3).
            </Provenance>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Connector({ flowing }: { flowing: boolean }) {
  return (
    <div className="relative mx-1 h-[2px] w-8 shrink-0 overflow-hidden bg-soc-lineStrong">
      {flowing && (
        <span className="absolute inset-y-0 w-3 animate-[flow_1.1s_linear_infinite] rounded-full bg-soc-accent shadow-[0_0_6px_#38bdf8]" />
      )}
      <style>{`@keyframes flow { from { left: -12px } to { left: 32px } }`}</style>
    </div>
  );
}
