"use client";

import { useEffect, useState } from "react";
import { api, type Overview } from "@/lib/api";
import { useReplay } from "@/lib/replay";
import { DEFAULT_THRESHOLD, useStatus } from "@/lib/status";
import { lanlTime, metric, num, pct } from "@/lib/format";
import { ActivityChart } from "@/components/charts";
import { AlertListItem, EventTable } from "@/components/EventTable";
import { Card, Empty, KeyValue, PageTitle, Provenance, Stat } from "@/components/ui";

export default function OverviewPage() {
  const { counters, feed, alerts, replay, tickCount, connection } = useReplay();
  const status = useStatus();
  const [overview, setOverview] = useState<Overview | null>(null);
  const threshold = status?.model.threshold ?? DEFAULT_THRESHOLD;

  // Refresh the per-minute activity and reference metrics every ~3 s.
  useEffect(() => {
    if (tickCount % 6 !== 1 && overview) return;
    api<Overview>("/api/overview").then(setOverview).catch(() => undefined);
  }, [tickCount, overview]);

  const evaluation = counters?.evaluation;
  const totalBins = replay ? Math.ceil((replay.end - replay.start) / 60) : 120;

  return (
    <div className="space-y-4">
      <PageTitle
        title="Overview"
        subtitle="Real LANL W2 authentication events replayed through the frozen W1 detector"
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <Stat label="Events replayed" value={num(counters?.events)} sub={status ? `of ${num(status.export.rows)} in W2` : undefined} />
        <Stat label="Alerts raised" value={num(counters?.alerts)} tone="danger" sub={`score ≥ ${threshold.toFixed(5)}`} />
        <Stat
          label="Redteam detected"
          value={evaluation ? `${evaluation.true_positives}/${evaluation.positives_seen}` : "—"}
          tone="ok"
          sub={`recall ${pct(evaluation?.recall)}`}
        />
        <Stat label="False positives" value={num(evaluation?.false_positives)} tone="warn" sub={`precision ${pct(evaluation?.precision)}`} />
        <Stat label="Missed redteam" value={num(evaluation?.missed)} sub="ground truth, not alerted" />
        <Stat
          label="Replay clock"
          value={<span className="mono text-[18px]">{replay ? lanlTime(replay.position, false) : "—"}</span>}
          tone="accent"
          sub={replay ? `${pct(replay.progress, 0)} of W2 · ${replay.speed}×` : connection}
        />
      </div>
      <Provenance>
        Detection counts are live over the replayed prefix. Redteam, recall, precision and missed use the exact
        4-field redteam match (research ground truth), which a production SOC would not have.
      </Provenance>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card
          className="xl:col-span-2"
          title="Authentication activity"
          subtitle="Per-minute event volume across the replayed part of W2"
        >
          {overview && overview.activity.length ? (
            <ActivityChart bins={overview.activity} totalBins={totalBins} />
          ) : (
            <Empty>Waiting for replay data…</Empty>
          )}
        </Card>

        <Card title="System & model status">
          {status ? (
            <div className="space-y-3">
              <KeyValue
                items={[
                  ["Mode", <span key="m" className="text-soc-ok">Replay · real LANL W2</span>],
                  ["Stream", connection === "live" ? <span key="s" className="text-soc-ok">Connected</span> : <span key="s" className="text-soc-danger">{connection}</span>],
                  ["Detector", status.model.name],
                  ["Threshold", <span key="t" className="mono">{status.model.threshold.toFixed(6)}</span>],
                  ["Trained on", `W1 TRAIN · ${num(status.model.train_rows)} events (${status.model.train_positives}+)`],
                  ["Features", `${status.model.features} causal behavioural`],
                  ["Model digest", <span key="d" className="mono">{status.model.digest.slice(0, 16)}…</span>],
                  [
                    "Integrity",
                    status.model.integrity.scores_identical ? (
                      <span key="i" className="text-soc-ok">{num(status.model.integrity.rows_checked)} rows re-scored ✓</span>
                    ) : (
                      <span key="i" className="text-soc-danger">FAILED</span>
                    ),
                  ],
                  ["Export built", new Date(status.export.created_at).toLocaleString()],
                ]}
              />
            </div>
          ) : (
            <Empty>Loading status…</Empty>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card
          className="xl:col-span-2"
          title={
            <span className="flex items-center gap-2">
              Live event stream
              <span className={`h-1.5 w-1.5 rounded-full ${replay?.playing ? "animate-pulseDot bg-soc-ok" : "bg-soc-warn"}`} />
            </span>
          }
          subtitle="Most recent replayed events, sampled per tick; every event is scored by the frozen model"
          bodyClass="p-0"
        >
          <div className="h-[420px] overflow-y-auto">
            {feed.length ? <EventTable events={feed} threshold={threshold} flashKey dense /> : <Empty>Waiting for events…</Empty>}
          </div>
        </Card>

        <Card title="Recent alerts" subtitle="Newest first · click to investigate" bodyClass="p-0">
          <div className="h-[420px] overflow-y-auto">
            {alerts.length ? (
              alerts.slice(0, 40).map((alert) => (
                <div key={alert.event_id} className="animate-fadeIn">
                  <AlertListItem alert={alert} />
                </div>
              ))
            ) : (
              <Empty>No alerts yet in the replayed prefix</Empty>
            )}
          </div>
        </Card>
      </div>

      {overview && (
        <Card title="Detection metrics over complete splits" subtitle={overview.reference_metrics.note}>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-soc-line text-left text-[10.5px] uppercase tracking-wider text-soc-faint">
                  <th className="py-2 pr-4 font-medium">Split</th>
                  {["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"].map((h) => (
                    <th key={h} className="py-2 pr-4 text-right font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["W1 test (in-window)", overview.reference_metrics.w1_test],
                    ["W2 full (cross-window)", overview.reference_metrics.w2_full],
                  ] as const
                ).map(([name, m]) => (
                  <tr key={name} className="table-row">
                    <td className="py-2 pr-4 text-soc-text">{name}</td>
                    <td className="mono py-2 pr-4 text-right">{metric(m.average_precision)}</td>
                    <td className="mono py-2 pr-4 text-right">{metric(m.roc_auc, 6)}</td>
                    <td className="mono py-2 pr-4 text-right">{metric(m.precision)}</td>
                    <td className="mono py-2 pr-4 text-right">{metric(m.recall)}</td>
                    <td className="mono py-2 pr-4 text-right">{metric(m.f1)}</td>
                    <td className="mono py-2 pr-4 text-right text-soc-ok">{m.tp}</td>
                    <td className="mono py-2 pr-4 text-right text-soc-warn">{m.fp}</td>
                    <td className="mono py-2 pr-4 text-right text-soc-danger">{m.fn}</td>
                    <td className="mono py-2 pr-4 text-right text-soc-muted">{num(m.tn)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
