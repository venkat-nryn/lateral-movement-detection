"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type Experiments, type SplitResult } from "@/lib/api";
import { metric, num, pct } from "@/lib/format";
import { ConfusionMatrix, LineChart } from "@/components/charts";
import { Badge, Card, Empty, ErrorNote, Provenance, Segmented, Spinner, Stat } from "@/components/ui";
import { PageTitle } from "@/components/ui";

type SplitKey = "w1_validation" | "w1_test" | "w2";

export default function ExperimentsPage() {
  const [data, setData] = useState<Experiments | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState<SplitKey>("w1_test");

  useEffect(() => {
    api<Experiments>("/api/experiments").then(setData).catch((err) => setError((err as Error).message));
  }, []);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <Empty><Spinner /></Empty>;
  const current = data.computed.splits[split];

  return (
    <div className="space-y-4">
      <PageTitle
        title="Experiments"
        subtitle="The served detector recomputed from its real scores, alongside the recorded research comparison"
      />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge className="border-soc-ok/40 bg-soc-ok/10 text-soc-ok">Computed</Badge>
          <span className="text-[12.5px] text-soc-muted">{data.computed.model} · threshold <span className="mono">{data.computed.threshold.toFixed(6)}</span> (selected on W1 validation, frozen)</span>
        </div>
        <Segmented
          value={split}
          onChange={setSplit}
          options={[
            { value: "w1_validation", label: "W1 validation" },
            { value: "w1_test", label: "W1 test" },
            { value: "w2", label: "W2 (cross-window)" },
          ]}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="PR-AUC" value={metric(current.metrics.average_precision)} tone="accent" sub="primary metric" />
        <Stat label="ROC-AUC" value={metric(current.metrics.roc_auc, 6)} sub="inflated by imbalance" />
        <Stat label="Precision" value={pct(current.metrics.precision)} />
        <Stat label="Recall" value={pct(current.metrics.recall)} />
        <Stat label="F1" value={metric(current.metrics.f1, 3)} />
        <Stat label="False alerts / hour" value={num(current.metrics.fp_per_hour, 1)} tone="warn" sub={`${num(current.rows)} events · ${current.positives} redteam`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Confusion matrix" subtitle={`${current.title} at the frozen threshold`}>
          <ConfusionMatrix metrics={current.metrics} />
          <div className="mt-3">
            <Provenance>
              {current.positives} positives against {num(current.rows - current.positives)} negatives: one or two
              detections move precision and recall substantially (PROJECT_STATE §10).
            </Provenance>
          </div>
        </Card>
        <Card title="Precision–recall curve" subtitle="Exact corners, one per redteam event">
          <LineChart
            height={220}
            xLabel="Recall"
            yLabel="Precision"
            xDomain={[0, 1]}
            series={[
              {
                name: current.title,
                color: "#38bdf8",
                step: true,
                points: current.curves.pr.map((p) => ({ x: p.recall, y: p.precision })),
              },
            ]}
          />
        </Card>
        <Card title="ROC curve" subtitle="True vs false positive rate">
          <LineChart
            height={220}
            xLabel="False positive rate"
            yLabel="True positive rate"
            xDomain={[0, 1]}
            series={[
              { name: "chance", color: "#2a3341", dashed: true, points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
              { name: current.title, color: "#a78bfa", step: true, points: current.curves.roc.map((p) => ({ x: p.fpr, y: p.tpr })) },
            ]}
          />
        </Card>
      </div>

      <ThresholdCard split={current} threshold={data.computed.threshold} />

      <Card title="W1 vs W2" subtitle="Same frozen model and threshold; W2 is a later, disjoint window never used for fitting or tuning">
        <ComparisonTable splits={data.computed.splits} />
        <div className="mt-3">
          <Provenance>
            PR-AUC falls from W1 to W2 while ROC-AUC barely moves: with ~24,000 negatives per positive, ROC-AUC is not
            evidence of useful performance (RESEARCH_CONSTRAINTS §15). Both windows contain the same attacker host (PROJECT_STATE §35).
          </Provenance>
        </div>
      </Card>

      <RecordedResults data={data} />
    </div>
  );
}

function ThresholdCard({ split, threshold }: { split: SplitResult; threshold: number }) {
  const rows = split.sweep;
  const selected = rows.find((r) => r.selected);
  const maxFp = Math.max(1, ...rows.map((r) => r.fp_per_hour ?? 0));
  return (
    <Card
      title="Threshold visualisation"
      subtitle="How precision, recall, F1 and alert load change with the decision threshold (log scale)"
    >
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_1fr]">
        <LineChart
          height={250}
          logX
          xDomain={[1e-4, 0.999]}
          xLabel="Decision threshold"
          yLabel="Rate"
          marker={{ x: threshold, label: `frozen ${threshold.toFixed(4)}` }}
          series={[
            { name: "precision", color: "#38bdf8", points: rows.map((r) => ({ x: r.threshold, y: r.precision })) },
            { name: "recall", color: "#22c55e", points: rows.map((r) => ({ x: r.threshold, y: r.recall })) },
            { name: "F1", color: "#f59e0b", points: rows.map((r) => ({ x: r.threshold, y: r.f1 })) },
          ]}
          format={(v) => v.toFixed(2)}
        />
        <LineChart
          height={250}
          logX
          xDomain={[1e-4, 0.999]}
          yDomain={[0, maxFp]}
          xLabel="Decision threshold"
          yLabel="False alerts / hour"
          marker={{ x: threshold, label: "frozen" }}
          series={[{ name: "false alerts per hour", color: "#f43f5e", points: rows.map((r) => ({ x: r.threshold, y: r.fp_per_hour ?? 0 })) }]}
          format={(v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))}
        />
      </div>
      {selected && (
        <div className="mt-2 text-[12px] text-soc-muted">
          At the frozen threshold: {selected.predicted_positive} alerts, {selected.tp} true positives, {selected.fp} false
          positives, {num(selected.fp_per_hour, 1)} false alerts per hour.
        </div>
      )}
    </Card>
  );
}

function ComparisonTable({ splits }: { splits: Experiments["computed"]["splits"] }) {
  const entries = Object.entries(splits) as [SplitKey, SplitResult][];
  const cols: [string, (s: SplitResult) => React.ReactNode][] = [
    ["Events", (s) => num(s.rows)],
    ["Redteam", (s) => s.positives],
    ["PR-AUC", (s) => metric(s.metrics.average_precision)],
    ["ROC-AUC", (s) => metric(s.metrics.roc_auc, 6)],
    ["Precision", (s) => metric(s.metrics.precision)],
    ["Recall", (s) => metric(s.metrics.recall)],
    ["F1", (s) => metric(s.metrics.f1)],
    ["TP", (s) => s.metrics.tp],
    ["FP", (s) => s.metrics.fp],
    ["FN", (s) => s.metrics.fn],
    ["FP / h", (s) => num(s.metrics.fp_per_hour, 1)],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-soc-line text-left text-[10.5px] uppercase tracking-wider text-soc-faint">
            <th className="py-2 pr-4 font-medium">Split</th>
            {cols.map(([h]) => <th key={h} className="py-2 pr-4 text-right font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, s]) => (
            <tr key={key} className="table-row">
              <td className="py-2 pr-4 text-soc-text">{s.title}</td>
              {cols.map(([h, f]) => <td key={h} className="mono py-2 pr-4 text-right">{f(s)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RecordedResults({ data }: { data: Experiments }) {
  const recorded = data.recorded;
  const w1 = data.computed.splits.w1_test.metrics;
  const w2 = data.computed.splits.w2.metrics;
  const rangeMax = useMemo(() => Math.max(...recorded.ablation.rows.map((r) => r.w1_range[1])), [recorded]);

  return (
    <>
      <Card
        title={<span className="flex items-center gap-2">Model comparison on W1 test <Badge className="border-soc-warn/40 bg-soc-warn/10 text-soc-warn">Recorded</Badge></span>}
        subtitle={recorded.note}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-soc-line text-left text-[10.5px] uppercase tracking-wider text-soc-faint">
                {["Model", "Inputs", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "TP", "FP", "FN", "Source"].map((h, i) => (
                  <th key={h} className={`py-2 pr-4 font-medium ${i > 1 && i < 10 ? "text-right" : ""}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="table-row bg-soc-ok/5">
                <td className="py-2 pr-4 text-soc-text">XGBoost (standard) · served</td>
                <td className="mono py-2 pr-4">B</td>
                {[w1.average_precision, w1.roc_auc, w1.precision, w1.recall, w1.f1].map((v, i) => (
                  <td key={i} className="mono py-2 pr-4 text-right">{metric(v, i === 1 ? 6 : 4)}</td>
                ))}
                <td className="mono py-2 pr-4 text-right">{w1.tp}</td>
                <td className="mono py-2 pr-4 text-right">{w1.fp}</td>
                <td className="mono py-2 pr-4 text-right">{w1.fn}</td>
                <td className="py-2 pr-4"><Badge className="border-soc-ok/40 text-soc-ok">Computed</Badge></td>
              </tr>
              {recorded.w1_test_single_runs.map((run) => (
                <tr key={run.model} className="table-row">
                  <td className="py-2 pr-4 text-soc-text">
                    {run.model}
                    {!run.reproducible && <span className="ml-2 text-[10.5px] text-soc-warn" title="Not reproducible run to run (§34)">single draw</span>}
                  </td>
                  <td className="mono py-2 pr-4">{run.inputs}</td>
                  {[run.pr_auc, run.roc_auc, run.precision, run.recall, run.f1].map((v, i) => (
                    <td key={i} className="mono py-2 pr-4 text-right">{metric(v, i === 1 ? 6 : 4)}</td>
                  ))}
                  <td className="mono py-2 pr-4 text-right">{run.tp}</td>
                  <td className="mono py-2 pr-4 text-right">{run.fp}</td>
                  <td className="mono py-2 pr-4 text-right">{run.fn}</td>
                  <td className="py-2 pr-4 text-soc-faint">§{run.section}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-soc-faint">
          {Object.entries(recorded.input_legend).map(([k, v]) => (
            <span key={k}><span className="mono text-soc-muted">{k}</span> = {v}</span>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_380px]">
        <Card
          title={<span className="flex items-center gap-2">Information-source ablation <Badge className="border-soc-warn/40 bg-soc-warn/10 text-soc-warn">Recorded §{recorded.ablation.section}</Badge></span>}
          subtitle={`${recorded.ablation.seeds} seeds per variant: mean with min–max range. W2 is the corrected full window.`}
        >
          <div className="space-y-2.5">
            <div className="grid grid-cols-[210px_1fr_1fr_70px_70px] gap-3 text-[10.5px] uppercase tracking-wider text-soc-faint">
              <span>Model</span><span>W1 test PR-AUC</span><span>W2 PR-AUC</span><span className="text-right">W2 recall</span><span className="text-right">W2 FP/h</span>
            </div>
            <AblationRow label="XGBoost (served)" inputs="B" w1={[w1.average_precision ?? 0, w1.average_precision ?? 0, w1.average_precision ?? 0]} w2={[w2.average_precision ?? 0, w2.average_precision ?? 0, w2.average_precision ?? 0]} recall={w2.recall} fp={w2.fp_per_hour ?? 0} max={rangeMax} computed />
            {recorded.ablation.rows.map((row) => (
              <AblationRow
                key={row.model}
                label={row.model}
                inputs={row.inputs}
                w1={[row.w1_pr_auc, ...row.w1_range]}
                w2={[row.w2_pr_auc, ...row.w2_range]}
                recall={row.w2_recall}
                fp={row.w2_fp_per_hour}
                max={rangeMax}
              />
            ))}
          </div>
        </Card>

        <Card
          title={<span className="flex items-center gap-2">Marginal value <Badge className="border-soc-warn/40 bg-soc-warn/10 text-soc-warn">§{recorded.marginal_contributions.section}</Badge></span>}
          subtitle="PR-AUC gain from adding each source (difference of seed means)"
        >
          <div className="space-y-2">
            {recorded.marginal_contributions.rows.map((row) => (
              <div key={row.contribution} className="text-[12px]">
                <div className="text-soc-muted">{row.contribution}</div>
                <div className="mt-1 grid grid-cols-2 gap-2">
                  {(["w1", "w2"] as const).map((k) => (
                    <DeltaBar key={k} label={k.toUpperCase()} value={row[k]} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3">
            <Provenance>
              Only the behavioural features carry large, consistent value. Graph embeddings add nothing distinguishable
              from seed noise once they are present (PROJECT_STATE §36.2).
            </Provenance>
          </div>
        </Card>
      </div>
    </>
  );
}

function AblationRow({
  label,
  inputs,
  w1,
  w2,
  recall,
  fp,
  max,
  computed = false,
}: {
  label: string;
  inputs: string;
  w1: [number, number, number];
  w2: [number, number, number];
  recall: number;
  fp: number;
  max: number;
  computed?: boolean;
}) {
  return (
    <div className="grid grid-cols-[210px_1fr_1fr_70px_70px] items-center gap-3 text-[12px]">
      <span className={computed ? "text-soc-ok" : "text-soc-text"}>
        {label} <span className="mono text-[10.5px] text-soc-faint">{inputs}</span>
      </span>
      <RangeBar values={w1} max={max} color="#38bdf8" />
      <RangeBar values={w2} max={max} color="#f43f5e" />
      <span className="mono text-right">{recall.toFixed(3)}</span>
      <span className="mono text-right">{num(fp, 0)}</span>
    </div>
  );
}

function RangeBar({ values, max, color }: { values: [number, number, number]; max: number; color: string }) {
  const [mean, low, high] = values;
  const f = (v: number) => `${(v / max) * 100}%`;
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 flex-1 rounded-full bg-soc-line">
        <div className="absolute inset-y-0 rounded-full opacity-35" style={{ left: f(low), width: `calc(${f(high)} - ${f(low)})`, background: color }} />
        <div className="absolute -top-0.5 h-3 w-[3px] rounded-sm" style={{ left: f(mean), background: color }} />
      </div>
      <span className="mono w-12 text-right text-soc-text">{mean.toFixed(3)}</span>
    </div>
  );
}

function DeltaBar({ label, value }: { label: string; value: number }) {
  const width = Math.min(50, Math.abs(value) * 55);
  return (
    <div className="flex items-center gap-2">
      <span className="w-6 text-[10.5px] text-soc-faint">{label}</span>
      <div className="relative h-2 flex-1">
        <div className="absolute inset-y-0 left-1/2 w-px bg-soc-lineStrong" />
        <div
          className={`absolute inset-y-0 rounded-sm ${value >= 0 ? "bg-soc-ok/70" : "bg-soc-danger/70"}`}
          style={value >= 0 ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
        />
      </div>
      <span className={`mono w-12 text-right ${value >= 0 ? "text-soc-ok" : "text-soc-danger"}`}>
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );
}
