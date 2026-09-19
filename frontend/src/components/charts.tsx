"use client";

import { useMemo, useState } from "react";
import type { BinaryMetrics } from "@/lib/api";
import { lanlTime, num, pct } from "@/lib/format";

// ---------------------------------------------------------------------------
// Activity: events per minute with alert and ground-truth markers.
// ---------------------------------------------------------------------------
export function ActivityChart({
  bins,
  totalBins,
  height = 132,
}: {
  bins: { t: number; events: number; alerts: number; positives: number }[];
  totalBins: number;
  height?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const maxEvents = Math.max(1, ...bins.map((b) => b.events));
  const maxAlerts = Math.max(1, ...bins.map((b) => b.alerts));
  const width = 1000;
  const slot = width / Math.max(totalBins, 1);
  const chartH = height - 18;
  const active = hover !== null ? bins[hover] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-[132px] w-full" onMouseLeave={() => setHover(null)}>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} x2={width} y1={chartH * f} y2={chartH * f} stroke="#1e2530" strokeDasharray="2 4" />
        ))}
        {bins.map((b, i) => {
          const h = (b.events / maxEvents) * (chartH - 26);
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={i * slot} y={0} width={slot} height={chartH} fill="transparent" />
              <rect
                x={i * slot + slot * 0.12}
                y={chartH - h}
                width={slot * 0.76}
                height={h}
                fill={hover === i ? "#38bdf8" : "#1f4b63"}
                className="transition-colors"
              />
              {b.alerts > 0 && (
                <rect
                  x={i * slot + slot * 0.12}
                  y={4 + (1 - b.alerts / maxAlerts) * 14}
                  width={slot * 0.76}
                  height={Math.max(2, (b.alerts / maxAlerts) * 14)}
                  fill="#f43f5e"
                  opacity={0.9}
                />
              )}
              {b.positives > 0 && <circle cx={i * slot + slot / 2} cy={chartH + 8} r={Math.min(3.5, slot / 3)} fill="#a78bfa" />}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex items-center gap-4 text-[10.5px] text-soc-faint">
        <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-sm bg-[#1f4b63]" />events / min</span>
        <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-sm bg-soc-danger" />alerts / min</span>
        <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-soc-violet" />redteam events (ground truth)</span>
        {active && (
          <span className="ml-auto mono text-soc-muted">
            {lanlTime(active.t)} · {num(active.events)} events · {active.alerts} alerts · {active.positives} redteam
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic line chart (PR, ROC, threshold sweeps).
// ---------------------------------------------------------------------------
export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  step?: boolean;
  dashed?: boolean;
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  logX = false,
  xDomain,
  yDomain = [0, 1],
  marker,
  height = 240,
  format = (v: number) => v.toFixed(2),
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  logX?: boolean;
  xDomain?: [number, number];
  yDomain?: [number, number];
  marker?: { x: number; label: string };
  height?: number;
  format?: (v: number) => string;
}) {
  const width = 520;
  const pad = { l: 40, r: 12, t: 10, b: 30 };
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p.x);
  const [x0, x1] = xDomain ?? [Math.min(...xs), Math.max(...xs)];
  const tx = (x: number) => {
    const f = logX ? (Math.log10(x) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : (x - x0) / (x1 - x0 || 1);
    return pad.l + f * (width - pad.l - pad.r);
  };
  const ty = (y: number) => pad.t + (1 - (y - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * (height - pad.t - pad.b);

  const path = (s: Series) =>
    s.points
      .map((p, i) => {
        const x = tx(p.x);
        const y = ty(p.y);
        if (i === 0) return `M${x},${y}`;
        if (s.step) {
          const prev = s.points[i - 1];
          return `L${x},${ty(prev.y)} L${x},${y}`;
        }
        return `L${x},${y}`;
      })
      .join(" ");

  const xticks = logX
    ? [-4, -3, -2, -1, 0].map((e) => 10 ** e).filter((v) => v >= x0 * 0.999 && v <= x1 * 1.001)
    : [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  const yticks = [0, 0.25, 0.5, 0.75, 1].map((f) => yDomain[0] + f * (yDomain[1] - yDomain[0]));

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
        {yticks.map((v) => (
          <g key={`y${v}`}>
            <line x1={pad.l} x2={width - pad.r} y1={ty(v)} y2={ty(v)} stroke="#1e2530" />
            <text x={pad.l - 6} y={ty(v) + 3} textAnchor="end" className="fill-soc-faint text-[9.5px]">{format(v)}</text>
          </g>
        ))}
        {xticks.map((v) => (
          <text key={`x${v}`} x={tx(v)} y={height - pad.b + 14} textAnchor="middle" className="fill-soc-faint text-[9.5px]">
            {logX ? (v >= 1 ? "1" : `1e${Math.round(Math.log10(v))}`) : format(v)}
          </text>
        ))}
        <text x={(width + pad.l) / 2} y={height - 3} textAnchor="middle" className="fill-soc-muted text-[10px]">{xLabel}</text>
        <text x={10} y={(height - pad.b) / 2} textAnchor="middle" transform={`rotate(-90 10 ${(height - pad.b) / 2})`} className="fill-soc-muted text-[10px]">{yLabel}</text>
        {marker && (
          <g>
            <line x1={tx(marker.x)} x2={tx(marker.x)} y1={pad.t} y2={height - pad.b} stroke="#f59e0b" strokeDasharray="3 3" />
            <text x={tx(marker.x) + 4} y={pad.t + 10} className="fill-soc-warn text-[9.5px]">{marker.label}</text>
          </g>
        )}
        {series.map((s) => (
          <path key={s.name} d={path(s)} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dashed ? "4 3" : undefined} />
        ))}
      </svg>
      <div className="mt-1 flex flex-wrap gap-3 text-[10.5px] text-soc-muted">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5">
            <i className="h-[2px] w-3" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Confusion matrix at the frozen threshold.
// ---------------------------------------------------------------------------
export function ConfusionMatrix({ metrics }: { metrics: BinaryMetrics }) {
  const cells = [
    { label: "TP", value: metrics.tp, hint: "redteam, alerted", tone: "border-soc-ok/40 bg-soc-ok/10 text-soc-ok" },
    { label: "FN", value: metrics.fn, hint: "redteam, missed", tone: "border-soc-danger/40 bg-soc-danger/10 text-soc-danger" },
    { label: "FP", value: metrics.fp, hint: "benign, alerted", tone: "border-soc-warn/40 bg-soc-warn/10 text-soc-warn" },
    { label: "TN", value: metrics.tn, hint: "benign, not alerted", tone: "border-soc-line bg-soc-raised text-soc-muted" },
  ];
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-[11px]">
      <div />
      <div className="text-center text-soc-faint">Alerted</div>
      <div className="text-center text-soc-faint">Not alerted</div>
      <div className="flex items-center pr-1 text-soc-faint">Redteam</div>
      {[cells[0], cells[1]].map((c) => <Cell key={c.label} {...c} />)}
      <div className="flex items-center pr-1 text-soc-faint">Benign</div>
      {[cells[2], cells[3]].map((c) => <Cell key={c.label} {...c} />)}
    </div>
  );
}

function Cell({ label, value, hint, tone }: { label: string; value: number; hint: string; tone: string }) {
  return (
    <div className={`rounded border px-3 py-2 ${tone}`}>
      <div className="flex items-baseline justify-between">
        <span className="text-[10.5px] font-semibold">{label}</span>
        <span className="text-[17px] font-semibold tabular-nums">{num(value)}</span>
      </div>
      <div className="text-[10px] opacity-70">{hint}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TreeSHAP diverging bars (log-odds contributions).
// ---------------------------------------------------------------------------
export function ShapBars({
  contributions,
  bias,
  margin,
  limit = 17,
}: {
  contributions: { name: string; value: number; contribution: number }[];
  bias: number;
  margin: number;
  limit?: number;
}) {
  const rows = contributions.slice(0, limit);
  const max = useMemo(() => Math.max(0.01, ...rows.map((c) => Math.abs(c.contribution))), [rows]);
  return (
    <div className="space-y-1">
      {rows.map((c) => {
        const w = (Math.abs(c.contribution) / max) * 50;
        const positive = c.contribution >= 0;
        return (
          <div key={c.name} className="grid grid-cols-[minmax(0,210px)_1fr_64px] items-center gap-2 text-[11.5px]">
            <div className="truncate text-soc-muted" title={c.name}>
              {c.name}
              <span className="ml-1.5 mono text-soc-faint">= {Number.isInteger(c.value) ? c.value : c.value.toFixed(2)}</span>
            </div>
            <div className="relative h-3.5">
              <div className="absolute inset-y-0 left-1/2 w-px bg-soc-lineStrong" />
              <div
                className={`absolute inset-y-0.5 rounded-sm transition-all duration-500 ${positive ? "bg-soc-danger/80" : "bg-soc-accent/70"}`}
                style={positive ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
              />
            </div>
            <div className={`mono text-right tabular-nums ${positive ? "text-soc-danger" : "text-soc-accent"}`}>
              {c.contribution >= 0 ? "+" : ""}
              {c.contribution.toFixed(3)}
            </div>
          </div>
        );
      })}
      <div className="mt-2 flex justify-between border-t border-soc-line pt-2 text-[11px] text-soc-muted">
        <span>
          bias <span className="mono text-soc-text">{bias.toFixed(3)}</span> + Σ contributions ={" "}
          <span className="mono text-soc-text">{margin.toFixed(3)}</span> log-odds
        </span>
        <span>
          sigmoid → <span className="mono text-soc-text">{pct(1 / (1 + Math.exp(-margin)), 2)}</span>
        </span>
      </div>
    </div>
  );
}
