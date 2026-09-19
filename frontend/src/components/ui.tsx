"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import type { Severity, TriageStatus } from "@/lib/api";
import { SEVERITY_STYLE, STATUS_LABEL, STATUS_STYLE, score as fmtScore } from "@/lib/format";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
  bodyClass = "p-4",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={`panel flex min-w-0 flex-col ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-soc-line px-4 py-2.5">
          <div className="min-w-0">
            {title && <h2 className="truncate text-[13px] font-semibold text-soc-text">{title}</h2>}
            {subtitle && <p className="truncate text-[11.5px] text-soc-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClass}`}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "default" | "danger" | "ok" | "warn" | "accent";
}) {
  const color = {
    default: "text-soc-text",
    danger: "text-soc-danger",
    ok: "text-soc-ok",
    warn: "text-soc-warn",
    accent: "text-soc-accent",
  }[tone];
  return (
    <div className="panel px-4 py-3">
      <div className="label">{label}</div>
      <div className={`mt-1.5 text-[22px] font-semibold leading-none tabular-nums ${color}`}>{value}</div>
      {sub && <div className="mt-1.5 text-[11.5px] text-soc-muted">{sub}</div>}
    </div>
  );
}

export function Badge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-[1px] text-[10.5px] font-semibold uppercase tracking-wide ${className}`}
    >
      {children}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: Severity | null }) {
  if (!severity) return <span className="text-[11px] text-soc-faint">—</span>;
  return <Badge className={SEVERITY_STYLE[severity]}>{severity}</Badge>;
}

export function StatusBadge({ status }: { status: TriageStatus }) {
  return <Badge className={STATUS_STYLE[status]}>{STATUS_LABEL[status]}</Badge>;
}

export function GroundTruthBadge({ value }: { value: boolean }) {
  return value ? (
    <Badge className="border-soc-violet/50 bg-soc-violet/15 text-soc-violet" >Redteam</Badge>
  ) : (
    <Badge className="border-soc-line text-soc-faint">Benign</Badge>
  );
}

export function ScoreBar({ value, threshold }: { value: number; threshold: number }) {
  // Log scale so scores near the 0.015 threshold remain distinguishable.
  const toPos = (v: number) => Math.min(1, Math.max(0, (Math.log10(Math.max(v, 1e-5)) + 5) / 5));
  const above = value >= threshold;
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 w-20 overflow-hidden rounded-full bg-soc-line">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${above ? "bg-soc-danger" : "bg-soc-accent/60"}`}
          style={{ width: `${toPos(value) * 100}%` }}
        />
        <div className="absolute inset-y-0 w-px bg-soc-text/70" style={{ left: `${toPos(threshold) * 100}%` }} />
      </div>
      <span title={String(value)} className={`mono whitespace-nowrap tabular-nums ${above ? "text-soc-danger" : "text-soc-muted"}`}>{fmtScore(value)}</span>
    </div>
  );
}

export function Hosts({ source, destination }: { source: string; destination: string }) {
  return (
    <span className="mono whitespace-nowrap">
      <span className="text-soc-text">{source}</span>
      <span className="mx-1 text-soc-faint">→</span>
      <span className="text-soc-text">{destination}</span>
    </span>
  );
}

export function InvestigateLink({ eventId, children }: { eventId: string; children?: ReactNode }) {
  return (
    <Link
      href={`/investigation?id=${encodeURIComponent(eventId)}`}
      className="text-soc-accent underline-offset-2 transition-colors hover:text-sky-300 hover:underline"
    >
      {children ?? "Investigate"}
    </Link>
  );
}

export function Provenance({ children }: { children: ReactNode }) {
  return <p className="text-[11px] leading-relaxed text-soc-faint">{children}</p>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-24 items-center justify-center text-[12.5px] text-soc-faint">{children}</div>;
}

export function Spinner() {
  return <div className="h-4 w-4 animate-spin rounded-full border-2 border-soc-line border-t-soc-accent" />;
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="rounded border border-soc-danger/40 bg-soc-danger/10 px-3 py-2 text-[12px] text-soc-danger">
      {error}
    </div>
  );
}

export function KeyValue({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12px]">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-soc-muted">{k}</dt>
          <dd className="min-w-0 truncate text-right text-soc-text tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="inline-flex rounded border border-soc-line bg-soc-bg p-0.5">
      {options.map((option) => (
        <button
          key={String(option.value)}
          onClick={() => onChange(option.value)}
          className={`h-6 rounded-[3px] px-2.5 text-[11.5px] font-medium transition-colors ${
            option.value === value ? "bg-soc-raised text-soc-accent shadow-sm" : "text-soc-muted hover:text-soc-text"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function PageTitle({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: React.ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4">
      <div>
        <h1 className="text-[18px] font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-[12.5px] text-soc-muted">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}
