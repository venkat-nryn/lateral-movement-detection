"use client";

import { useEffect, useMemo, useState } from "react";
import { api, patch, type AlertDetail, type AlertList, type AlertRecord, type Severity, type TriageStatus } from "@/lib/api";
import { useReplay } from "@/lib/replay";
import { DEFAULT_THRESHOLD, useStatus } from "@/lib/status";
import { FEATURE_HELP, STATUS_LABEL, lanlTime, num } from "@/lib/format";
import { EventTable } from "@/components/EventTable";
import { Card, Empty, ErrorNote, GroundTruthBadge, InvestigateLink, KeyValue, Provenance, SeverityBadge, Spinner } from "@/components/ui";
import { PageTitle } from "@/components/ui";

const SEVERITIES: Severity[] = ["critical", "high", "medium", "low"];
const PAGE = 100;

interface Filters {
  q: string;
  user: string;
  source: string;
  destination: string;
  severity: string;
  status: string;
  ground_truth: string;
  from: string;
  to: string;
}

const EMPTY: Filters = { q: "", user: "", source: "", destination: "", severity: "", status: "", ground_truth: "", from: "", to: "" };

export default function AlertsPage() {
  const { alertVersion, replay } = useReplay();
  const status = useStatus();
  const threshold = status?.model.threshold ?? DEFAULT_THRESHOLD;
  const [filters, setFilters] = useState<Filters>(EMPTY);
  const [debounced, setDebounced] = useState<Filters>(EMPTY);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<AlertList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<AlertRecord | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(filters), 250);
    return () => clearTimeout(timer);
  }, [filters]);
  useEffect(() => setPage(0), [debounced]);

  useEffect(() => {
    const params = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE) });
    for (const [key, value] of Object.entries(debounced)) if (value) params.set(key, value);
    api<AlertList>(`/api/alerts?${params}`)
      .then((result) => {
        setData(result);
        setError(null);
      })
      .catch((err) => setError((err as Error).message));
  }, [debounced, page, alertVersion]);

  const set = (key: keyof Filters) => (value: string) => setFilters((f) => ({ ...f, [key]: value }));
  const active = useMemo(() => Object.values(filters).some(Boolean), [filters]);

  const quickRange = (minutes: number) => {
    if (!replay) return;
    setFilters((f) => ({ ...f, from: String(Math.floor(replay.position - minutes * 60)), to: "" }));
  };

  return (
    <div className="space-y-4">
      <PageTitle
        title="Alerts"
        subtitle="Every replayed event scored at or above the frozen validation threshold"
        actions={
          data && (
            <div className="text-right text-[12px] text-soc-muted">
              <span className="text-[18px] font-semibold text-soc-text">{num(data.total)}</span> matching ·{" "}
              {num(data.revealed)} raised so far
            </div>
          )
        }
      />

      <div className="panel space-y-2.5 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <input className="input w-64" placeholder="Search id, user, host…" value={filters.q} onChange={(e) => set("q")(e.target.value)} />
          <input className="input w-40" placeholder="User" value={filters.user} onChange={(e) => set("user")(e.target.value)} />
          <input className="input w-32" placeholder="Source host" value={filters.source} onChange={(e) => set("source")(e.target.value)} />
          <input className="input w-32" placeholder="Destination" value={filters.destination} onChange={(e) => set("destination")(e.target.value)} />
          <select className="input" value={filters.status} onChange={(e) => set("status")(e.target.value)}>
            <option value="">Any status</option>
            {(Object.keys(STATUS_LABEL) as TriageStatus[]).map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
          </select>
          <select className="input" value={filters.ground_truth} onChange={(e) => set("ground_truth")(e.target.value)}>
            <option value="">Any ground truth</option>
            <option value="true">Redteam only</option>
            <option value="false">Benign only</option>
          </select>
          {active && <button className="btn" onClick={() => setFilters(EMPTY)}>Reset filters</button>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="label">Severity</span>
          {SEVERITIES.map((severity) => (
            <button
              key={severity}
              onClick={() => set("severity")(filters.severity === severity ? "" : severity)}
              className={`rounded border px-2 py-0.5 text-[11.5px] transition-all ${
                filters.severity === severity ? "border-soc-accent bg-soc-accent/10" : "border-soc-line hover:border-soc-lineStrong"
              }`}
            >
              <SeverityBadge severity={severity} /> <span className="ml-1 tabular-nums text-soc-muted">{data?.severity_counts[severity] ?? 0}</span>
            </button>
          ))}
          <span className="label ml-3">Time</span>
          <input className="input w-28" placeholder="from (s)" value={filters.from} onChange={(e) => set("from")(e.target.value)} />
          <input className="input w-28" placeholder="to (s)" value={filters.to} onChange={(e) => set("to")(e.target.value)} />
          {[5, 15, 60].map((m) => (
            <button key={m} className="btn h-7" onClick={() => quickRange(m)}>Last {m}m</button>
          ))}
          {filters.from && <span className="mono text-[11px] text-soc-faint">from {lanlTime(Number(filters.from))}</span>}
        </div>
      </div>
      <ErrorNote error={error} />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_400px]">
        <Card bodyClass="p-0">
          <div className="max-h-[calc(100vh-330px)] overflow-y-auto">
            {data ? (
              data.items.length ? (
                <EventTable
                  events={data.items}
                  threshold={threshold}
                  showStatus
                  selectedId={selected?.event_id}
                  onSelect={(event) => setSelected(event as AlertRecord)}
                />
              ) : (
                <Empty>No alerts match these filters</Empty>
              )
            ) : (
              <Empty><Spinner /></Empty>
            )}
          </div>
          {data && data.total > PAGE && (
            <div className="flex items-center justify-between border-t border-soc-line px-3 py-2 text-[12px] text-soc-muted">
              <span>
                {page * PAGE + 1}–{Math.min((page + 1) * PAGE, data.total)} of {num(data.total)}
              </span>
              <div className="flex gap-2">
                <button className="btn h-7" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                <button className="btn h-7" disabled={(page + 1) * PAGE >= data.total} onClick={() => setPage(page + 1)}>Next</button>
              </div>
            </div>
          )}
        </Card>

        <div className="xl:sticky xl:top-0 xl:self-start">
          {selected ? (
            <AlertDrawer key={selected.event_id} alert={selected} onUpdated={(updated) => setSelected(updated)} />
          ) : (
            <Card title="Alert details"><Empty>Select an alert to see its details and triage it.</Empty></Card>
          )}
        </div>
      </div>
      <Provenance>
        Severity is a fixed presentation band on the model score (critical ≥ 0.9, high ≥ 0.5, medium ≥ 0.1, low ≥ threshold),
        not a model output. Triage status is analyst workflow state, stored beside the export and never used by any model or
        metric.
      </Provenance>
    </div>
  );
}

function AlertDrawer({ alert, onUpdated }: { alert: AlertRecord; onUpdated: (alert: AlertRecord) => void }) {
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [value, setValue] = useState<TriageStatus>(alert.status);
  const [note, setNote] = useState(alert.note);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<AlertDetail>(`/api/alerts/${encodeURIComponent(alert.event_id)}`).then(setDetail).catch((err) => setError((err as Error).message));
  }, [alert.event_id]);

  const save = async () => {
    setSaving(true);
    try {
      const result = await patch<{ status: TriageStatus; note: string; updated_at: string }>(
        `/api/alerts/${encodeURIComponent(alert.event_id)}`,
        { status: value, note },
      );
      onUpdated({ ...alert, status: result.status, note: result.note, status_updated: result.updated_at });
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const top = detail
    ? [...detail.features].filter((f) => f.value !== 0).slice(0, 17)
    : [];

  return (
    <Card
      title={
        <span className="mono">
          {alert.source} → {alert.destination}
        </span>
      }
      subtitle={alert.event_id}
      actions={<InvestigateLink eventId={alert.event_id}>Open investigation →</InvestigateLink>}
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <SeverityBadge severity={alert.severity} />
          <GroundTruthBadge value={alert.ground_truth} />
        </div>
        <KeyValue
          items={[
            ["Time", <span key="t" className="mono">{lanlTime(alert.timestamp)}</span>],
            ["User", alert.user],
            ["Score", <span key="s" className="mono text-soc-danger">{alert.score.toFixed(4)}</span>],
            ["Authentication", alert.success ? "Success" : <span key="f" className="text-soc-warn">Failed</span>],
          ]}
        />
        <div>
          <div className="label mb-1.5">Non-zero behavioural features</div>
          {detail ? (
            <div className="space-y-1">
              {top.map((f) => (
                <div key={f.name} className="flex justify-between gap-3 text-[11.5px]" title={FEATURE_HELP[f.name]}>
                  <span className="truncate text-soc-muted">{f.name}</span>
                  <span className="mono text-soc-text">{Number.isInteger(f.value) ? f.value : f.value.toFixed(2)}</span>
                </div>
              ))}
            </div>
          ) : (
            <Spinner />
          )}
        </div>
        <div className="space-y-2 border-t border-soc-line pt-3">
          <div className="label">Investigation status</div>
          <select className="input w-full" value={value} onChange={(e) => setValue(e.target.value as TriageStatus)}>
            {(Object.keys(STATUS_LABEL) as TriageStatus[]).map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
          </select>
          <textarea
            className="input h-20 w-full resize-none py-1.5"
            placeholder="Analyst note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <button className="btn btn-primary w-full justify-center" onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save status"}
          </button>
          {alert.status_updated && <p className="text-[11px] text-soc-faint">Last updated {new Date(alert.status_updated).toLocaleString()}</p>}
          <ErrorNote error={error} />
        </div>
      </div>
    </Card>
  );
}
