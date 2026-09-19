"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, patch, type AlertContext, type AlertDetail, type AlertRecord, type Explanation, type TriageStatus } from "@/lib/api";
import { useReplay } from "@/lib/replay";
import { DEFAULT_THRESHOLD, useStatus } from "@/lib/status";
import { FEATURE_HELP, STATUS_LABEL, duration, lanlTime, pct } from "@/lib/format";
import { ShapBars } from "@/components/charts";
import { AlertListItem } from "@/components/EventTable";
import { Badge, Card, Empty, ErrorNote, GroundTruthBadge, Provenance, SeverityBadge, Spinner, StatusBadge } from "@/components/ui";
import { PageTitle } from "@/components/ui";

export default function InvestigationPage() {
  return (
    <Suspense fallback={<Empty><Spinner /></Empty>}>
      <Investigation />
    </Suspense>
  );
}

function Investigation() {
  const params = useSearchParams();
  const router = useRouter();
  const eventId = params.get("id");
  const { alerts } = useReplay();
  const [lookup, setLookup] = useState("");

  return (
    <div className="space-y-4">
      <PageTitle title="Investigation" subtitle="Evidence behind one detection: path, timeline, features, explanation and ground truth" />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[300px_1fr]">
        <Card title="Alert queue" subtitle="Replayed alerts, newest first" bodyClass="p-0" className="xl:sticky xl:top-0 xl:self-start">
          <form
            className="flex gap-2 border-b border-soc-line p-3"
            onSubmit={(event) => {
              event.preventDefault();
              if (lookup.trim()) router.push(`/investigation?id=${encodeURIComponent(lookup.trim())}`);
            }}
          >
            <input className="input flex-1" placeholder="Event id (lanl_…)" value={lookup} onChange={(e) => setLookup(e.target.value)} />
            <button className="btn" type="submit">Open</button>
          </form>
          <div className="max-h-[calc(100vh-240px)] overflow-y-auto">
            {alerts.length ? (
              alerts.slice(0, 80).map((alert) => <AlertListItem key={alert.event_id} alert={alert} active={alert.event_id === eventId} />)
            ) : (
              <Empty>No alerts yet</Empty>
            )}
          </div>
        </Card>
        {eventId ? <Case key={eventId} eventId={eventId} /> : <Empty>Select an alert from the queue to open a case.</Empty>}
      </div>
    </div>
  );
}

function Case({ eventId }: { eventId: string }) {
  const status = useStatus();
  const threshold = status?.model.threshold ?? DEFAULT_THRESHOLD;
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [context, setContext] = useState<AlertContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [relationFilter, setRelationFilter] = useState<string>("all");

  useEffect(() => {
    const id = encodeURIComponent(eventId);
    setError(null);
    Promise.all([
      api<AlertDetail>(`/api/alerts/${id}`),
      api<Explanation>(`/api/alerts/${id}/explain`),
      api<AlertContext>(`/api/alerts/${id}/context`),
    ])
      .then(([d, e, c]) => {
        setDetail(d);
        setExplanation(e);
        setContext(c);
      })
      .catch((err) => setError((err as Error).message));
  }, [eventId]);

  const related = useMemo(() => {
    if (!context) return [];
    return context.related.events.filter((e) => relationFilter === "all" || e.relation.includes(relationFilter));
  }, [context, relationFilter]);

  if (error) return <ErrorNote error={error} />;
  if (!detail || !explanation || !context) return <Empty><Spinner /></Empty>;
  const event = detail.event;
  const isAlert = event.score >= threshold;

  return (
    <div className="space-y-4">
      <div className="panel p-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <SeverityBadge severity={event.severity} />
              {"status" in event && <StatusBadge status={(event as AlertRecord).status} />}
              <GroundTruthBadge value={event.ground_truth} />
              {!event.success && <Badge className="border-soc-warn/40 text-soc-warn">Failed auth</Badge>}
            </div>
            <div className="mono text-[20px] font-semibold tracking-tight">
              {event.source} <span className="text-soc-faint">→</span> {event.destination}
            </div>
            <div className="text-[12.5px] text-soc-muted">
              <span className="text-soc-text">{event.user}</span> · {lanlTime(event.timestamp)} · <span className="mono">{event.event_id}</span>
            </div>
          </div>
          <ScoreGauge score={event.score} threshold={threshold} alert={isAlert} />
        </div>
        {"status" in event && <Triage alert={event as AlertRecord} />}
      </div>

      <Card title="Movement trace" subtitle={`Heuristic: follows account ${context.path.account} through the alert's hosts`}>
        <div className="flex items-stretch gap-0 overflow-x-auto pb-1">
          {context.path.hops.map((hop, i) => (
            <div key={hop.event_id} className="flex items-center">
              {i === 0 && <HostChip name={hop.source} />}
              <div className="flex flex-col items-center px-1">
                <div className={`mono text-[10px] ${hop.anchor ? "text-soc-danger" : "text-soc-faint"}`}>
                  {hop.anchor ? "ALERT" : duration(hop.timestamp - event.timestamp)}
                </div>
                <div className={`relative h-[2px] w-24 ${hop.anchor ? "bg-soc-danger" : hop.alert ? "bg-soc-danger/60" : "bg-soc-lineStrong"}`}>
                  <span className={`absolute -right-1 -top-[3px] h-2 w-2 rotate-45 border-r-2 border-t-2 ${hop.anchor ? "border-soc-danger" : "border-soc-lineStrong"}`} />
                </div>
                <div className="max-w-24 truncate text-[10px] text-soc-muted" title={hop.user}>{hop.user}</div>
                <div className="mono text-[10px] text-soc-faint">{hop.score.toFixed(3)}</div>
              </div>
              <HostChip name={hop.destination} active={hop.anchor} />
            </div>
          ))}
        </div>
        <div className="mt-3"><Provenance>{context.path.note}</Provenance></div>
      </Card>

      <Card
        title="Source fan-out"
        subtitle={`${context.fanout.source} reached ${context.fanout.distinct_destinations} distinct destinations in the ${Math.round(context.fanout.lookback_seconds / 60)} minutes up to this alert · ${context.fanout.alerted} alerted · ${context.fanout.ground_truth} redteam`}
      >
        <div className="flex flex-wrap gap-1.5">
          {context.fanout.first_contacts.map((hop) => (
            <a
              key={hop.event_id}
              href={`/investigation?id=${encodeURIComponent(hop.event_id)}`}
              title={`${hop.user} · ${lanlTime(hop.timestamp)} · score ${hop.score.toFixed(4)}`}
              className={`mono rounded border px-2 py-1 text-[11px] transition-colors hover:border-soc-lineStrong ${
                hop.anchor
                  ? "border-soc-danger bg-soc-danger/15 text-soc-danger"
                  : hop.ground_truth
                    ? "border-soc-violet/50 bg-soc-violet/10 text-soc-violet"
                    : hop.alert
                      ? "border-soc-danger/40 text-soc-danger"
                      : "border-soc-line text-soc-muted"
              }`}
            >
              {hop.destination}
              <span className="ml-1.5 text-soc-faint">{duration(hop.offset_seconds)}</span>
            </a>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-[10.5px] text-soc-faint">
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-soc-danger" />this alert / alerted</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-soc-violet" />redteam (ground truth)</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-soc-lineStrong" />benign, not alerted</span>
        </div>
        <div className="mt-2"><Provenance>{context.fanout.note}</Provenance></div>
      </Card>

      <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
        <Card title="TreeSHAP explanation" subtitle={explanation.method}>
          <ShapBars contributions={explanation.contributions} bias={explanation.bias} margin={explanation.margin} limit={12} />
          <div className="mt-2">
            <Provenance>
              Red pushes toward “attack”, blue toward “benign”. Exact Shapley values of the frozen model; local accuracy
              error {explanation.local_accuracy_error?.toExponential(1)}. Correlated features share credit (PROJECT_STATE §30.10).
            </Provenance>
          </div>
        </Card>

        <Card title="17 behavioural features" subtitle="Causal: computed only from events strictly before this one (M3.4)" bodyClass="p-0">
          <table className="w-full text-[12px]">
            <tbody>
              {detail.features.map((feature) => {
                const phi = explanation.contributions.find((c) => c.name === feature.name)?.contribution ?? 0;
                return (
                  <tr key={feature.name} className="table-row">
                    <td className="px-4 py-1.5">
                      <div className="text-soc-text">{feature.name}</div>
                      <div className="text-[10.5px] text-soc-faint">{FEATURE_HELP[feature.name]}</div>
                    </td>
                    <td className="mono px-2 text-right tabular-nums text-soc-text">
                      {Number.isInteger(feature.value) ? feature.value : feature.value.toFixed(2)}
                    </td>
                    <td className={`mono w-20 px-4 text-right tabular-nums ${phi > 0.05 ? "text-soc-danger" : phi < -0.05 ? "text-soc-accent" : "text-soc-faint"}`}>
                      {phi >= 0 ? "+" : ""}
                      {phi.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      </div>

      <Card
        title="Related activity timeline"
        subtitle={`${context.related.total_matching.toLocaleString()} events share this user or host within ±${Math.round(context.related.window_seconds / 60)} min (showing ${context.related.returned} nearest, up to the replay clock)`}
        bodyClass="p-0"
        actions={
          <select className="input h-7" value={relationFilter} onChange={(e) => setRelationFilter(e.target.value)}>
            <option value="all">All relations</option>
            <option value="same_user">Same user</option>
            <option value="touches_source">Touches source host</option>
            <option value="touches_destination">Touches destination host</option>
          </select>
        }
      >
        <div className="max-h-[440px] overflow-y-auto px-4 py-3">
          <ol className="relative space-y-0 border-l border-soc-line pl-4">
            {related.map((item) => {
              const anchor = item.event_id === event.event_id;
              return (
                <li key={item.event_id} className="relative py-1">
                  <span
                    className={`absolute -left-[21px] top-2.5 h-2 w-2 rounded-full ${
                      anchor ? "bg-soc-danger ring-4 ring-soc-danger/20" : item.alert ? "bg-soc-danger/70" : item.ground_truth ? "bg-soc-violet" : "bg-soc-lineStrong"
                    }`}
                  />
                  <div className={`grid grid-cols-[76px_120px_1fr_150px_70px] items-center gap-2 rounded px-2 py-1 text-[12px] ${anchor ? "bg-soc-danger/10" : "hover:bg-soc-raised"}`}>
                    <span className={`mono ${anchor ? "text-soc-danger" : "text-soc-faint"}`}>{anchor ? "ALERT" : duration(item.offset_seconds)}</span>
                    <span className="mono text-soc-muted">{lanlTime(item.timestamp, false)}</span>
                    <span className="mono truncate">
                      {item.source} <span className="text-soc-faint">→</span> {item.destination}
                      {!item.success && <span className="ml-1 text-[10px] text-soc-warn">FAIL</span>}
                    </span>
                    <span className="truncate text-soc-muted" title={item.user}>{item.user}</span>
                    <span className={`mono text-right ${item.alert ? "text-soc-danger" : "text-soc-faint"}`}>{item.score.toFixed(3)}</span>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      </Card>

      <Card title="Ground-truth status">
        <div className="flex items-center gap-3">
          <GroundTruthBadge value={event.ground_truth} />
          <span className="text-[12.5px] text-soc-muted">
            {event.ground_truth
              ? "This event exactly matches a LANL redteam record on (timestamp, user, source, destination)."
              : "No LANL redteam record matches this event exactly."}{" "}
            {isAlert ? (event.ground_truth ? "Outcome: true positive." : "Outcome: false positive.") : event.ground_truth ? "Outcome: missed (false negative)." : ""}
          </span>
        </div>
        <div className="mt-2"><Provenance>{detail.ground_truth_note}</Provenance></div>
      </Card>
    </div>
  );
}

function ScoreGauge({ score, threshold, alert }: { score: number; threshold: number; alert: boolean }) {
  const angle = Math.min(1, score) * 180;
  const t = Math.min(1, threshold) * 180;
  const point = (deg: number, r: number) => {
    const rad = ((180 - deg) * Math.PI) / 180;
    return [60 + r * Math.cos(rad), 60 - r * Math.sin(rad)];
  };
  const [ex, ey] = point(angle, 46);
  const [tx, ty] = point(t, 52);
  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 120 68" className="h-[68px] w-[120px]">
        <path d="M14 60 A46 46 0 0 1 106 60" fill="none" stroke="#1e2530" strokeWidth="9" strokeLinecap="round" />
        <path d={`M14 60 A46 46 0 0 1 ${ex} ${ey}`} fill="none" stroke={alert ? "#f43f5e" : "#38bdf8"} strokeWidth="9" strokeLinecap="round" className="transition-all duration-700" />
        <line x1={tx} y1={ty} x2={point(t, 38)[0]} y2={point(t, 38)[1]} stroke="#f59e0b" strokeWidth="2" />
      </svg>
      <div>
        <div className={`text-[26px] font-semibold tabular-nums ${alert ? "text-soc-danger" : "text-soc-accent"}`}>{pct(score, 2)}</div>
        <div className="text-[11px] text-soc-muted">detection score · threshold <span className="mono text-soc-warn">{threshold.toFixed(5)}</span></div>
      </div>
    </div>
  );
}

function HostChip({ name, active = false }: { name: string; active?: boolean }) {
  return (
    <div
      className={`mono shrink-0 rounded border px-2.5 py-1.5 text-[12px] ${
        active ? "border-soc-danger/60 bg-soc-danger/10 text-soc-danger" : "border-soc-line bg-soc-raised text-soc-text"
      }`}
    >
      {name}
    </div>
  );
}

function Triage({ alert }: { alert: AlertRecord }) {
  const status = useStatus();
  const [value, setValue] = useState<TriageStatus>(alert.status);
  const [note, setNote] = useState(alert.note);
  const [saved, setSaved] = useState<string | null>(alert.status_updated);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    setSaving(true);
    try {
      const result = await patch<{ updated_at: string }>(`/api/alerts/${encodeURIComponent(alert.event_id)}`, { status: value, note });
      setSaved(result.updated_at);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-soc-line pt-3">
      <span className="label mr-1">Triage</span>
      <select className="input" value={value} onChange={(e) => setValue(e.target.value as TriageStatus)}>
        {(status?.triage_statuses ?? (Object.keys(STATUS_LABEL) as TriageStatus[])).map((s) => (
          <option key={s} value={s}>{STATUS_LABEL[s]}</option>
        ))}
      </select>
      <input className="input min-w-64 flex-1" placeholder="Analyst note" value={note} onChange={(e) => setNote(e.target.value)} />
      <button className="btn btn-primary" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
      {saved && <span className="text-[11px] text-soc-faint">saved {new Date(saved).toLocaleTimeString()}</span>}
      {error && <ErrorNote error={error} />}
    </div>
  );
}
