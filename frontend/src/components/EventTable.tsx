"use client";

import Link from "next/link";
import type { AlertRecord, EventRecord } from "@/lib/api";
import { lanlTime } from "@/lib/format";
import { GroundTruthBadge, Hosts, ScoreBar, SeverityBadge, StatusBadge } from "./ui";

export function EventTable({
  events,
  threshold,
  showStatus = false,
  flashKey,
  onSelect,
  selectedId,
  dense = false,
}: {
  events: (EventRecord | AlertRecord)[];
  threshold: number;
  showStatus?: boolean;
  flashKey?: boolean;
  onSelect?: (event: EventRecord) => void;
  selectedId?: string | null;
  dense?: boolean;
}) {
  return (
    <table className="w-full table-fixed text-left text-[12px]">
      <thead className="sticky top-0 z-10 bg-soc-panel">
        <tr className="border-b border-soc-line text-[10.5px] uppercase tracking-wider text-soc-faint">
          <th className="w-[120px] px-3 py-2 font-medium">Time</th>
          <th className="w-[150px] px-2 py-2 font-medium">User</th>
          <th className="px-2 py-2 font-medium">Source → Destination</th>
          <th className="w-[128px] px-2 py-2 font-medium">Score</th>
          <th className="w-[74px] px-2 py-2 font-medium">Severity</th>
          <th className="w-[74px] px-2 py-2 font-medium">Truth</th>
          {showStatus && <th className="w-[108px] px-2 py-2 font-medium">Status</th>}
        </tr>
      </thead>
      <tbody>
        {events.map((event) => {
          const status = (event as AlertRecord).status;
          const selected = selectedId === event.event_id;
          return (
            <tr
              key={event.event_id}
              onClick={onSelect ? () => onSelect(event) : undefined}
              className={`table-row ${onSelect ? "cursor-pointer" : ""} ${
                selected ? "bg-soc-accent/10" : ""
              } ${flashKey ? (event.alert ? "animate-flashDanger" : "animate-fadeIn") : ""}`}
            >
              <td className={`mono px-3 ${dense ? "py-1" : "py-1.5"} text-soc-muted`}>{lanlTime(event.timestamp)}</td>
              <td className="truncate px-2 text-soc-text" title={event.user}>
                {event.user}
                {!event.success && <span className="ml-1.5 text-[10px] font-semibold text-soc-warn">FAIL</span>}
              </td>
              <td className="truncate px-2">
                <Hosts source={event.source} destination={event.destination} />
              </td>
              <td className="px-2"><ScoreBar value={event.score} threshold={threshold} /></td>
              <td className="px-2"><SeverityBadge severity={event.severity} /></td>
              <td className="px-2">{event.ground_truth ? <GroundTruthBadge value /> : <span className="text-soc-faint">—</span>}</td>
              {showStatus && <td className="px-2">{status ? <StatusBadge status={status} /> : null}</td>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function AlertListItem({ alert, active }: { alert: AlertRecord; active?: boolean }) {
  return (
    <Link
      href={`/investigation?id=${encodeURIComponent(alert.event_id)}`}
      className={`block border-b border-soc-line/70 px-4 py-2.5 transition-colors hover:bg-soc-raised ${
        active ? "bg-soc-accent/10" : ""
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <SeverityBadge severity={alert.severity} />
          {alert.ground_truth && <GroundTruthBadge value />}
        </div>
        <span className="mono text-soc-faint">{lanlTime(alert.timestamp)}</span>
      </div>
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <Hosts source={alert.source} destination={alert.destination} />
        <span className="mono text-soc-danger">{alert.score.toFixed(3)}</span>
      </div>
      <div className="mt-0.5 truncate text-[11px] text-soc-muted">{alert.user}</div>
    </Link>
  );
}
