"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useReplay } from "@/lib/replay";

const NAV = [
  { href: "/", label: "Overview", icon: "M3 13h8V3H3zm10 8h8V11h-8zM3 21h8v-6H3zm10-18v6h8V3z" },
  { href: "/graph", label: "Network Graph", icon: "M6 6a2 2 0 1 0 0-.01M18 6a2 2 0 1 0 0-.01M12 18a2 2 0 1 0 0-.01M7.5 7l3.5 9M16.5 7l-3.5 9M8 6h8" },
  { href: "/investigation", label: "Investigation", icon: "M10 4a6 6 0 1 0 0 12 6 6 0 0 0 0-12zm9 17-4.3-4.3" },
  { href: "/alerts", label: "Alerts", icon: "M12 3 2 20h20L12 3zm0 6v5m0 3v.01" },
  { href: "/experiments", label: "Experiments", icon: "M4 20V10m6 10V4m6 16v-8m4 8H2" },
  { href: "/pipeline", label: "Pipeline", icon: "M3 7h5v10H3zm8-3h5v16h-5zm8 5h2v6h-2M8 12h3m5 0h3" },
  { href: "/lab", label: "Demo Lab", icon: "M9 3h6M10 3v6L4 19a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 19l-6-10V3" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { alerts, counters } = useReplay();
  const open = alerts.filter((a) => a.status === "new").length;

  return (
    <aside className="flex w-[212px] shrink-0 flex-col border-r border-soc-line bg-[#0c1016]">
      <div className="flex items-center gap-2.5 border-b border-soc-line px-4 py-3.5">
        <div className="grid h-7 w-7 place-items-center rounded bg-soc-accent/15 ring-1 ring-soc-accent/40">
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-soc-accent" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2 4 6v6c0 5 3.5 8.5 8 10 4.5-1.5 8-5 8-10V6l-8-4z" />
          </svg>
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold tracking-tight">LMD Console</div>
          <div className="text-[10.5px] text-soc-faint">Lateral movement · LANL</div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-2">
        {NAV.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          const isLab = item.href === "/lab";
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group relative flex items-center gap-2.5 rounded px-2.5 py-2 text-[12.5px] transition-colors ${
                active ? "bg-soc-raised text-soc-text" : "text-soc-muted hover:bg-soc-panel hover:text-soc-text"
              }`}
            >
              {active && <span className="absolute inset-y-1.5 left-0 w-[2px] rounded-r bg-soc-accent" />}
              <svg viewBox="0 0 24 24" className={`h-4 w-4 ${active ? "text-soc-accent" : "text-soc-faint group-hover:text-soc-muted"}`} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d={item.icon} />
              </svg>
              <span className="flex-1">{item.label}</span>
              {item.href === "/alerts" && open > 0 && (
                <span className="rounded bg-soc-danger/20 px-1.5 text-[10.5px] font-semibold text-soc-danger">{open}</span>
              )}
              {isLab && <span className="rounded border border-soc-warn/40 px-1 text-[9.5px] font-semibold text-soc-warn">LAB</span>}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-1.5 border-t border-soc-line px-4 py-3 text-[11px] text-soc-faint">
        <div className="flex justify-between">
          <span>Events replayed</span>
          <span className="tabular-nums text-soc-muted">{counters ? counters.events.toLocaleString() : "—"}</span>
        </div>
        <div className="flex justify-between">
          <span>Alerts raised</span>
          <span className="tabular-nums text-soc-muted">{counters ? counters.alerts.toLocaleString() : "—"}</span>
        </div>
        <p className="pt-1 leading-snug">Real LANL W2 events, replayed. Detector frozen after W1.</p>
      </div>
    </aside>
  );
}
