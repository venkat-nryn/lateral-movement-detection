"use client";

import { useReplay } from "@/lib/replay";
import { lanlTime } from "@/lib/format";

export function ReplayBar() {
  const { connection, replay, control } = useReplay();
  const live = connection === "live";

  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-soc-line bg-[#0c1016]/95 px-5 backdrop-blur">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${
            !live ? "bg-soc-danger" : replay?.playing ? "animate-pulseDot bg-soc-ok" : "bg-soc-warn"
          }`}
        />
        <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-soc-muted">
          {!live ? "Backend offline" : replay?.finished ? "Replay complete" : replay?.playing ? "Live replay" : "Replay paused"}
        </span>
      </div>

      <div className="flex items-center gap-1.5">
        <button
          className="btn h-7 px-2"
          disabled={!replay}
          title={replay?.playing ? "Pause" : "Play"}
          onClick={() => control(replay?.playing ? "pause" : "play")}
        >
          {replay?.playing ? (
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M6 5h4v14H6zm8 0h4v14h-4z" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M7 5v14l12-7z" /></svg>
          )}
        </button>
        <button className="btn h-7 px-2" disabled={!replay} title="Restart replay" onClick={() => control("restart")}>
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.2"><path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5" /></svg>
        </button>
        <div className="ml-1 inline-flex rounded border border-soc-line bg-soc-bg p-0.5">
          {(replay?.speeds ?? [1, 5, 20, 60, 300]).map((speed) => (
            <button
              key={speed}
              onClick={() => control("speed", speed)}
              className={`h-6 rounded-[3px] px-2 text-[11px] font-medium tabular-nums transition-colors ${
                replay?.speed === speed ? "bg-soc-raised text-soc-accent" : "text-soc-faint hover:text-soc-text"
              }`}
            >
              {speed}×
            </button>
          ))}
        </div>
      </div>

      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span className="mono shrink-0 tabular-nums text-soc-text" title="LANL time: day and time since the start of the dataset">
          {replay ? lanlTime(replay.position) : "—"}
        </span>
        <div
          className="group relative h-1.5 flex-1 cursor-pointer overflow-hidden rounded-full bg-soc-line"
          title="Seek within W2"
          onClick={(event) => {
            if (!replay) return;
            const rect = event.currentTarget.getBoundingClientRect();
            const fraction = (event.clientX - rect.left) / rect.width;
            void control("seek", replay.start + fraction * (replay.end - replay.start));
          }}
        >
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-soc-accent/60 to-soc-accent transition-[width] duration-500 ease-linear"
            style={{ width: `${(replay?.progress ?? 0) * 100}%` }}
          />
        </div>
        <span className="mono shrink-0 text-soc-faint">{replay ? lanlTime(replay.end) : ""}</span>
      </div>

      <div className="hidden shrink-0 items-center gap-2 xl:flex">
        <span className="rounded border border-soc-line px-2 py-0.5 text-[10.5px] text-soc-muted">
          Source: <span className="text-soc-text">real LANL · W2</span>
        </span>
        <span className="rounded border border-soc-line px-2 py-0.5 text-[10.5px] text-soc-muted">
          Detector: <span className="text-soc-text">XGBoost (frozen)</span>
        </span>
      </div>
    </header>
  );
}
