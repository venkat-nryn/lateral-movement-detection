"""Investigation context for one event: related activity and a movement trace.

Both views are gated by the replay clock: nothing after ``until`` is returned.

The movement trace is a **heuristic** built for the analyst, not a model output.
The project has no attack-path reconstruction model (``ml/path_analysis`` is
empty). The trace follows the *same account* -- credential-based lateral
movement travels with the compromised account -- backwards to where it arrived
at the source host and forwards to where it moved next from the destination.
Following any account instead drifts into unrelated machine-account traffic.

The source fan-out view is not heuristic: it lists real events, namely every
distinct destination the alert's source host first reached within the lookback.
Every response says which is which.
"""

from __future__ import annotations

import numpy as np

from backend.services.store import DashboardStore

PATH_TRACE_NOTE = (
    "Heuristic temporal trace following the same account: where it arrived at "
    "each source host (backwards) and where it moved next from each destination "
    "(forwards). Not a model output; the project has no path-reconstruction model."
)

FANOUT_NOTE = (
    "Real events: the first contact with every distinct destination this source "
    "host reached in the lookback, up to and including the alert. Related to the "
    "fan-out feature (source_unique_destination_count), which counts from the "
    "start of the replay window rather than over this lookback."
)


def related_events(
    store: DashboardStore,
    row: int,
    *,
    until: float,
    window_seconds: float = 1800.0,
    limit: int = 150,
) -> dict:
    """Events sharing the user, source or destination within +/- the window."""
    t = float(store.timestamps[row])
    lo = int(np.searchsorted(store.timestamps, t - window_seconds, side="left"))
    hi = int(np.searchsorted(store.timestamps, min(t + window_seconds, until), side="right"))
    user, source, destination = store.user[row], store.source[row], store.destination[row]
    hosts = (source, destination)

    mask = (
        (store.user[lo:hi] == user)
        | np.isin(store.source[lo:hi], hosts)
        | np.isin(store.destination[lo:hi], hosts)
    )
    rows = np.flatnonzero(mask) + lo
    total = int(rows.size)
    if total > limit:
        nearest = np.argsort(np.abs(store.timestamps[rows] - t), kind="stable")[:limit]
        rows = np.sort(rows[nearest])

    events = []
    for r in rows:
        record = store.event(int(r))
        record["relation"] = sorted(
            name
            for name, hit in (
                ("same_user", store.user[r] == user),
                ("touches_source", store.source[r] == source or store.destination[r] == source),
                ("touches_destination", store.source[r] == destination or store.destination[r] == destination),
            )
            if hit
        )
        record["offset_seconds"] = record["timestamp"] - t
        events.append(record)
    return {
        "anchor_row": int(row),
        "window_seconds": window_seconds,
        "total_matching": total,
        "returned": len(events),
        "events": events,
    }


def movement_path(
    store: DashboardStore,
    row: int,
    *,
    until: float,
    lookback_seconds: float = 3600.0,
    max_hops: int = 6,
) -> dict:
    """Heuristic multi-hop movement trace of the alert's account."""
    anchor = store.event(row)
    account = store.user[row]
    hops_back: list[dict] = []
    host = store.source[row]
    t = float(store.timestamps[row])
    visited = {int(store.destination[row]), int(host)}
    for _ in range(max_hops):
        lo = int(np.searchsorted(store.timestamps, t - lookback_seconds, side="left"))
        hi = int(np.searchsorted(store.timestamps, t, side="left"))
        candidates = np.flatnonzero(
            (store.destination[lo:hi] == host)
            & (store.source[lo:hi] != host)
            & (store.user[lo:hi] == account)
            & store.success[lo:hi]
        )
        if not candidates.size:
            break
        r = int(candidates[-1] + lo)
        hops_back.append(store.event(r))
        host = store.source[r]
        t = float(store.timestamps[r])
        if int(host) in visited:
            break
        visited.add(int(host))
    hops_back.reverse()

    hops_forward: list[dict] = []
    host = store.destination[row]
    t = float(store.timestamps[row])
    visited = {int(store.source[row]), int(host)}
    for _ in range(max_hops):
        lo = int(np.searchsorted(store.timestamps, t, side="right"))
        hi = int(np.searchsorted(store.timestamps, min(t + lookback_seconds, until), side="right"))
        if hi <= lo:
            break
        candidates = np.flatnonzero(
            (store.source[lo:hi] == host)
            & (store.destination[lo:hi] != host)
            & (store.user[lo:hi] == account)
            & store.success[lo:hi]
        )
        if not candidates.size:
            break
        r = int(candidates[0] + lo)
        hops_forward.append(store.event(r))
        host = store.destination[r]
        t = float(store.timestamps[r])
        if int(host) in visited:
            break
        visited.add(int(host))

    return {
        "note": PATH_TRACE_NOTE,
        "lookback_seconds": lookback_seconds,
        "account": store.users[account],
        "hops": hops_back + [dict(anchor, anchor=True)] + hops_forward,
    }


def source_fanout(
    store: DashboardStore,
    row: int,
    *,
    lookback_seconds: float = 3600.0,
    limit: int = 60,
) -> dict:
    """First contact with each distinct destination of the alert's source host.

    Only rows up to and including the alert are considered, so every returned
    event precedes or is the alert; nothing later can appear.
    """
    t = float(store.timestamps[row])
    source = store.source[row]
    lo = int(np.searchsorted(store.timestamps, t - lookback_seconds, side="left"))
    hi = row + 1
    rows = np.flatnonzero(
        (store.source[lo:hi] == source) & (store.destination[lo:hi] != source)
    ) + lo
    seen: set[int] = set()
    first_contacts: list[dict] = []
    for r in rows:
        destination = int(store.destination[r])
        if destination in seen:
            continue
        seen.add(destination)
        record = store.event(int(r))
        record["offset_seconds"] = record["timestamp"] - t
        record["anchor"] = int(r) == row
        first_contacts.append(record)
    return {
        "note": FANOUT_NOTE,
        "source": store.hosts[source],
        "lookback_seconds": lookback_seconds,
        "distinct_destinations": len(first_contacts),
        "alerted": sum(1 for e in first_contacts if e["alert"]),
        "ground_truth": sum(1 for e in first_contacts if e["ground_truth"]),
        "first_contacts": first_contacts[-limit:],
    }
