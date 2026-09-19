"""Bounded real-LANL labeled window selection (M3.6).

M3.5 showed that a leading prefix of ``auth.txt.gz`` contains no redteam
activity at all: the first 1,000,000 events end at timestamp 10,208 while the
earliest redteam record is at 150,885. A prefix therefore cannot support a
supervised experiment.

This module selects a bounded *timestamp* window of the real dataset that
actually contains redteam activity, and builds the M3.5 chronological dataset
over it. Nothing here invents, moves, injects or rebalances anything: the window
is chosen from ``redteam.txt.gz`` alone by a deterministic rule, and every label
still comes from the exact 4-field match used everywhere else in the project.

Window anatomy
--------------
::

    context_start        emit_start                        emit_end
    |....................|=================================|
     history only         emitted, labeled, split
     (warm-up)            train -> validation -> test

* ``[context_start, emit_start)`` -- real LANL authentication events streamed
  through the feature extractor to build realistic history. They produce no
  samples. The extractor is **not** reset at ``emit_start``, so the first
  emitted sample already sees genuine prior behaviour.
* ``[emit_start, emit_end]`` -- the emitted region. Chronological
  train/validation/test boundaries are computed over this region only.

Cost control
------------
``auth.txt.gz`` is a single gzip stream, so reaching a window necessarily means
reading the compressed bytes before it. The adapter skips those records by
parsing only the leading timestamp field (measured ~1.4M lines/s versus ~220k
lines/s for a full parse) and stops reading the moment a record past
``emit_end`` appears. The remainder of the file is never touched, never
decompressed to disk and never materialized.

Completeness
------------
``max_events`` bounds one build so that a mis-specified window cannot degenerate
into a full-dataset run. It is a guard, not a trim: a window has a defined
extent, so a build that would cover less than the whole window raises
:class:`WindowTruncationError` rather than silently returning a shortened
dataset. Until M4.7 the bound was applied as a silent ``break``, which cut W2
short by its last 19 minutes and 16 of its 92 redteam records; see
PROJECT_STATE.md section 33.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from ml.evaluation.redteam import RedteamGroundTruth, RedteamRecord
from ml.preprocessing.features import DEFAULT_RECENT_WINDOW_SECONDS
from ml.preprocessing.lanl_adapter import iter_lanl_events
from ml.preprocessing.ml_dataset import (
    ChronologicalSplitConfig,
    MLDataset,
    TimestampRange,
)
from ml.preprocessing.schema import CanonicalEvent

#: Emitted window length. Two hours of real LANL time is the smallest span that
#: still places redteam records in all three chronological splits at the default
#: 70/15/15 boundaries, while keeping the retained sample count near the
#: 1,000,000-event scale already validated in M3.5.
DEFAULT_WINDOW_SPAN_SECONDS = 7200.0

#: Real authentication history streamed before the emitted region.
DEFAULT_CONTEXT_SECONDS = 3600.0

#: Hard ceiling on events streamed for one window build (context + emitted).
#: Prevents a mis-specified window from degenerating into a full-dataset run.
#: It must exceed the largest defined window: W2 reads 3,341,375 events
#: (1,167,143 context + 2,174,232 emitted). Reaching the ceiling is an error and
#: never a silent truncation -- see :class:`WindowTruncationError`.
DEFAULT_MAX_WINDOW_EVENTS = 4_000_000


class WindowSelectionError(ValueError):
    """Raised when no usable real window can be selected."""


class WindowTruncationError(WindowSelectionError):
    """Raised when a window build would not cover the whole window.

    The event ceiling exists to stop a runaway build, not to shorten a window
    that was specified correctly. Silently returning a partial window produced
    the W2 defect recorded in PROJECT_STATE.md section 33, so the build now
    fails and asks for a larger ceiling instead.
    """


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """A bounded, real-LANL timestamp window.

    All timestamps are genuine LANL values; none are shifted or invented.
    """

    context_start: float
    emit_start: float
    emit_end: float
    redteam_records_in_window: int

    def __post_init__(self) -> None:
        if self.context_start > self.emit_start:
            raise WindowSelectionError(
                "context_start must not be later than emit_start"
            )
        if self.emit_end <= self.emit_start:
            raise WindowSelectionError("emit_end must be later than emit_start")

    @property
    def span_seconds(self) -> float:
        return self.emit_end - self.emit_start

    @property
    def context_seconds(self) -> float:
        return self.emit_start - self.context_start

    def contains_emitted(self, timestamp: float) -> bool:
        return self.emit_start <= timestamp <= self.emit_end

    def is_context(self, timestamp: float) -> bool:
        return self.context_start <= timestamp < self.emit_start

    def timestamp_range(self) -> TimestampRange:
        """Split boundaries are defined over the emitted region only."""
        return TimestampRange(start=self.emit_start, end=self.emit_end)


def _occupies(spec: WindowSpec) -> tuple[float, float]:
    """The full timestamp region a window reads, context included."""
    return spec.context_start, spec.emit_end


def _overlaps(
    candidate_start: float,
    candidate_end: float,
    excluded: Sequence[WindowSpec],
) -> bool:
    for spec in excluded:
        low, high = _occupies(spec)
        if not (candidate_end < low or candidate_start > high):
            return True
    return False


def select_densest_redteam_window(
    ground_truth: RedteamGroundTruth,
    *,
    span_seconds: float = DEFAULT_WINDOW_SPAN_SECONDS,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    exclude: Sequence[WindowSpec] = (),
) -> WindowSpec:
    """Select the densest real redteam window of ``span_seconds``.

    The window start is the redteam timestamp maximising the number of redteam
    records in ``[start, start + span_seconds]``; ties break to the earliest
    start. The rule depends only on ``redteam.txt.gz``, never on authentication
    data and never on any model output, so it is reproducible and cannot be
    tuned toward a result.

    ``exclude`` rejects candidates whose read region ``[context_start,
    emit_end]`` touches that of an already-selected window. This is how a
    second, temporally disjoint window is chosen without changing the rule that
    picks it.

    ``context_start`` is clamped at 0 because LANL timestamps start at 1.
    """
    if span_seconds <= 0:
        raise WindowSelectionError("span_seconds must be positive")
    if context_seconds < 0:
        raise WindowSelectionError("context_seconds must be non-negative")

    timestamps = sorted(record.timestamp for record in ground_truth.iter_records())
    if not timestamps:
        raise WindowSelectionError(
            "redteam ground truth is empty; no real window can be selected"
        )

    best_count = -1
    best_start: float | None = None
    for index, start in enumerate(timestamps):
        if exclude and _overlaps(
            max(0.0, start - context_seconds), start + span_seconds, exclude
        ):
            continue
        limit = start + span_seconds
        count = 0
        for timestamp in timestamps[index:]:
            if timestamp > limit:
                break
            count += 1
        if count > best_count:
            best_count = count
            best_start = start

    if best_start is None:
        raise WindowSelectionError(
            "no redteam window remains after applying the exclusions"
        )

    return WindowSpec(
        context_start=max(0.0, best_start - context_seconds),
        emit_start=best_start,
        emit_end=best_start + span_seconds,
        redteam_records_in_window=best_count,
    )


@dataclass(slots=True)
class WindowStatistics:
    """Cardinality statistics observed while the window streams past.

    Only distinct identifier sets are retained (tens of thousands of short
    strings), never the events themselves.
    """

    context_events: int = 0
    window_events: int = 0
    users: set[str] = None  # type: ignore[assignment]
    source_hosts: set[str] = None  # type: ignore[assignment]
    destination_hosts: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.context_events = 0
        self.window_events = 0
        self.users = set()
        self.source_hosts = set()
        self.destination_hosts = set()

    def observe(self, event: CanonicalEvent, *, emitted: bool) -> None:
        if not emitted:
            self.context_events += 1
            return
        self.window_events += 1
        self.users.add(event.user)
        self.source_hosts.add(event.source_host)
        self.destination_hosts.add(event.destination_host)

    def as_dict(self) -> dict[str, int]:
        return {
            "context_events": self.context_events,
            "window_events": self.window_events,
            "unique_users": len(self.users),
            "unique_source_hosts": len(self.source_hosts),
            "unique_destination_hosts": len(self.destination_hosts),
        }


def iter_window_events(
    auth_path: Path | str,
    spec: WindowSpec,
    *,
    statistics: WindowStatistics | None = None,
) -> Iterator[CanonicalEvent]:
    """Stream the real events of ``spec`` (context region included).

    Reading stops as soon as a record later than ``spec.emit_end`` is seen.
    """
    if statistics is not None:
        statistics.reset()
    for event in iter_lanl_events(
        auth_path,
        start_timestamp=spec.context_start,
        end_timestamp=spec.emit_end,
    ):
        if statistics is not None:
            statistics.observe(event, emitted=event.timestamp >= spec.emit_start)
        yield event


def redteam_records_in_window(
    ground_truth: RedteamGroundTruth, spec: WindowSpec
) -> list[RedteamRecord]:
    """Redteam records whose timestamp falls inside the emitted region."""
    return ground_truth.filter_by_timestamp(spec.emit_start, spec.emit_end)


@dataclass(frozen=True, slots=True)
class RedteamIdentityProfile:
    """Which identities the attacker used inside one window."""

    users: frozenset[str]
    source_hosts: frozenset[str]
    destination_hosts: frozenset[str]
    record_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.record_count,
            "unique_users": len(self.users),
            "unique_source_hosts": len(self.source_hosts),
            "unique_destination_hosts": len(self.destination_hosts),
            "source_hosts": sorted(self.source_hosts),
        }


def redteam_identity_profile(
    records: Sequence[RedteamRecord],
) -> RedteamIdentityProfile:
    """Summarise the identities appearing in a set of redteam records."""
    return RedteamIdentityProfile(
        users=frozenset(record.user for record in records),
        source_hosts=frozenset(record.source_host for record in records),
        destination_hosts=frozenset(record.destination_host for record in records),
        record_count=len(records),
    )


def identity_overlap(
    first: RedteamIdentityProfile, second: RedteamIdentityProfile
) -> dict[str, dict[str, object]]:
    """Shared versus distinct attacker identities between two windows.

    ``jaccard`` is the shared count over the union; 0.0 when both are empty.
    """

    def compare(a: frozenset[str], b: frozenset[str]) -> dict[str, object]:
        union = a | b
        shared = a & b
        return {
            "first_only": len(a - b),
            "second_only": len(b - a),
            "shared": len(shared),
            "union": len(union),
            "jaccard": (len(shared) / len(union)) if union else 0.0,
            "shared_values": sorted(shared),
        }

    return {
        "users": compare(first.users, second.users),
        "source_hosts": compare(first.source_hosts, second.source_hosts),
        "destination_hosts": compare(
            first.destination_hosts, second.destination_hosts
        ),
    }


@dataclass(slots=True)
class WindowDataset:
    """A built window: the M3.5 dataset plus window provenance."""

    spec: WindowSpec
    dataset: MLDataset
    statistics: WindowStatistics
    redteam_records_in_range: int
    redteam_unique_keys_in_range: int

    def positive_count(self) -> int:
        return self.dataset.total_positive_count()

    def negative_count(self) -> int:
        return self.dataset.events_processed - self.positive_count()

    def positive_percentage(self) -> float:
        total = self.dataset.events_processed
        return (100.0 * self.positive_count() / total) if total else 0.0

    def summary(self) -> dict[str, object]:
        summary = dict(self.dataset.summary())
        summary["window"] = {
            "context_start": self.spec.context_start,
            "emit_start": self.spec.emit_start,
            "emit_end": self.spec.emit_end,
            "span_seconds": self.spec.span_seconds,
            "context_seconds": self.spec.context_seconds,
            "redteam_records_in_window": self.spec.redteam_records_in_window,
        }
        summary["class_distribution"] = {
            "total_events": self.dataset.events_processed,
            "positive": self.positive_count(),
            "negative": self.negative_count(),
            "positive_percentage": self.positive_percentage(),
            **self.statistics.as_dict(),
        }
        summary["redteam_in_range"] = {
            "records": self.redteam_records_in_range,
            "unique_match_keys": self.redteam_unique_keys_in_range,
            "matched_events": self.positive_count(),
        }
        return summary


def build_window_dataset(
    auth_path: Path | str,
    redteam_path: Path | str,
    *,
    spec: WindowSpec | None = None,
    span_seconds: float = DEFAULT_WINDOW_SPAN_SECONDS,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    split_config: ChronologicalSplitConfig | None = None,
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
    max_events: int = DEFAULT_MAX_WINDOW_EVENTS,
    retain_samples: bool = True,
) -> WindowDataset:
    """Build the bounded real-LANL labeled window dataset.

    A single streaming pass is used: the emitted timestamp range is known from
    ``spec``, so the M3.5 range-discovery pass is skipped rather than paying for
    a second seek through the gzip stream.

    Raises :class:`WindowTruncationError` when the build reaches ``max_events``,
    because the dataset would then cover only part of the window.
    """
    if max_events <= 0:
        raise WindowSelectionError("max_events must be positive")

    ground_truth = RedteamGroundTruth.from_file(redteam_path)
    if spec is None:
        spec = select_densest_redteam_window(
            ground_truth,
            span_seconds=span_seconds,
            context_seconds=context_seconds,
        )

    statistics = WindowStatistics()
    in_range = redteam_records_in_window(ground_truth, spec)
    unique_keys = {
        (r.timestamp, r.user, r.source_host, r.destination_host) for r in in_range
    }

    dataset = MLDataset.from_event_source(
        event_source_factory=lambda: iter_window_events(
            auth_path, spec, statistics=statistics
        ),
        ground_truth=ground_truth,
        split_config=split_config,
        recent_window_seconds=recent_window_seconds,
        max_events=max_events,
        retain_samples=retain_samples,
        timestamp_range=spec.timestamp_range(),
        emit_from_timestamp=spec.emit_start,
    )

    # Equality counts as truncation: a stream that stopped exactly at the
    # ceiling cannot be shown to have covered the whole window.
    streamed = dataset.events_processed + dataset.context_events_processed
    if streamed >= max_events:
        raise WindowTruncationError(
            f"window build reached the {max_events}-event ceiling after "
            f"{dataset.context_events_processed} context and "
            f"{dataset.events_processed} emitted events, so it would cover only "
            f"part of the window [{spec.emit_start:.0f}, {spec.emit_end:.0f}]. "
            "Raise max_events to cover the whole window."
        )

    return WindowDataset(
        spec=spec,
        dataset=dataset,
        statistics=statistics,
        redteam_records_in_range=len(in_range),
        redteam_unique_keys_in_range=len(unique_keys),
    )
