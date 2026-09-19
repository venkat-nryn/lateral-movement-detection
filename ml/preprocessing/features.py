"""Streaming temporal/graph feature extraction for canonical auth events.

This module builds fixed numeric feature vectors from real LANL-derived
``CanonicalEvent`` objects while enforcing strict temporal integrity:
features for an event at time ``T`` may only use events with timestamp
strictly less than ``T``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator

from ml.preprocessing.schema import CanonicalEvent

DEFAULT_RECENT_WINDOW_SECONDS = 3600.0
NO_PRIOR_TIME_DELTA = -1.0


@dataclass(frozen=True, slots=True)
class EventFeatures:
    """Immutable ML-ready feature result for one event."""

    event_id: str
    timestamp: float
    source_equals_destination: int
    user_destination_seen: int
    source_destination_seen: int
    user_destination_count: int
    source_destination_count: int
    user_unique_destination_count: int
    source_unique_destination_count: int
    destination_unique_source_count: int
    user_recent_event_count: int
    user_recent_unique_destination_count: int
    time_since_user_previous_event: float
    time_since_user_destination_event: float
    time_since_source_destination_event: float
    failed_authentication: int
    user_recent_failure_count: int
    destination_recent_event_count: int
    movement_edge: int

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        return tuple(field for field in cls.__dataclass_fields__ if field not in {"event_id", "timestamp"})

    def to_feature_dict(self) -> dict[str, int | float]:
        return {
            name: getattr(self, name)
            for name in self.feature_names()
        }

    def to_dict(self) -> dict[str, int | float | str]:
        record: dict[str, int | float | str] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }
        record.update(self.to_feature_dict())
        return record

    def to_vector(self) -> tuple[int | float, ...]:
        return tuple(getattr(self, name) for name in self.feature_names())


class TemporalGraphFeatureExtractor:
    """Streaming feature extractor with strict timestamp-before semantics."""

    def __init__(
        self, recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS
    ) -> None:
        if recent_window_seconds < 0:
            raise ValueError("recent_window_seconds must be non-negative")

        self.recent_window_seconds = float(recent_window_seconds)
        self.reset()

    def reset(self) -> None:
        """Clear all historical state."""
        self._user_destination_counts: dict[tuple[str, str], int] = {}
        self._source_destination_counts: dict[tuple[str, str], int] = {}
        self._user_destinations: dict[str, set[str]] = {}
        self._source_destinations: dict[str, set[str]] = {}
        self._destination_sources: dict[str, set[str]] = {}

        self._last_user_timestamp: dict[str, float] = {}
        self._last_user_destination_timestamp: dict[tuple[str, str], float] = {}
        self._last_source_destination_timestamp: dict[tuple[str, str], float] = {}

        # Single chronologically ordered index of every committed event still
        # inside the recent window, plus per-key aggregate counters derived from
        # it. Expiry walks only the events that actually fall out of the window,
        # which is amortized O(1) per event; the previous per-key layout forced
        # a scan of every tracked user and host on every single event.
        self._recent_window: Deque[tuple[float, str, str, bool]] = deque()
        self._user_recent_counts: dict[str, int] = {}
        self._user_recent_destination_counts: dict[str, Counter[str]] = {}
        self._user_recent_failure_counts: dict[str, int] = {}
        self._destination_recent_counts: dict[str, int] = {}

        self._pending_timestamp: float | None = None
        self._pending_events: list[CanonicalEvent] = []
        self._events_processed = 0

    def process_event(self, event: CanonicalEvent) -> EventFeatures:
        """Process one event using only events with strictly earlier timestamps."""
        self._advance_committed_history(event.timestamp)
        self._expire_recent_state(event.timestamp)
        features = self._build_features(event)
        self._queue_pending_event(event)
        self._events_processed += 1
        return features

    def process_events(
        self, events: Iterable[CanonicalEvent]
    ) -> Iterator[EventFeatures]:
        """Yield features for events without retaining vectors in memory."""
        for event in events:
            yield self.process_event(event)

    def feature_dimension(self) -> int:
        return len(EventFeatures.feature_names())

    def number_of_events_processed(self) -> int:
        return self._events_processed

    def get_state_size_estimate(self) -> dict[str, int]:
        return {
            "user_destination_counts": len(self._user_destination_counts),
            "source_destination_counts": len(self._source_destination_counts),
            "user_destination_edges": sum(
                len(destinations)
                for destinations in self._user_destinations.values()
            ),
            "source_destination_edges": sum(
                len(destinations)
                for destinations in self._source_destinations.values()
            ),
            "destination_source_edges": sum(
                len(sources)
                for sources in self._destination_sources.values()
            ),
            # Both counts equal the number of committed events still inside the
            # recent window; they are tracked by one shared chronological index.
            "user_recent_event_items": len(self._recent_window),
            "destination_recent_event_items": len(self._recent_window),
            "pending_events": len(self._pending_events),
        }

    def _build_features(self, event: CanonicalEvent) -> EventFeatures:
        user = event.user
        source = event.source_host
        destination = event.destination_host
        timestamp = event.timestamp

        user_destination_key = (user, destination)
        source_destination_key = (source, destination)

        user_recent_destination_counts = self._user_recent_destination_counts.get(
            user
        )

        return EventFeatures(
            event_id=event.event_id,
            timestamp=timestamp,
            source_equals_destination=int(source == destination),
            user_destination_seen=int(
                self._user_destination_counts.get(user_destination_key, 0) > 0
            ),
            source_destination_seen=int(
                self._source_destination_counts.get(source_destination_key, 0)
                > 0
            ),
            user_destination_count=self._user_destination_counts.get(
                user_destination_key, 0
            ),
            source_destination_count=self._source_destination_counts.get(
                source_destination_key, 0
            ),
            user_unique_destination_count=len(
                self._user_destinations.get(user, ())
            ),
            source_unique_destination_count=len(
                self._source_destinations.get(source, ())
            ),
            destination_unique_source_count=len(
                self._destination_sources.get(destination, ())
            ),
            user_recent_event_count=self._user_recent_counts.get(user, 0),
            user_recent_unique_destination_count=(
                len(user_recent_destination_counts)
                if user_recent_destination_counts is not None
                else 0
            ),
            time_since_user_previous_event=self._time_since(
                timestamp, self._last_user_timestamp.get(user)
            ),
            time_since_user_destination_event=self._time_since(
                timestamp,
                self._last_user_destination_timestamp.get(user_destination_key),
            ),
            time_since_source_destination_event=self._time_since(
                timestamp,
                self._last_source_destination_timestamp.get(
                    source_destination_key
                ),
            ),
            failed_authentication=int(not event.success),
            user_recent_failure_count=self._user_recent_failure_counts.get(
                user, 0
            ),
            destination_recent_event_count=self._destination_recent_counts.get(
                destination, 0
            ),
            movement_edge=int(source != destination),
        )

    def _advance_committed_history(self, timestamp: float) -> None:
        if self._pending_timestamp is None:
            return

        if timestamp < self._pending_timestamp:
            raise ValueError(
                "events must be processed in non-decreasing timestamp order"
            )

        if timestamp > self._pending_timestamp:
            for pending_event in self._pending_events:
                self._commit_event(pending_event)
            self._pending_events.clear()
            self._pending_timestamp = None

    def _queue_pending_event(self, event: CanonicalEvent) -> None:
        if self._pending_timestamp is None:
            self._pending_timestamp = event.timestamp
        elif event.timestamp != self._pending_timestamp:
            raise ValueError(
                "pending event timestamp mismatch; expected same-timestamp batch"
            )
        self._pending_events.append(event)

    def _commit_event(self, event: CanonicalEvent) -> None:
        user = event.user
        source = event.source_host
        destination = event.destination_host
        user_destination_key = (user, destination)
        source_destination_key = (source, destination)

        self._user_destination_counts[user_destination_key] = (
            self._user_destination_counts.get(user_destination_key, 0) + 1
        )
        self._source_destination_counts[source_destination_key] = (
            self._source_destination_counts.get(source_destination_key, 0) + 1
        )

        self._user_destinations.setdefault(user, set()).add(destination)
        self._source_destinations.setdefault(source, set()).add(destination)
        self._destination_sources.setdefault(destination, set()).add(source)

        self._last_user_timestamp[user] = event.timestamp
        self._last_user_destination_timestamp[user_destination_key] = (
            event.timestamp
        )
        self._last_source_destination_timestamp[source_destination_key] = (
            event.timestamp
        )

        was_failure = not event.success
        self._recent_window.append(
            (event.timestamp, user, destination, was_failure)
        )
        self._user_recent_counts[user] = (
            self._user_recent_counts.get(user, 0) + 1
        )
        destination_counts = self._user_recent_destination_counts.setdefault(
            user, Counter()
        )
        destination_counts[destination] += 1
        if was_failure:
            self._user_recent_failure_counts[user] = (
                self._user_recent_failure_counts.get(user, 0) + 1
            )
        else:
            self._user_recent_failure_counts.setdefault(user, 0)

        self._destination_recent_counts[destination] = (
            self._destination_recent_counts.get(destination, 0) + 1
        )

    def _expire_recent_state(self, timestamp: float) -> None:
        """Drop committed events that have fallen out of the recent window.

        Walks the shared chronological index from the front, so the cost is
        proportional to the number of events actually expiring rather than to
        the number of tracked users and hosts.
        """
        window_start = timestamp - self.recent_window_seconds
        recent_window = self._recent_window

        while recent_window and recent_window[0][0] <= window_start:
            _, user, destination, was_failure = recent_window.popleft()

            user_count = self._user_recent_counts[user] - 1
            destination_counts = self._user_recent_destination_counts[user]
            destination_counts[destination] -= 1
            if destination_counts[destination] <= 0:
                del destination_counts[destination]
            if was_failure:
                self._user_recent_failure_counts[user] -= 1

            if user_count <= 0:
                del self._user_recent_counts[user]
                del self._user_recent_destination_counts[user]
                self._user_recent_failure_counts.pop(user, None)
            else:
                self._user_recent_counts[user] = user_count

            destination_count = self._destination_recent_counts[destination] - 1
            if destination_count <= 0:
                del self._destination_recent_counts[destination]
            else:
                self._destination_recent_counts[destination] = destination_count

    @staticmethod
    def _time_since(current_timestamp: float, previous_timestamp: float | None) -> float:
        if previous_timestamp is None:
            return NO_PRIOR_TIME_DELTA
        return float(current_timestamp - previous_timestamp)
