"""Temporal/behavioral baseline detector for lateral movement (Module M3.3).

A streaming, interpretable baseline that detects unusual authentication
patterns using only information available before the current event.

Features are calculated from authentication history maintained during
streaming. Scoring is deterministic and does not use redteam labels.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ml.preprocessing.schema import CanonicalEvent

# ============================================================================
# Configuration: Explicit scoring weights and thresholds
# ============================================================================

# Time window for "recent" user activity (in seconds)
RECENT_WINDOW_SECONDS = 86400  # 1 day

# Suspicious score threshold
SUSPICIOUS_THRESHOLD = 0.50

# Feature weights (all between 0-1, controls contribution to final score)
WEIGHT_NEW_USER_DESTINATION = 0.25
WEIGHT_NEW_SOURCE_DESTINATION = 0.20
WEIGHT_SOURCE_DIFFERS_DESTINATION = 0.05
WEIGHT_RAPID_USER_MOVEMENT = 0.20
WEIGHT_RECENT_DESTINATION_COUNT = 0.10
WEIGHT_FAILED_AUTHENTICATION = 0.15
WEIGHT_USER_DESTINATION_FREQUENCY = 0.05


@dataclass(frozen=True)
class DetectorFeatures:
    """Calculated features for a single event."""

    new_user_destination: bool
    new_source_destination: bool
    source_differs_destination: bool
    rapid_user_movement: bool
    recent_user_destination_count: int
    failed_authentication: bool
    user_destination_frequency: int


@dataclass(frozen=True)
class DetectorPrediction:
    """Prediction result for a single event."""

    event_id: str
    timestamp: float
    user: str
    source_host: str
    destination_host: str
    score: float
    suspicious: bool
    features: DetectorFeatures


class TemporalBehaviorDetector:
    """Streaming detector for authentication anomalies.

    Processes events sequentially, maintaining minimal state of prior
    authentication history. Never uses redteam labels or future events.
    """

    def __init__(
        self,
        suspicious_threshold: float = SUSPICIOUS_THRESHOLD,
        recent_window_seconds: float = RECENT_WINDOW_SECONDS,
    ):
        """Initialize detector.

        Args:
            suspicious_threshold: Score threshold for suspicious flag (0-1).
            recent_window_seconds: Time window for "recent" activity.
        """
        self.suspicious_threshold = suspicious_threshold
        self.recent_window_seconds = recent_window_seconds

        # State: authentication history (all prior events)
        self._seen_user_destinations: set[tuple[str, str]] = set()
        self._seen_source_destinations: set[tuple[str, str]] = set()

        # State: recent user activity (timestamp, destination pairs per user)
        self._user_recent_events: dict[str, deque] = {}

        # State: frequency counts (user, destination) -> count
        self._user_destination_frequency: dict[tuple[str, str], int] = {}

        # Statistics
        self._events_processed = 0
        self._suspicious_predictions_count = 0

    def _calculate_features(self, event: CanonicalEvent) -> DetectorFeatures:
        """Calculate feature vector for event using prior history only."""
        user = event.user
        src = event.source_host
        dst = event.destination_host
        timestamp = event.timestamp

        # Feature 1: Has this user ever authenticated to this destination?
        new_user_destination = (user, dst) not in self._seen_user_destinations

        # Feature 2: Has this source host connected to this destination?
        new_source_destination = (src, dst) not in self._seen_source_destinations

        # Feature 3: Source differs from destination (lateral movement)?
        source_differs_destination = src != dst

        # Feature 4: Rapid user movement (moved within recent window)?
        rapid_user_movement = False
        if user in self._user_recent_events:
            recent_events = self._user_recent_events[user]
            if len(recent_events) > 0:
                last_timestamp, last_dst = recent_events[-1]
                time_since_last = timestamp - last_timestamp
                if time_since_last < 300 and last_dst != dst:  # < 5 minutes to different dest
                    rapid_user_movement = True

        # Feature 5: Count of distinct destinations for user in recent window
        recent_user_destination_count = 0
        if user in self._user_recent_events:
            recent_dests = set()
            for evt_time, evt_dst in self._user_recent_events[user]:
                if timestamp - evt_time <= self.recent_window_seconds:
                    recent_dests.add(evt_dst)
            recent_user_destination_count = len(recent_dests)

        # Feature 6: Did this authentication fail?
        failed_authentication = not event.success

        # Feature 7: Frequency of previous (user, destination) interactions
        user_destination_frequency = self._user_destination_frequency.get(
            (user, dst), 0
        )

        return DetectorFeatures(
            new_user_destination=new_user_destination,
            new_source_destination=new_source_destination,
            source_differs_destination=source_differs_destination,
            rapid_user_movement=rapid_user_movement,
            recent_user_destination_count=recent_user_destination_count,
            failed_authentication=failed_authentication,
            user_destination_frequency=user_destination_frequency,
        )

    def _calculate_score(self, features: DetectorFeatures) -> float:
        """Calculate deterministic suspicion score (0-1).

        Weighted sum of feature contributions. Uses only features, not labels.
        """
        score = 0.0

        if features.new_user_destination:
            score += WEIGHT_NEW_USER_DESTINATION

        if features.new_source_destination:
            score += WEIGHT_NEW_SOURCE_DESTINATION

        if features.source_differs_destination:
            score += WEIGHT_SOURCE_DIFFERS_DESTINATION

        if features.rapid_user_movement:
            score += WEIGHT_RAPID_USER_MOVEMENT

        # Normalize recent destination count to 0-1
        normalized_dest_count = min(
            features.recent_user_destination_count / 50.0, 1.0
        )
        score += WEIGHT_RECENT_DESTINATION_COUNT * normalized_dest_count

        if features.failed_authentication:
            score += WEIGHT_FAILED_AUTHENTICATION

        # Normalize frequency to 0-1 (cap at 100)
        normalized_frequency = min(features.user_destination_frequency / 100.0, 1.0)
        score += WEIGHT_USER_DESTINATION_FREQUENCY * normalized_frequency

        # Clamp to [0, 1]
        return min(max(score, 0.0), 1.0)

    def process_event(self, event: CanonicalEvent) -> DetectorPrediction:
        """Process one event and return prediction.

        Uses only prior history (state before this event).
        Updates state after prediction.

        Args:
            event: The event to process.

        Returns:
            DetectorPrediction with score, features, and decision.
        """
        # Calculate features using current state (before update)
        features = self._calculate_features(event)

        # Calculate score
        score = self._calculate_score(features)

        # Make prediction
        suspicious = score >= self.suspicious_threshold

        # Create prediction
        prediction = DetectorPrediction(
            event_id=event.event_id,
            timestamp=event.timestamp,
            user=event.user,
            source_host=event.source_host,
            destination_host=event.destination_host,
            score=score,
            suspicious=suspicious,
            features=features,
        )

        # Update state for next event (never use redteam labels)
        self._seen_user_destinations.add((event.user, event.destination_host))
        self._seen_source_destinations.add((event.source_host, event.destination_host))

        # Track recent user events
        if event.user not in self._user_recent_events:
            self._user_recent_events[event.user] = deque(maxlen=1000)
        self._user_recent_events[event.user].append(
            (event.timestamp, event.destination_host)
        )

        # Update frequency
        key = (event.user, event.destination_host)
        self._user_destination_frequency[key] = (
            self._user_destination_frequency.get(key, 0) + 1
        )

        # Record prediction and update count
        if suspicious:
            self._suspicious_predictions_count += 1
        self._events_processed += 1

        return prediction

    def process_events(self, events) -> list[DetectorPrediction]:
        """Process multiple events and return predictions.

        Args:
            events: Iterable of CanonicalEvent objects.

        Returns:
            List of DetectorPrediction objects (also stored internally).
        """
        predictions = []
        for event in events:
            prediction = self.process_event(event)
            predictions.append(prediction)
        return predictions

    def reset(self) -> None:
        """Reset detector state (clear history)."""
        self._seen_user_destinations.clear()
        self._seen_source_destinations.clear()
        self._user_recent_events.clear()
        self._user_destination_frequency.clear()
        self._events_processed = 0
        self._suspicious_predictions_count = 0

    def number_of_events_processed(self) -> int:
        """Total events processed since initialization or last reset."""
        return self._events_processed

    def number_of_predictions(self) -> int:
        """Total predictions generated."""
        return self._events_processed

    def number_of_suspicious_predictions(self) -> int:
        """Count of predictions flagged as suspicious."""
        return self._suspicious_predictions_count

    def get_predictions(self) -> list[DetectorPrediction]:
        """Return all predictions generated so far."""
        raise NotImplementedError("Memory bounded streaming restricts retrieving all predictions.")

    def get_state_size_estimate(self) -> dict[str, int]:
        """Estimate memory usage of detector state."""
        return {
            "seen_user_destinations": len(self._seen_user_destinations),
            "seen_source_destinations": len(self._seen_source_destinations),
            "user_recent_events_keys": len(self._user_recent_events),
            "user_recent_events_items": sum(len(d) for d in self._user_recent_events.values()),
            "user_destination_frequency": len(self._user_destination_frequency),
        }
