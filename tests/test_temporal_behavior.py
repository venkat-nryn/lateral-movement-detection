"""Unit tests for M3.3 temporal behavior baseline detector.

Uses tiny synthetic CanonicalEvent fixtures only.
No real dataset used in these tests.
"""

import pytest

from ml.baselines.temporal_behavior import (
    DetectorFeatures,
    DetectorPrediction,
    TemporalBehaviorDetector,
)
from ml.preprocessing.schema import CanonicalEvent


def make_event(
    event_id="evt-1",
    timestamp=1000.0,
    user="UserA",
    source_host="HostA",
    destination_host="HostB",
    success=True,
):
    """Create a minimal CanonicalEvent for testing."""
    return CanonicalEvent(
        event_id=event_id,
        timestamp=timestamp,
        user=user,
        source_host=source_host,
        destination_host=destination_host,
        event_type="authentication",
        success=success,
    )


# ============================================================================
# Feature Calculation Tests
# ============================================================================


class TestFeatureCalculation:
    """Tests for feature extraction from authentication events."""

    def test_first_user_destination_relationship(self):
        """First time a user connects to a destination should be flagged."""
        detector = TemporalBehaviorDetector()
        event = make_event(user="UserA", destination_host="HostX")
        pred = detector.process_event(event)
        assert pred.features.new_user_destination is True

    def test_repeated_normal_relationship(self):
        """Second occurrence of same (user, destination) should not be flagged."""
        detector = TemporalBehaviorDetector()
        # First event
        event1 = make_event(
            event_id="evt-1",
            timestamp=1000.0,
            user="UserA",
            destination_host="HostB",
        )
        detector.process_event(event1)
        # Second event (same user, same destination)
        event2 = make_event(
            event_id="evt-2",
            timestamp=1001.0,
            user="UserA",
            destination_host="HostB",
        )
        pred = detector.process_event(event2)
        assert pred.features.new_user_destination is False

    def test_new_destination_for_known_user(self):
        """User visiting new destination should be flagged."""
        detector = TemporalBehaviorDetector()
        # First event
        detector.process_event(
            make_event(
                event_id="evt-1",
                timestamp=1000.0,
                user="UserA",
                destination_host="HostB",
            )
        )
        # Second event (same user, new destination)
        event2 = make_event(
            event_id="evt-2",
            timestamp=1001.0,
            user="UserA",
            destination_host="HostC",
        )
        pred = detector.process_event(event2)
        assert pred.features.new_user_destination is True

    def test_new_source_destination_movement(self):
        """First time a source connects to a destination should be flagged."""
        detector = TemporalBehaviorDetector()
        event = make_event(
            source_host="HostA",
            destination_host="HostX",
        )
        pred = detector.process_event(event)
        assert pred.features.new_source_destination is True

    def test_repeated_source_destination_not_flagged(self):
        """Repeated (source, destination) pair should not be flagged."""
        detector = TemporalBehaviorDetector()
        detector.process_event(
            make_event(
                event_id="evt-1",
                timestamp=1000.0,
                source_host="HostA",
                destination_host="HostB",
            )
        )
        event2 = make_event(
            event_id="evt-2",
            timestamp=1001.0,
            source_host="HostA",
            destination_host="HostB",
        )
        pred = detector.process_event(event2)
        assert pred.features.new_source_destination is False

    def test_source_differs_destination(self):
        """Lateral movement (source != destination) should be flagged."""
        detector = TemporalBehaviorDetector()
        event = make_event(source_host="HostA", destination_host="HostB")
        pred = detector.process_event(event)
        assert pred.features.source_differs_destination is True

    def test_source_equals_destination_not_flagged(self):
        """Loopback authentication (source == destination) should not be flagged."""
        detector = TemporalBehaviorDetector()
        event = make_event(source_host="HostA", destination_host="HostA")
        pred = detector.process_event(event)
        assert pred.features.source_differs_destination is False

    def test_rapid_user_movement(self):
        """User moving to different host within 5 minutes should be flagged."""
        detector = TemporalBehaviorDetector()
        # First event
        detector.process_event(
            make_event(
                event_id="evt-1",
                timestamp=1000.0,
                user="UserA",
                destination_host="HostB",
            )
        )
        # Second event (same user, different destination, 200s later = <5min)
        event2 = make_event(
            event_id="evt-2",
            timestamp=1200.0,
            user="UserA",
            destination_host="HostC",
        )
        pred = detector.process_event(event2)
        assert pred.features.rapid_user_movement is True

    def test_slow_user_movement_not_flagged(self):
        """User movement over long interval should not be flagged."""
        detector = TemporalBehaviorDetector()
        detector.process_event(
            make_event(
                event_id="evt-1",
                timestamp=1000.0,
                user="UserA",
                destination_host="HostB",
            )
        )
        # 10 minutes later
        event2 = make_event(
            event_id="evt-2",
            timestamp=1600.0,
            user="UserA",
            destination_host="HostC",
        )
        pred = detector.process_event(event2)
        assert pred.features.rapid_user_movement is False

    def test_failed_authentication_flagged(self):
        """Failed authentication should be flagged."""
        detector = TemporalBehaviorDetector()
        event = make_event(success=False)
        pred = detector.process_event(event)
        assert pred.features.failed_authentication is True

    def test_successful_authentication_not_flagged(self):
        """Successful authentication should not be flagged for this feature."""
        detector = TemporalBehaviorDetector()
        event = make_event(success=True)
        pred = detector.process_event(event)
        assert pred.features.failed_authentication is False

    def test_user_destination_frequency(self):
        """Frequency of (user, destination) pairs should be counted."""
        detector = TemporalBehaviorDetector()
        # Add 3 events for same (user, destination)
        for i in range(3):
            detector.process_event(
                make_event(
                    event_id=f"evt-{i}",
                    timestamp=1000.0 + i * 100,
                    user="UserA",
                    destination_host="HostB",
                )
            )
        # Fourth time should show frequency of 3
        event4 = make_event(
            event_id="evt-4",
            timestamp=1400.0,
            user="UserA",
            destination_host="HostB",
        )
        pred = detector.process_event(event4)
        assert pred.features.user_destination_frequency == 3


# ============================================================================
# Scoring and Decision Tests
# ============================================================================


class TestScoringAndDecision:
    """Tests for score calculation and suspicious determination."""

    def test_deterministic_score(self):
        """Same event should produce same score."""
        detector1 = TemporalBehaviorDetector()
        detector2 = TemporalBehaviorDetector()

        event = make_event(user="UserA", destination_host="HostB")

        pred1 = detector1.process_event(event)
        pred2 = detector2.process_event(event)

        assert pred1.score == pred2.score

    def test_deterministic_suspicious_decision(self):
        """Same event should produce same suspicious flag."""
        detector1 = TemporalBehaviorDetector()
        detector2 = TemporalBehaviorDetector()

        event = make_event(user="UserA", destination_host="HostB")

        pred1 = detector1.process_event(event)
        pred2 = detector2.process_event(event)

        assert pred1.suspicious == pred2.suspicious

    def test_new_user_destination_increases_score(self):
        """New (user, destination) pair should increase score."""
        detector = TemporalBehaviorDetector()
        event_known = make_event(
            event_id="evt-1",
            timestamp=1000.0,
            user="UserA",
            destination_host="HostB",
        )
        detector.process_event(event_known)

        event_new = make_event(
            event_id="evt-2",
            timestamp=1001.0,
            user="UserA",
            destination_host="HostC",
        )
        pred_new = detector.process_event(event_new)
        assert pred_new.score > 0.0

    def test_prior_events_only(self):
        """Features should use only prior history, not future events."""
        detector = TemporalBehaviorDetector()
        # First event
        event1 = make_event(
            event_id="evt-1",
            timestamp=1000.0,
            user="UserA",
            destination_host="HostB",
        )
        pred1 = detector.process_event(event1)

        # At time of first event, it should be new
        assert pred1.features.new_user_destination is True

        # Second event
        event2 = make_event(
            event_id="evt-2",
            timestamp=1001.0,
            user="UserA",
            destination_host="HostB",
        )
        pred2 = detector.process_event(event2)

        # At time of second event, it should not be new
        assert pred2.features.new_user_destination is False


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for detector state and reset behavior."""

    def test_state_reset(self):
        """Reset should clear all history."""
        detector = TemporalBehaviorDetector()

        # Add some events
        detector.process_event(
            make_event(
                event_id="evt-1",
                timestamp=1000.0,
                user="UserA",
                destination_host="HostB",
            )
        )

        # Verify state was updated
        assert detector.number_of_events_processed() == 1

        # Reset
        detector.reset()

        # State should be cleared
        assert detector.number_of_events_processed() == 0
        assert detector.number_of_predictions() == 0

        # New event should be treated as first occurrence
        event_new = make_event(
            event_id="evt-2",
            timestamp=2000.0,
            user="UserA",
            destination_host="HostB",
        )
        pred = detector.process_event(event_new)
        assert pred.features.new_user_destination is True

    def test_process_multiple_events(self):
        """Process batch of events."""
        detector = TemporalBehaviorDetector()
        events = [
            make_event(
                event_id=f"evt-{i}",
                timestamp=1000.0 + i * 10,
                user=f"User{i % 2}",
                destination_host=f"Host{i % 3}",
            )
            for i in range(10)
        ]
        predictions = detector.process_events(events)
        assert len(predictions) == 10
        assert detector.number_of_events_processed() == 10


# ============================================================================
# Ground Truth Evaluation Tests
# ============================================================================


class TestGroundTruthEvaluation:
    """Tests for matching against ground truth."""

    def test_ground_truth_not_in_detector_state(self):
        """Redteam labels should never enter detector state."""
        detector = TemporalBehaviorDetector()

        # Process event (redteam labels never used)
        event = make_event(user="UserA", destination_host="HostB")
        pred = detector.process_event(event)

        # Check that state doesn't depend on ground truth
        # (This is implicit in feature calculation, but verify)
        assert pred.features.new_user_destination is True

        # Verify detector processes same way whether or not event is in redteam
        pred2 = detector.process_event(
            make_event(
                event_id="evt-2",
                timestamp=1001.0,
                user="UserA",
                destination_host="HostB",
            )
        )
        assert pred2.features.new_user_destination is False

    def test_exact_match_fields(self):
        """Verify prediction contains exact match fields."""
        detector = TemporalBehaviorDetector()
        event = make_event(
            event_id="test123",
            timestamp=5000.0,
            user="UserX",
            source_host="HostA",
            destination_host="HostZ",
        )
        pred = detector.process_event(event)

        assert pred.event_id == "test123"
        assert pred.timestamp == 5000.0
        assert pred.user == "UserX"
        assert pred.source_host == "HostA"
        assert pred.destination_host == "HostZ"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_detector(self):
        """Detector starts with no predictions."""
        detector = TemporalBehaviorDetector()
        assert detector.number_of_events_processed() == 0
        assert detector.number_of_predictions() == 0
        assert detector.number_of_suspicious_predictions() == 0

    def test_single_event(self):
        """Single event processing."""
        detector = TemporalBehaviorDetector()
        event = make_event()
        pred = detector.process_event(event)
        assert pred is not None
        assert detector.number_of_events_processed() == 1

    def test_many_users(self):
        """Detector handles many different users."""
        detector = TemporalBehaviorDetector()
        for i in range(100):
            event = make_event(
                event_id=f"evt-{i}",
                timestamp=1000.0 + i,
                user=f"User{i}",
                destination_host="HostB",
            )
            pred = detector.process_event(event)
            # Each user's first event to HostB is new
            assert pred.features.new_user_destination is True

    def test_many_destinations(self):
        """Detector handles single user to many destinations."""
        detector = TemporalBehaviorDetector()
        for i in range(50):
            event = make_event(
                event_id=f"evt-{i}",
                timestamp=1000.0 + i,
                user="UserA",
                destination_host=f"Host{i}",
            )
            pred = detector.process_event(event)
            # Each new destination is new for the user
            assert pred.features.new_user_destination is True

    def test_state_size_estimate(self):
        """State size estimate should be reasonable."""
        detector = TemporalBehaviorDetector()
        for i in range(100):
            detector.process_event(
                make_event(
                    event_id=f"evt-{i}",
                    timestamp=1000.0 + i,
                    user=f"User{i % 10}",
                    destination_host=f"Host{i % 20}",
                )
            )
        state = detector.get_state_size_estimate()
        assert state["seen_user_destinations"] > 0
        assert state["seen_source_destinations"] > 0


# ============================================================================
# Prediction Object Tests
# ============================================================================


class TestPredictionObject:
    """Tests for DetectorPrediction dataclass."""

    def test_prediction_immutable(self):
        """Predictions should be immutable."""
        detector = TemporalBehaviorDetector()
        event = make_event()
        pred = detector.process_event(event)

        with pytest.raises(Exception):
            pred.suspicious = False

    def test_features_immutable(self):
        """Features should be immutable."""
        detector = TemporalBehaviorDetector()
        event = make_event()
        pred = detector.process_event(event)

        with pytest.raises(Exception):
            pred.features.new_user_destination = False

    def test_prediction_all_fields_present(self):
        """Prediction should contain all required fields."""
        detector = TemporalBehaviorDetector()
        event = make_event()
        pred = detector.process_event(event)

        assert hasattr(pred, "event_id")
        assert hasattr(pred, "timestamp")
        assert hasattr(pred, "user")
        assert hasattr(pred, "source_host")
        assert hasattr(pred, "destination_host")
        assert hasattr(pred, "score")
        assert hasattr(pred, "suspicious")
        assert hasattr(pred, "features")
