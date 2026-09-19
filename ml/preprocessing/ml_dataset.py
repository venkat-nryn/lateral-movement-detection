"""Chronological ML dataset preparation for real LANL authentication data (M3.5).

Scope
-----
This module is **dataset preparation only**. It produces deterministic,
chronologically split, labeled feature samples that are trivially convertible
to ``X`` (numeric matrix) / ``y`` (binary labels). It trains nothing.

Research rules enforced here
----------------------------
* **Chronological split only.** Splits are a pure function of the event
  timestamp; there is no shuffling, no randomness and no RNG seed anywhere in
  this module. Train precedes validation precedes test by construction.
* **No future information.** Feature extraction is delegated to
  :class:`~ml.preprocessing.features.TemporalGraphFeatureExtractor`, which only
  exposes events with a timestamp *strictly* earlier than the current event.
  Same-timestamp events are held in a pending batch and therefore cannot see
  each other.
* **Labels come from redteam.txt.gz only**, via the exact 4-field match
  (timestamp, user, source_host, destination_host) defined by
  :func:`ml.evaluation.redteam.matches_canonical_event`. Redteam information is
  never fed into feature extraction, and labels are never resampled, balanced
  or synthesised.

Memory design
-------------
Full-dataset processing is **not** the default and cannot happen by accident:

* :meth:`MLDataset.from_lanl_files` defaults to
  :data:`DEVELOPMENT_MAX_EVENTS` (1,000,000 events) and raises
  :class:`FullDatasetGuardError` unless the caller explicitly passes
  ``allow_full_dataset=True``.
* Retained feature rows are stored in flat :mod:`array` buffers
  (``float64`` features, ``float64`` timestamps, ``int8`` labels) instead of
  one Python tuple per row, which keeps a bounded development run at roughly
  ``17 * 8`` bytes of feature payload per sample.
* ``retain_samples=False`` builds a counts-only dataset, streaming every event
  through the extractor while retaining **no** feature rows at all. That is the
  intended mode for the later controlled full-dataset research run.
* :func:`iter_chronological_samples` exposes the same pipeline as a pure
  generator with zero retention for callers that want to consume samples
  incrementally.

Nothing in this module decompresses ``auth.txt.gz`` to disk or writes any
intermediate dataset file.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from ml.evaluation.redteam import RedteamGroundTruth
from ml.preprocessing.features import (
    DEFAULT_RECENT_WINDOW_SECONDS,
    EventFeatures,
    TemporalGraphFeatureExtractor,
)
from ml.preprocessing.lanl_adapter import iter_lanl_events
from ml.preprocessing.schema import CanonicalEvent

NumericFeature = int | float
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

#: Maximum number of real LANL authentication events a development run may
#: process without an explicit opt-in. The complete dataset must only ever be
#: processed by a deliberate, controlled research run.
DEVELOPMENT_MAX_EVENTS = 1_000_000

#: Exact-match label key: (timestamp, user, source_host, destination_host).
LabelKey = tuple[float, str, str, str]


class FullDatasetGuardError(RuntimeError):
    """Raised when a call would process more than the development bound.

    The guard exists so that an accidental ``max_events=None`` cannot stream the
    complete 7.6 GB LANL authentication file and retain tens of millions of
    feature rows in RAM.
    """


@dataclass(frozen=True, slots=True)
class ChronologicalSplitConfig:
    """Timestamp-based split configuration.

    Fractions are fractions of the observed **timestamp span**, not of the event
    count. This keeps the split a pure function of the timestamp, so every event
    sharing a timestamp is guaranteed to land in the same split and no
    same-timestamp group can straddle a split boundary.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(fraction < 0 for fraction in fractions):
            raise ValueError("split fractions must be non-negative")
        total = sum(fractions)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")

    def train_end_timestamp(
        self, min_timestamp: float, max_timestamp: float
    ) -> float:
        return min_timestamp + (
            (max_timestamp - min_timestamp) * self.train_fraction
        )

    def validation_end_timestamp(
        self, min_timestamp: float, max_timestamp: float
    ) -> float:
        return min_timestamp + (
            (max_timestamp - min_timestamp)
            * (self.train_fraction + self.validation_fraction)
        )

    def split_for_timestamp(
        self, timestamp: float, min_timestamp: float, max_timestamp: float
    ) -> str:
        if max_timestamp <= min_timestamp:
            return SPLIT_TRAIN

        if timestamp <= self.train_end_timestamp(min_timestamp, max_timestamp):
            return SPLIT_TRAIN
        if timestamp <= self.validation_end_timestamp(
            min_timestamp, max_timestamp
        ):
            return SPLIT_VALIDATION
        return SPLIT_TEST


@dataclass(frozen=True, slots=True)
class LabeledFeatureSample:
    """Single labeled sample with a fixed-width numeric feature vector."""

    event_id: str
    timestamp: float
    features: tuple[NumericFeature, ...]
    label: int

    def to_dict(self, feature_names: tuple[str, ...]) -> dict[str, object]:
        record: dict[str, object] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "label": self.label,
        }
        for name, value in zip(feature_names, self.features):
            record[name] = value
        return record


@dataclass(frozen=True, slots=True)
class TimestampRange:
    """Observed timestamp range for the bounded event prefix."""

    start: float
    end: float


@dataclass(slots=True)
class DatasetSplit:
    """Samples for one chronological split.

    Feature rows are kept in a single flat ``array("d")`` buffer of
    ``len(split) * feature_dimension`` values rather than one Python tuple per
    row. Timestamps use ``array("d")`` and labels ``array("b")``.

    When ``retain`` is ``False`` the split accumulates counts and the timestamp
    range only; no per-sample data is stored. That mode makes it safe to stream
    an arbitrarily large event source without growing memory.
    """

    name: str
    feature_names: tuple[str, ...]
    retain: bool = True
    event_ids: list[str] = field(default_factory=list)
    timestamps: array = field(default_factory=lambda: array("d"))
    labels: array = field(default_factory=lambda: array("b"))
    _feature_values: array = field(default_factory=lambda: array("d"))
    _sample_count: int = 0
    _positive_count: int = 0
    _min_timestamp: float | None = None
    _max_timestamp: float | None = None

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_names)

    def add_sample(
        self,
        event_id: str,
        timestamp: float,
        features: EventFeatures,
        label: int,
    ) -> None:
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")

        self._sample_count += 1
        self._positive_count += label
        if self._min_timestamp is None or timestamp < self._min_timestamp:
            self._min_timestamp = timestamp
        if self._max_timestamp is None or timestamp > self._max_timestamp:
            self._max_timestamp = timestamp

        if not self.retain:
            return

        self.event_ids.append(event_id)
        self.timestamps.append(float(timestamp))
        self.labels.append(label)
        self._feature_values.extend(features.to_vector())

    def __len__(self) -> int:
        return self._sample_count

    def positive_count(self) -> int:
        return self._positive_count

    def negative_count(self) -> int:
        return self._sample_count - self._positive_count

    def timestamp_range(self) -> TimestampRange | None:
        if self._min_timestamp is None or self._max_timestamp is None:
            return None
        return TimestampRange(start=self._min_timestamp, end=self._max_timestamp)

    def iter_feature_rows(self) -> Iterator[tuple[NumericFeature, ...]]:
        """Yield feature rows lazily without materializing the whole matrix."""
        width = self.feature_dimension
        values = self._feature_values
        for offset in range(0, len(values), width):
            yield tuple(values[offset : offset + width])

    @property
    def feature_rows(self) -> list[tuple[NumericFeature, ...]]:
        """Materialize all feature rows (development-scale convenience)."""
        return list(self.iter_feature_rows())

    def feature_buffer(self) -> array:
        """Flat row-major ``float64`` feature buffer.

        Converts to a matrix without adding any dependency to this module::

            import numpy as np
            X = np.frombuffer(split.feature_buffer(), dtype=np.float64)
            X = X.reshape(len(split), split.feature_dimension)
        """
        return self._feature_values

    def iter_samples(self) -> Iterator[LabeledFeatureSample]:
        for event_id, timestamp, features, label in zip(
            self.event_ids,
            self.timestamps,
            self.iter_feature_rows(),
            self.labels,
        ):
            yield LabeledFeatureSample(
                event_id=event_id,
                timestamp=timestamp,
                features=features,
                label=int(label),
            )

    def X_y(self) -> tuple[tuple[tuple[NumericFeature, ...], ...], tuple[int, ...]]:
        """Return ``(X, y)`` as plain Python tuples (development-scale)."""
        return (
            tuple(self.iter_feature_rows()),
            tuple(int(label) for label in self.labels),
        )


def build_label_index(ground_truth: RedteamGroundTruth) -> frozenset[LabelKey]:
    """Build the exact-match redteam label index.

    A sample is positive only when ``(timestamp, user, source_host,
    destination_host)`` matches a redteam record exactly -- the same definition
    as :func:`ml.evaluation.redteam.matches_canonical_event`.
    """
    return frozenset(
        (
            record.timestamp,
            record.user,
            record.source_host,
            record.destination_host,
        )
        for record in ground_truth.iter_records()
    )


def label_for_event(event: CanonicalEvent, label_index: frozenset[LabelKey]) -> int:
    """Return the binary ground-truth label for ``event`` (exact match only)."""
    return int(
        (
            event.timestamp,
            event.user,
            event.source_host,
            event.destination_host,
        )
        in label_index
    )


def iter_chronological_samples(
    events: Iterable[CanonicalEvent],
    label_index: frozenset[LabelKey],
    *,
    timestamp_range: TimestampRange,
    split_config: ChronologicalSplitConfig | None = None,
    recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
    max_events: int | None = None,
) -> Iterator[tuple[str, str, float, EventFeatures, int]]:
    """Stream ``(split_name, event_id, timestamp, features, label)`` tuples.

    Retains nothing beyond the feature extractor bounded sliding state, so it is
    safe for arbitrarily long event sources. ``timestamp_range`` must be
    supplied because chronological split boundaries are defined against the full
    observed span.
    """
    config = split_config or ChronologicalSplitConfig()
    extractor = TemporalGraphFeatureExtractor(
        recent_window_seconds=recent_window_seconds
    )

    for index, event in enumerate(events):
        if max_events is not None and index >= max_events:
            break
        features = extractor.process_event(event)
        label = label_for_event(event, label_index)
        split_name = config.split_for_timestamp(
            event.timestamp, timestamp_range.start, timestamp_range.end
        )
        yield split_name, event.event_id, event.timestamp, features, label


class MLDataset:
    """Bounded chronological dataset built from real LANL auth events."""

    def __init__(
        self,
        *,
        splits: Mapping[str, DatasetSplit],
        feature_names: tuple[str, ...],
        split_config: ChronologicalSplitConfig,
        timestamp_range: TimestampRange | None,
        events_processed: int,
        max_events: int | None,
        extractor_state_size_estimate: Mapping[str, int],
        redteam_record_count: int = 0,
        redteam_timestamp_range: TimestampRange | None = None,
        retained_samples: bool = True,
        context_events_processed: int = 0,
    ) -> None:
        self._splits = dict(splits)
        self.feature_names = feature_names
        self.split_config = split_config
        self.timestamp_range = timestamp_range
        self.events_processed = events_processed
        self.max_events = max_events
        self.extractor_state_size_estimate = dict(extractor_state_size_estimate)
        self.redteam_record_count = redteam_record_count
        self.redteam_timestamp_range = redteam_timestamp_range
        self.retained_samples = retained_samples
        self.context_events_processed = context_events_processed

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_lanl_files(
        cls,
        auth_path: Path | str,
        redteam_path: Path | str,
        *,
        split_config: ChronologicalSplitConfig | None = None,
        recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
        max_events: int | None = DEVELOPMENT_MAX_EVENTS,
        allow_full_dataset: bool = False,
        retain_samples: bool = True,
        timestamp_range: TimestampRange | None = None,
    ) -> "MLDataset":
        """Build a dataset from the real LANL files.

        ``max_events`` defaults to :data:`DEVELOPMENT_MAX_EVENTS`. Processing
        more than that (including ``max_events=None``) requires an explicit
        ``allow_full_dataset=True``; otherwise :class:`FullDatasetGuardError` is
        raised before the authentication file is opened. For a genuine
        full-dataset run also pass ``retain_samples=False`` so that no feature
        rows are accumulated.
        """
        _check_development_bound(max_events, allow_full_dataset)

        ground_truth = RedteamGroundTruth.from_file(redteam_path)
        return cls.from_event_source(
            event_source_factory=lambda: iter_lanl_events(
                auth_path, limit=max_events
            ),
            ground_truth=ground_truth,
            split_config=split_config,
            recent_window_seconds=recent_window_seconds,
            max_events=max_events,
            retain_samples=retain_samples,
            timestamp_range=timestamp_range,
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[CanonicalEvent],
        ground_truth: RedteamGroundTruth,
        *,
        split_config: ChronologicalSplitConfig | None = None,
        recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
        max_events: int | None = None,
        retain_samples: bool = True,
    ) -> "MLDataset":
        """Build a dataset from an in-memory event sequence.

        Intended for unit tests and tiny fixtures: ``events`` is materialized so
        that the timestamp-range pass and the feature pass see the same data.
        Never pass a real LANL stream here -- use :meth:`from_lanl_files`.
        """
        event_list = list(events)
        return cls.from_event_source(
            event_source_factory=lambda: iter(event_list),
            ground_truth=ground_truth,
            split_config=split_config,
            recent_window_seconds=recent_window_seconds,
            max_events=max_events,
            retain_samples=retain_samples,
        )

    @classmethod
    def from_event_source(
        cls,
        *,
        event_source_factory: Callable[[], Iterable[CanonicalEvent]],
        ground_truth: RedteamGroundTruth,
        split_config: ChronologicalSplitConfig | None = None,
        recent_window_seconds: float = DEFAULT_RECENT_WINDOW_SECONDS,
        max_events: int | None = None,
        retain_samples: bool = True,
        timestamp_range: TimestampRange | None = None,
        emit_from_timestamp: float | None = None,
    ) -> "MLDataset":
        """Build a dataset from a re-iterable event source.

        Two bounded passes are performed over the source: the first discovers
        the timestamp range and validates non-decreasing order, the second
        extracts features and assigns splits. Neither pass loads the source into
        memory. Supplying ``timestamp_range`` skips the first pass.

        ``emit_from_timestamp`` marks the start of the emitted region. Events
        earlier than it are still fed through the feature extractor so that the
        first emitted sample sees realistic historical behaviour, but they
        produce no sample and are counted separately as context events. The
        extractor is never reset at that boundary.
        """
        config = split_config or ChronologicalSplitConfig()
        feature_names = EventFeatures.feature_names()

        if timestamp_range is None:
            observed_range, events_scanned = cls._scan_timestamp_range(
                event_source_factory=event_source_factory,
                max_events=max_events,
                emit_from_timestamp=emit_from_timestamp,
            )
        else:
            observed_range, events_scanned = timestamp_range, 0

        splits = {
            name: DatasetSplit(
                name=name, feature_names=feature_names, retain=retain_samples
            )
            for name in SPLIT_NAMES
        }
        label_index = build_label_index(ground_truth)
        extractor_state: Mapping[str, int] = {}
        events_processed = 0
        context_events = 0

        if observed_range is not None:
            extractor = TemporalGraphFeatureExtractor(
                recent_window_seconds=recent_window_seconds
            )
            for index, event in enumerate(event_source_factory()):
                if max_events is not None and index >= max_events:
                    break
                # Context events build history but are never emitted, and the
                # extractor state is deliberately carried across the boundary.
                features = extractor.process_event(event)
                if (
                    emit_from_timestamp is not None
                    and event.timestamp < emit_from_timestamp
                ):
                    context_events += 1
                    continue
                label = label_for_event(event, label_index)
                split_name = config.split_for_timestamp(
                    event.timestamp,
                    observed_range.start,
                    observed_range.end,
                )
                splits[split_name].add_sample(
                    event.event_id,
                    event.timestamp,
                    features,
                    label,
                )
                events_processed += 1
            extractor_state = extractor.get_state_size_estimate()

        return cls(
            splits=splits,
            feature_names=feature_names,
            split_config=config,
            timestamp_range=observed_range,
            events_processed=max(events_scanned, events_processed),
            max_events=max_events,
            extractor_state_size_estimate=extractor_state,
            redteam_record_count=ground_truth.number_of_records(),
            redteam_timestamp_range=_redteam_timestamp_range(ground_truth),
            retained_samples=retain_samples,
            context_events_processed=context_events,
        )

    @staticmethod
    def _scan_timestamp_range(
        *,
        event_source_factory: Callable[[], Iterable[CanonicalEvent]],
        max_events: int | None,
        emit_from_timestamp: float | None = None,
    ) -> tuple[TimestampRange | None, int]:
        first_timestamp: float | None = None
        last_timestamp: float | None = None
        previous_timestamp: float | None = None
        count = 0

        for index, event in enumerate(event_source_factory()):
            if max_events is not None and index >= max_events:
                break
            if previous_timestamp is not None and event.timestamp < previous_timestamp:
                raise ValueError(
                    "events must be provided in non-decreasing timestamp order"
                )
            previous_timestamp = event.timestamp
            # Split boundaries are defined over the emitted region only.
            if (
                emit_from_timestamp is not None
                and event.timestamp < emit_from_timestamp
            ):
                continue
            if first_timestamp is None:
                first_timestamp = event.timestamp
            last_timestamp = event.timestamp
            count += 1

        if first_timestamp is None or last_timestamp is None:
            return None, 0
        return TimestampRange(start=first_timestamp, end=last_timestamp), count

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get_split(self, split_name: str) -> DatasetSplit:
        return self._splits[split_name]

    def iter_split(self, split_name: str) -> Iterator[LabeledFeatureSample]:
        yield from self.get_split(split_name).iter_samples()

    def split_sample_counts(self) -> dict[str, int]:
        return {name: len(split) for name, split in self._splits.items()}

    def split_label_counts(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "positive": split.positive_count(),
                "negative": split.negative_count(),
            }
            for name, split in self._splits.items()
        }

    def feature_dimension(self) -> int:
        return len(self.feature_names)

    def total_positive_count(self) -> int:
        return sum(split.positive_count() for split in self._splits.values())

    def split_boundaries(self) -> dict[str, float] | None:
        """Timestamps at which train->validation and validation->test occur."""
        if self.timestamp_range is None:
            return None
        start, end = self.timestamp_range.start, self.timestamp_range.end
        return {
            "train_end": self.split_config.train_end_timestamp(start, end),
            "validation_end": self.split_config.validation_end_timestamp(
                start, end
            ),
        }

    def redteam_overlap(self) -> dict[str, object]:
        """Honest report of how the redteam timeline overlaps this prefix.

        The first portion of LANL authentication data occurs long before the
        redteam campaign starts, so a bounded development prefix can legitimately
        contain zero positive labels. Labels are never altered to compensate.
        """
        redteam_range = self.redteam_timestamp_range
        auth_range = self.timestamp_range
        overlap: dict[str, object] = {
            "redteam_record_count": self.redteam_record_count,
            "redteam_timestamp_min": redteam_range.start if redteam_range else None,
            "redteam_timestamp_max": redteam_range.end if redteam_range else None,
            "auth_timestamp_min": auth_range.start if auth_range else None,
            "auth_timestamp_max": auth_range.end if auth_range else None,
            "matched_positive_count": self.total_positive_count(),
        }
        overlap["timestamp_ranges_overlap"] = bool(
            auth_range is not None
            and redteam_range is not None
            and redteam_range.start <= auth_range.end
            and auth_range.start <= redteam_range.end
        )
        return overlap

    def summary(self) -> dict[str, object]:
        """Concise, machine-readable summary for terminal reporting."""
        return {
            "events_processed": self.events_processed,
            "context_events_processed": self.context_events_processed,
            "max_events": self.max_events,
            "retained_samples": self.retained_samples,
            "feature_dimension": self.feature_dimension(),
            "timestamp_min": (
                self.timestamp_range.start if self.timestamp_range else None
            ),
            "timestamp_max": (
                self.timestamp_range.end if self.timestamp_range else None
            ),
            "split_boundaries": self.split_boundaries(),
            "split_sample_counts": self.split_sample_counts(),
            "split_label_counts": self.split_label_counts(),
            "redteam_overlap": self.redteam_overlap(),
            "extractor_state_size_estimate": self.extractor_state_size_estimate,
        }


def _check_development_bound(
    max_events: int | None, allow_full_dataset: bool
) -> None:
    if allow_full_dataset:
        return
    if max_events is None:
        raise FullDatasetGuardError(
            "refusing to process the complete LANL authentication dataset: "
            "max_events=None requires allow_full_dataset=True (development "
            f"bound is {DEVELOPMENT_MAX_EVENTS} events)"
        )
    if max_events > DEVELOPMENT_MAX_EVENTS:
        raise FullDatasetGuardError(
            f"max_events={max_events} exceeds the development bound of "
            f"{DEVELOPMENT_MAX_EVENTS}; pass allow_full_dataset=True for a "
            "deliberate controlled research run"
        )


def _redteam_timestamp_range(
    ground_truth: RedteamGroundTruth,
) -> TimestampRange | None:
    minimum: float | None = None
    maximum: float | None = None
    for record in ground_truth.iter_records():
        if minimum is None or record.timestamp < minimum:
            minimum = record.timestamp
        if maximum is None or record.timestamp > maximum:
            maximum = record.timestamp
    if minimum is None or maximum is None:
        return None
    return TimestampRange(start=minimum, end=maximum)
