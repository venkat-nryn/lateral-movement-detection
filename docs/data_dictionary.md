# Data Dictionary

This document has two parts:

1. **Canonical schema (M2.0)** — the dataset-independent internal project representation.
2. **Dataset-specific fields** — a template to be filled in only after a real dataset is obtained and inspected.

> **Note:** No real dataset has been downloaded or inspected yet. The canonical schema below does not assume any undocumented dataset field.

---

## Part 1 — Canonical Schema (Internal Project Representation)

The canonical schema is an internal project representation. Dataset-specific fields will be mapped into this representation only after the real dataset has been inspected.

Pipeline position:

```
RAW DATASET → DATASET ADAPTER → CANONICAL SECURITY EVENT → TEMPORAL GRAPH → DETECTION / RISK / ATTACK PATH
```

Implementation locations:

| Component | Module |
|-----------|--------|
| Canonical event | `ml/preprocessing/schema.py` (`CanonicalEvent`) |
| Entity types, temporal relationship | `ml/graph/schema.py` (`EntityType`, `TemporalRelationship`) |
| Unit tests | `tests/test_schema.py` |

Schema version: `1.0` (`SCHEMA_VERSION` in both modules).

### Canonical Event Schema

Class: `ml.preprocessing.schema.CanonicalEvent`

All fields are **required**; none are nullable in schema v1. There are deliberately **no optional fields** in v1 so that future extensions cannot silently change existing behaviour.

| # | Field | Type | Required | Nullable | Description |
|---|-------|------|----------|----------|-------------|
| 1 | `event_id` | `str` (non-empty) | Yes | No | Unique identifier of the event within the dataset ingestion run |
| 2 | `timestamp` | `int \| float` (Unix epoch seconds, UTC, finite, ≥ 0) | Yes | No | Event occurrence time |
| 3 | `user` | `str` (non-empty) | Yes | No | Account name associated with the event |
| 4 | `source_host` | `str` (non-empty) | Yes | No | Identifier of the source host |
| 5 | `destination_host` | `str` (non-empty) | Yes | No | Identifier of the destination host |
| 6 | `event_type` | `str`, one of `CANONICAL_EVENT_TYPES` | Yes | No | Canonical event vocabulary (currently `"authentication"`) |
| 7 | `success` | `bool` (strictly `True`/`False`) | Yes | No | Whether the action succeeded |

### Graph Entity Definitions

Class: `ml.graph.schema.EntityType`

Supported node entity types (no additional mandatory types until a later module requires them):

| Entity type | Enum member | Description | Notes |
|-------------|-------------|-------------|-------|
| User | `EntityType.USER` (`"user"`) | An account/user principal appearing as `user` on events | Identified by its string name |
| Host | `EntityType.HOST` (`"host"`) | A machine identified by `source_host` / `destination_host` | Identified by its string identifier |

### Relationship Definition

Class: `ml.graph.schema.TemporalRelationship` — a directed, time-stamped edge between two entities.

| # | Field | Type | Required | Nullable | Description |
|---|-------|------|----------|----------|-------------|
| 1 | `source` | `str` (non-empty) | Yes | No | Source node identifier |
| 2 | `destination` | `str` (non-empty) | Yes | No | Destination node identifier |
| 3 | `timestamp` | `int \| float` (Unix epoch seconds, UTC) | Yes | No | Interaction time |
| 4 | `event_id` | `str` (non-empty) | Yes | No | Links back to the originating `CanonicalEvent` |
| 5 | `event_type` | `str`, one of `CANONICAL_EVENT_TYPES` | Yes | No | Same vocabulary as canonical events |
| 6 | `user` | `str` (non-empty) | Yes | No | Account associated with the interaction |
| 7 | `success` | `bool` (strictly `True`/`False`) | Yes | No | Outcome of the interaction |

Multiple events between the same pair of entities at different timestamps are supported by design: each relationship carries its own `(source, destination, timestamp, event_id)` identity (`identity_key()`), and parallel edges are distinguished from each other. `TemporalRelationship.from_event()` derives a relationship directly from a `CanonicalEvent`.

### Validation Rules

Validation is performed in `__post_init__` and in `from_dict`; failures raise `ml.preprocessing.schema.SchemaValidationError` with explicit messages listing every violated rule.

| Rule | Applies to | Behaviour on violation |
|------|------------|------------------------|
| Missing / empty / whitespace-only identifier | `event_id`, `user`, `source_host`, `destination_host`, `source`, `destination` | Raise `SchemaValidationError` |
| Timestamp not numeric (`int`/`float`), non-finite (NaN/inf), negative, or boolean | `timestamp` | Raise `SchemaValidationError` |
| Value outside canonical event-type vocabulary | `event_type` | Raise `SchemaValidationError` |
| Value not exactly `True` or `False` (e.g., `1`, `"true"`, `None`) | `success` | Raise `SchemaValidationError` |
| Missing required key in record mapping | all `from_dict` inputs | Raise `SchemaValidationError` listing missing keys |
| Unknown extra key in record mapping | all `from_dict` inputs | Raise `SchemaValidationError` listing unknown keys (adapters must map dataset fields explicitly) |
| Non-mapping input | `from_dict` | Raise `SchemaValidationError` |

Malformed records are never silently repaired, coerced, or dropped. Instances are frozen (immutable) once validated.

### Canonical Schema vs. Future Dataset-Specific Fields

- The canonical schema is **dataset-independent**: it contains no LANL-specific or otherwise dataset-specific fields.
- Raw dataset columns (including any encoded/anonymised values in the future dataset) will be mapped into this representation by a dedicated **dataset adapter** module in a later milestone.
- The strict `from_dict` key checking forces every adapter to map dataset-specific fields explicitly rather than passing raw records through.
- Any extension to this contract (new optional metadata, new event types, new entity types) must be versioned and documented here before use.

---

## Part 2 — Dataset Identification (to be completed after dataset inspection)

| Field | Value |
|-------|-------|
| Dataset name | _TBD_ |
| Version / release | _TBD_ |
| Source / URL | _TBD_ |
| License / usage terms | _TBD_ |
| Time span covered | _TBD_ |
| File format(s) | _TBD_ |
| Size on disk | _TBD_ |

## Table / File Inventory

| Table or file name | Description | Rows (approx.) | Notes |
|--------------------|-------------|----------------|-------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Field Definitions

For each table/file, document fields using the structure below. These are dataset-specific fields only; they will be mapped into the canonical schema above by the dataset adapter.

### Table: `<table_name>` — _TBD_

| # | Field name | Data type | Units / format | Description | Example value | Missing values? | Maps to canonical field | Notes |
|---|-----------|-----------|----------------|-------------|---------------|-----------------|-------------------------|-------|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Labels / Ground Truth

| Item | Value |
|------|-------|
| Label source | _TBD_ |
| Attack scenarios included | _TBD_ |
| Labelling methodology | _TBD_ |
| Known limitations of labels | _TBD_ |

## Preprocessing Decisions Log

| Date | Decision | Reason | Performed by |
|------|----------|--------|--------------|
| _TBD_ | Defined canonical event and graph schemas independent of any dataset (M2.0) | Stable interface between dataset adapters and downstream modelling stages | M2.0 |
