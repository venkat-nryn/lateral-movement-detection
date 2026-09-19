# Dataset Requirements Specification

**Module:** M1.1 — Dataset Requirements Specification
**Status:** Complete (requirements only; no dataset selected)
**Parent question:** See `docs/research_question.md`

> Can temporal graph learning improve the detection and analysis of multi-hop lateral movement compared with conventional machine-learning and static graph-based approaches while providing useful confidence and attack-path information for security analysts?

---

## 1. Purpose and Scope

This document defines the **requirements that a candidate dataset must satisfy** to be usable for this research project. It does **not** select, name, or evaluate any specific dataset. Dataset selection is deferred to a later module and must be performed against this specification using documented evidence only.

This document covers:

- Required characteristics of a candidate dataset.
- Highly desirable characteristics.
- Optional characteristics.
- The rationale connecting each required characteristic to this project's research formulation.
- A reusable evaluation scorecard for comparing candidates later.

This document does **not** cover dataset acquisition, preprocessing, graph construction, or model design.

---

## 2. Research Principle

> **"The dataset will be selected based on evidence that it supports the research formulation. The implementation will not assume that a particular dataset contains labels, fields, or attack information until those characteristics are verified from the dataset documentation and inspection."**

Consequences of this principle for all downstream modules:

1. No field, label, or attack scenario is assumed to exist until it is confirmed from official documentation **and** direct inspection of the data itself.
2. Where documentation is ambiguous or unavailable, the scorecard entry is recorded as **Unknown**, never guessed.
3. Any candidate that fails a REQUIRED item cannot be used as the primary evaluation dataset, regardless of other merits.
4. Claims about the final dataset's properties may appear only in modules after verification is complete.

---

## 3. Requirement Tiers

### 3.1 REQUIRED

A candidate dataset that fails any REQUIRED item is disqualified from serving as the primary evaluation dataset.

| ID | Requirement |
|----|-------------|
| R1 | **Temporal information** — timestamps or equivalent ordering information for events. |
| R2 | **Source entity** — a source host/device/entity identifier (or equivalent) per event. |
| R3 | **Destination entity** — a destination host/device/entity identifier (or equivalent) per event. |
| R4 | **User/account identity** — user/account information where available. |
| R5 | **Event/action information** — authentication/access/network/process event type or equivalent. |
| R6 | **Relationship expressiveness** — sufficient information to represent relationships between entities (i.e., events can be interpreted as edges among entities). |
| R7 | **Ground truth or defensible labeling mechanism** — either ground-truth attack information or a labeling approach that can be justified and documented. |
| R8 | **Sufficient temporal activity** — enough events over time to investigate multi-step/multi-hop behavior. |
| R9 | **Publicly documented data format** — the format and schema of the data are publicly documented. |
| R10 | **Research-use suitability/licensing information** — licensing/usage terms that permit research use and that can be cited in the project documentation. |

Notes on interpretation:

- R4 ("where available") means the *absence* of user/account fields must be explicitly documented by the candidate if it lacks them; a candidate with no identity information at all weakens the identity-based analysis but is assessed during scoring rather than auto-disqualified at specification time. If absent, the project must document how sub-question analysis adapts.
- "Equivalent" in R2/R3/R5 acknowledges that different telemetry types represent entities and actions differently; the mapping from raw fields to graph nodes/edges is decided in later modules **after inspection**.
- R8 is satisfied only if the volume and duration of activity make multi-hop path investigation feasible; a single isolated event stream with minimal connectivity does not qualify.

### 3.2 HIGHLY DESIRABLE

These characteristics strongly improve fitness for the research formulation but their absence does not automatically disqualify a candidate. Each absence must be noted in the scorecard and its impact discussed in the selection decision.

| ID | Characteristic |
|----|----------------|
| H1 | Enterprise-scale environment (realistic organizational scale). |
| H2 | Realistic authentication behavior. |
| H3 | Multiple users. |
| H4 | Multiple hosts. |
| H5 | Multiple servers. |
| H6 | Known/documented attack scenarios. |
| H7 | Known malicious activity identifiable in the data. |
| H8 | Sufficient benign background activity to avoid trivial separation of attacks. |
| H9 | Prior academic research usage (enables comparability of results). |
| H10 | Reproducibility (stable versions, checksums, or fixed releases). |
| H11 | Manageable local processing requirements (size/compute compatible with available hardware). |

### 3.3 OPTIONAL

These characteristics add analytical richness but are **not** mandatory. Their absence must never be treated as a failure.

| ID | Characteristic |
|----|----------------|
| O1 | Process information. |
| O2 | Network flow information. |
| O3 | DNS information. |
| O4 | Privilege information (e.g., privilege level of accounts). |
| O5 | Asset criticality information (e.g., importance/value of hosts). |
| O6 | Additional security telemetry beyond the core event types. |

Optional fields may enrich modeling (e.g., privilege escalation cues, asset-weighted risk) but no downstream module may make them mandatory assumptions.

---

## 4. Task-Specific Rationale for REQUIRED Items

Each required requirement is tied directly to the research formulation. These rationales describe what the *project needs*; they make no claim about what any particular dataset provides.

**R1 — Temporal information → temporal modeling.**
The central comparison in this research is between *temporal* graph learning and static approaches. Without timestamps or equivalent ordering, no temporal representation can be constructed, and the primary research question cannot be evaluated.

**R2 + R3 — Source and destination entities → graph edges.**
Graph representations require endpoints. Source/destination information is what allows events to be converted into directed edges between entities; without both endpoints, no graph — temporal or static — can be built.

**R4 — User/account identity → identity-based movement.**
Lateral movement is frequently characterized by credentials being reused across hosts. User/account fields allow the model to capture identity-based movement patterns (user–host interactions over time), which are central to detecting multi-hop behavior.

**R5 — Event/action information → edge semantics.**
Knowing whether an event is an authentication, access, network connection, or process action gives edges semantic meaning. This supports interpretable attack-path reconstruction (analysts see *what happened*, not just that two entities are connected).

**R6 — Relationship expressiveness → graph construction feasibility.**
The entire pipeline assumes entity–relationship structure. If events cannot be interpreted as relations among entities, neither temporal graphs nor static-graph baselines can be built on equal footing.

**R7 — Ground truth or defensible labels → meaningful supervised evaluation.**
Detection quality claims (and comparisons against conventional ML baselines) require some notion of ground truth. Without ground-truth attack information or a defensible, documented labeling mechanism, results would be unverifiable and the supervised comparison invalid.

**R8 — Sufficient temporal activity → multi-hop investigation.**
Sub-questions concern *multi-step/multi-hop* movement and attack-path reconstruction. Sparse or short-lived activity cannot support the study of chained behaviors across multiple entities, so adequate temporal depth and connectivity are essential.

**R9 — Publicly documented data format → reproducibility and correct interpretation.**
A publicly documented format ensures fields can be interpreted correctly, parsing is reproducible, and others can verify results — necessary for defensible research claims.

**R10 — Licensing/research-use suitability → legitimate, citable usage.**
The dataset must be legally and ethically usable for this research, with terms that can be documented in the thesis/project report.

---

## 5. Dataset Evaluation Scorecard

To be completed **per candidate dataset** in a later module, strictly from documented evidence and direct inspection. Use `Unknown` whenever evidence has not yet been gathered — do not guess.

**Candidate:** _<to be filled>_
**Evidence sources consulted:** _<to be filled>_
**Date assessed:** _<to be filled>_

| Requirement | Importance | Pass/Fail | Evidence | Notes |
|-------------|------------|-----------|----------|-------|
| R1 Temporal information | REQUIRED | Unknown | None yet | |
| R2 Source entity | REQUIRED | Unknown | None yet | |
| R3 Destination entity | REQUIRED | Unknown | None yet | |
| R4 User/account identity | REQUIRED | Unknown | None yet | |
| R5 Event/action information | REQUIRED | Unknown | None yet | |
| R6 Relationship expressiveness | REQUIRED | Unknown | None yet | |
| R7 Ground truth / defensible labels | REQUIRED | Unknown | None yet | |
| R8 Sufficient temporal activity | REQUIRED | Unknown | None yet | |
| R9 Publicly documented format | REQUIRED | Unknown | None yet | |
| R10 Licensing / research-use suitability | REQUIRED | Unknown | None yet | |
| H1 Enterprise-scale environment | Highly desirable | Unknown | None yet | |
| H2 Realistic authentication behavior | Highly desirable | Unknown | None yet | |
| H3 Multiple users | Highly desirable | Unknown | None yet | |
| H4 Multiple hosts | Highly desirable | Unknown | None yet | |
| H5 Multiple servers | Highly desirable | Unknown | None yet | |
| H6 Known attack scenarios | Highly desirable | Unknown | None yet | |
| H7 Known malicious activity | Highly desirable | Unknown | None yet | |
| H8 Sufficient benign activity | Highly desirable | Unknown | None yet | |
| H9 Academic research usage | Highly desirable | Unknown | None yet | |
| H10 Reproducibility | Highly desirable | Unknown | None yet | |
| H11 Manageable local processing | Highly desirable | Unknown | None yet | |
| O1 Process information | Optional | Unknown | None yet | |
| O2 Network flow information | Optional | Unknown | None yet | |
| O3 DNS information | Optional | Unknown | None yet | |
| O4 Privilege information | Optional | Unknown | None yet | |
| O5 Asset criticality information | Optional | Unknown | None yet | |
| O6 Additional security telemetry | Optional | Unknown | None yet | |

Scoring rules:

- **REQUIRED:** any Fail ⇒ candidate ineligible as primary dataset. Any Unknown ⇒ selection blocked until resolved via documentation or inspection.
- **HIGHLY DESIRABLE:** each Fail must be accompanied by a written impact note; cumulative failures weigh against selection.
- **OPTIONAL:** Fails carry no penalty; they simply narrow optional analyses.
- Evidence entries must cite documentation sections, schema files, or inspection outputs — not assumptions.

---

## 6. Scope Boundaries

- This document specifies requirements only. It does not assert that any dataset exists which satisfies them.
- No candidate dataset is named, compared, or ranked here.
- Field-level mappings from raw data to graph structures are defined only after a candidate passes this scorecard in a later module.

**Stop point:** M1.1 ends here. Dataset identification and scoring belong to subsequent modules.
