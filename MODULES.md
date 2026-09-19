# MODULES.md

Each module is implemented independently.

Before implementation:

- read PROJECT_SPEC.md
- read PROJECT_STATE.md
- inspect existing code
- inspect dependencies

Claude may choose implementation details unless the project specification explicitly constrains them.

---

# M0 — PROJECT FOUNDATION

Purpose:

Establish a clean Python project and reproducible development environment.

Tasks:

- project structure
- virtual environment
- dependencies
- basic health checks
- testing infrastructure

Completion:

- Python environment works
- required packages import
- health checks pass
- pytest works

Status:

COMPLETE

---

# M1 — DATASET REQUIREMENTS

Purpose:

Define what the research dataset must provide.

Tasks:

- authentication events
- temporal information
- user identity
- source/destination hosts
- ground truth
- scale
- temporal coverage

Completion:

Dataset requirements defined.

Status:

COMPLETE

---

# M2.0 — CANONICAL SCHEMA

Purpose:

Create a stable internal event representation.

Requirements:

- CanonicalEvent
- strict validation
- deterministic event identity
- tests

Status:

COMPLETE

---

# M2.1 — TEMPORAL GRAPH

Purpose:

Represent authentication activity as a temporal enterprise graph.

Requirements:

- user nodes
- host nodes
- identity edges
- movement edges
- chronological indexing
- temporal queries
- deterministic construction

Status:

COMPLETE

---

# M2.2 — SYNTHETIC SOFTWARE FIXTURES

Purpose:

Provide deterministic graph scenarios for testing.

Requirements:

- benign
- multi-hop
- mixed
- deterministic

Restriction:

Never use as research evidence.

Status:

COMPLETE

---

# M2.3 — VISUALIZATION

Purpose:

Provide visual inspection of graph and temporal behavior.

Requirements:

- graph visualization
- temporal/timeline visualization
- deterministic example

Status:

COMPLETE

---

# M3.1 — REAL LANL ADAPTER

Purpose:

Stream original LANL authentication data into CanonicalEvent.

Requirements:

- gzip streaming
- strict parsing
- deterministic event IDs
- success/failure mapping
- no full-file loading
- malformed input handling

Status:

COMPLETE

---

# M3.2 — REDTEAM GROUND TRUTH

Purpose:

Parse and isolate LANL redteam activity.

Requirements:

- exact matching
- ground-truth representation
- duplicate preservation
- no label leakage

Status:

COMPLETE

---

# M3.3 — TEMPORAL BEHAVIORAL BASELINE

Purpose:

Establish non-graph temporal anomaly detection.

Requirements:

- chronological processing
- causal history
- bounded state
- evaluation

Status:

COMPLETE

---

# M3.4 — TEMPORAL FEATURES

Purpose:

Build the 17 causal temporal/behavioral features.

Requirements:

- no future leakage
- deterministic feature generation
- efficient historical state

Status:

COMPLETE

---

# M3.5 — ML DATASET

Purpose:

Convert temporal LANL events/features into machine-learning datasets.

Requirements:

- chronological split
- train/validation/test
- exact labels
- streaming
- memory-safe storage
- full-dataset guard

Status:

COMPLETE

---

# M3.6 — W1 RESEARCH WINDOW

Purpose:

Create deterministic, redteam-heavy real-LANL development window.

Requirements:

- context
- emission window
- chronological train/validation/test
- exact redteam labels

Status:

COMPLETE

---

# M4.0 — XGBOOST BASELINE

Purpose:

Establish a strong non-graph baseline.

Requirements:

- standard XGBoost
- class-weighted variant
- validation threshold
- test evaluation
- PR-AUC and classification metrics

Status:

COMPLETE

---

# M4.1 — CROSS-WINDOW BASELINE GENERALIZATION

Purpose:

Determine whether the W1 baseline generalizes temporally.

Requirements:

- freeze W1 model
- freeze W1 threshold
- evaluate W2
- no W2 tuning

Status:

COMPLETE

NOTE (2026-09-13): the W2 used here was silently truncated by the M3.6
event cap (last ~19 minutes and 16 of 92 attacks missing). The recorded
result describes the truncated set. See PROJECT_STATE.md section 33.
SUPERSEDED (2026-09-17) by the corrected full-W2 evaluation in
PROJECT_STATE.md section 35.

---

# M4.2 — GRAPH REPRESENTATION

Purpose:

Convert temporal LANL events into a PyTorch Geometric-compatible representation.

Requirements:

- node features
- edge features
- temporal cutoff
- event-edge mapping
- deterministic conversion
- memory-safe representation

Status:

COMPLETE

---

# M4.3 — TEMPORAL GAT

Purpose:

Evaluate whether graph neural learning improves lateral movement detection.

Requirements:

- temporal graph snapshots
- chronological blocks
- bounded history
- GPU-safe training
- validation threshold
- W1 evaluation

Status:

COMPLETE

---

# M4.4 — HYBRID GNN

Purpose:

Determine whether graph embeddings add value to the 17 temporal/behavioral features.

Required comparison:

- XGBoost
- Temporal GAT
- Hybrid GNN

Use W1.

Do not use W2 for tuning.

Do not run full LANL.

Tests:

- dimensions
- all 17 features reach classifier
- temporal correctness
- no leakage
- deterministic preprocessing
- forward/backward
- memory safety

Completion:

Actual W1 metrics and honest comparison recorded in PROJECT_STATE.md.

Status:

COMPLETE

Result: graph embeddings did not add value to the 17 features on W1
(hybrid PR-AUC 0.8444 < behavioral_only 0.9088 < XGBoost 0.9517).
See PROJECT_STATE.md sections 28 and 29.

---

# M4.5 — EXPLAINABILITY

Purpose:

Explain predictions from the selected model.

Tasks:

- determine appropriate attribution method
- implement it
- generate interpretable examples
- validate attribution behavior
- avoid introducing label leakage

Input:

Best model from M4.4.

Completion:

Explainability results recorded.

Status:

COMPLETE

Result: exact TreeSHAP on the validation-selected M4.0 XGBoost. Attributions
pass local-accuracy, dummy and deletion checks on real W1 data. Detection is
driven by source-host fan-out (source_unique_destination_count). XGBoost gain
importance disagrees with TreeSHAP (Spearman 0.10) and should not be used to
explain this model. All 102 W1 redteam records originate from a single source
host (C17693), so W1 measures detection of one attacker's sweep; this is the
leading hypothesis for the W2 collapse and should be examined in M4.6.
See PROJECT_STATE.md section 30.

---

# M4.6 — CROSS-WINDOW FINAL GENERALIZATION

Purpose:

Evaluate the selected model on W2.

Requirements:

- no W2 training
- no W2 tuning
- frozen model
- frozen threshold
- compare W1 vs W2

Completion:

Generalization results recorded.

Status:

COMPLETE

Result: the frozen W1 XGBoost does not generalise to W2 (PR-AUC 0.9517 to
0.0843, recall 1.00 to 0.47, 10 to 215 false positives per hour). The ad-hoc
M4.1 result is now reproduced exactly from code. Both windows contain the same
single attacker host (C17693), so the M4.5 different-host hypothesis is refuted:
missed W2 attacks are repeat contacts and busy-account activity outside the
narrow W1 signature. Cumulative count features are also stream-position
dependent. See PROJECT_STATE.md section 31.

NOTE (2026-09-13): these W2 figures were computed on a W2 silently
truncated by the M3.6 event cap (1,832,857 of 2,174,232 events; 76 of 92
attacks). CORRECTED (2026-09-17): on the full W2 the frozen model scores
PR-AUC 0.1257 (was 0.0843), recall 0.489, 253 false alerts per hour. The
generalisation-collapse conclusion is unchanged. See PROJECT_STATE.md
sections 33 and 35.

---

# M4.7 — ABLATION

Purpose:

Determine the value of each information source.

Compare:

- temporal/behavioral
- graph
- hybrid

Avoid excessive tuning.

Completion:

Ablation results recorded.

Status:

COMPLETE

Result (PROJECT_STATE.md sections 32 and 36): the 17 temporal/behavioural
features are the only information source with real value. The graph adds
nothing once they are present (hybrid minus behavioral_only: -0.076 W1,
+0.020 W2, overlapping seed ranges) and the 7 target attributes add
nothing. Answers RQ2 and RQ3 in the negative. Along the way this module
found the W2 truncation defect (section 33, corrected in section 35) and
that deterministic CUDA does not make GAT training reproducible on real
data (section 34).

---

# M5 — FINAL EVALUATION

Purpose:

Produce the final research evidence.

Tasks:

- select final model based on predefined methodology
- consolidate W1 results
- consolidate W2 results
- explainability
- ablation
- limitations
- computational analysis

Do not change methodology after seeing final results merely to improve performance.

Status:

NEXT

---

# M6 — FINAL PROJECT INTEGRATION

Purpose:

Turn the research implementation into a coherent final-year project.

Tasks:

- clean code paths
- reproducible commands
- final experiment scripts
- final visualizations
- final result tables
- verify tests
- verify raw data integrity

No unnecessary architectural redesign.

---

# M7 — FINAL RESEARCH DOCUMENTATION

Purpose:

Prepare the final academic material.

Potential sections:

- problem
- background
- dataset
- methodology
- architecture
- experiments
- results
- explainability
- generalization
- ablation
- limitations
- conclusion

Only begin after experimental results are stable.
