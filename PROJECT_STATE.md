# PROJECT_SPEC.md

# 1. PROJECT TITLE

Real-World Lateral Movement Detection in Evolving Enterprise Networks

---

# 2. PROJECT PURPOSE

Build and experimentally evaluate a cybersecurity system for detecting lateral movement in enterprise authentication activity using temporal behavioral features and temporal graph learning.

The project should investigate whether graph structure provides useful information beyond conventional temporal/behavioral features and whether the resulting models generalize across different temporal windows of an evolving enterprise network.

The project is a research prototype.

---

# 3. CORE RESEARCH PROBLEM

Lateral movement occurs when an attacker moves through an enterprise environment after obtaining access to one or more systems.

Authentication logs contain signals of this behavior, but enterprise networks are:

- large
- highly imbalanced
- dynamic
- temporally evolving
- heterogeneous
- noisy

The project therefore investigates:

1. temporal behavioral representations
2. graph representations
3. temporal graph neural networks
4. hybrid graph + behavioral representations
5. explainability
6. temporal/cross-window generalization

---

# 4. CENTRAL RESEARCH QUESTION

Does incorporating temporal graph structure improve detection of lateral movement over conventional temporal/behavioral features, and does the resulting model generalize to later periods of an evolving enterprise network?

---

# 5. RESEARCH SUBQUESTIONS

## RQ1

How well can temporal/behavioral features detect known lateral movement?

## RQ2

Does a temporal graph representation improve detection?

## RQ3

Does combining graph representations with temporal/behavioral features improve performance?

## RQ4

Does the model generalize across temporally separated enterprise windows?

## RQ5

What behaviors/features contribute most to suspicious predictions?

---

# 6. DATASET

Primary dataset:

LANL Comprehensive Multi-Source Cyber-Security Events.

Current research files:

data/raw/lanl/auth.txt.gz
data/raw/lanl/redteam.txt.gz

Relevant data:

Authentication events provide:

- timestamp
- user
- source host
- destination host
- authentication outcome and related fields

Redteam data provides known red-team activity used as ground truth.

Raw files must remain untouched.

---

# 7. GROUND TRUTH

Redteam records are matched to authentication events using:

(timestamp, user, source_host, destination_host)

Matching is exact.

Do not use fuzzy matching.

Do not infer labels from model predictions.

Do not alter redteam labels.

Destination mismatches must remain mismatches.

---

# 8. CANONICAL EVENT

CanonicalEvent contains:

- event_id
- timestamp
- user
- source_host
- destination_host
- event_type
- success

Current event type:

authentication

Event IDs must be deterministic.

---

# 9. GRAPH MODEL

The enterprise is represented as a temporal directed multigraph.

Node types:

- user
- host

Deterministic node identifiers:

user:<id>
host:<id>

Each authentication event creates two conceptual edges:

## Identity edge

USER → DESTINATION_HOST

## Movement edge

SOURCE_HOST → DESTINATION_HOST

Both edges correspond to the same underlying authentication event.

This is important because source_host must be explicitly represented for host-to-host movement analysis.

---

# 10. TEMPORAL GRAPH REQUIREMENTS

The graph must support:

- chronological event ordering
- event lookup
- node histories
- temporal neighborhoods
- deterministic construction
- temporal cutoffs
- event-to-edge mapping
- bounded historical context where necessary

No future events may enter historical context.

---

# 11. TEMPORAL/BEHAVIORAL FEATURES

Current M3.4 feature set contains 17 features:

1. source_equals_destination
2. user_destination_seen
3. source_destination_seen
4. user_destination_count
5. source_destination_count
6. user_unique_destination_count
7. source_unique_destination_count
8. destination_unique_source_count
9. user_recent_event_count
10. user_recent_unique_destination_count
11. time_since_user_previous_event
12. time_since_user_destination_event
13. time_since_source_destination_event
14. failed_authentication
15. user_recent_failure_count
16. destination_recent_event_count
17. movement_edge

The feature pipeline must be strictly causal.

At prediction time, only historical information is visible.

---

# 12. DATASET SPLITTING

Splits are chronological.

Current W1:

Context:

760506–764106

Emission:

764106–771306

Current W1 dataset:

Context events:
645,665

Emitted events:
1,279,764

Train:

892,424
69 positives

Validation:

186,504
15 positives

Test:

200,836
11 positives

---

# 13. W2 GENERALIZATION WINDOW

W2:

Context:

1067648–1071248

Emission:

1071248–1078448

Emitted:

1,832,857

Exact positives:

76 / 92 redteam records

W2 is reserved primarily for cross-window generalization evaluation.

Do not tune the model on W2 unless a future module explicitly defines this.

---

# 14. CURRENT BASELINE

M4.0 Standard XGBoost currently provides the strongest result.

W1:

PR-AUC:
0.9517

Precision:
0.7857

Recall:
1.0

F1:
0.880

TP:
11

FP:
3

TN:
200822

FN:
0

ROC-AUC:
0.999997

Validation-selected threshold:

0.015060

Important:

The test set contains only 11 positives, so metrics have substantial uncertainty.

---

# 15. CLASS-WEIGHTED XGBOOST

W1:

PR-AUC:
0.8588

Precision:
0.8182

Recall:
0.8182

F1:
0.8182

TP:
9

FP:
2

TN:
200823

FN:
2

This is a secondary baseline.

---

# 16. CROSS-WINDOW RESULT

Frozen W1 standard XGBoost evaluated on W2.

At W1 threshold 0.015060:

TP:
36

FP:
430

TN:
1,832,351

FN:
40

Precision:
0.077

Recall:
0.474

F1:
0.133

PR-AUC:
0.0843

ROC-AUC:
0.9970

Important research finding:

The model performs extremely well on W1 but generalizes poorly to W2.

This is a central finding, not an error to hide.

---

# 17. GRAPH REPRESENTATION

M4.2 provides a NumPy-first graph snapshot.

Node features:

8

- is_user
- is_host
- event_count
- out_degree
- in_degree
- recent_event_count
- success_count
- failure_count

Edge features:

6

- success
- is_identity
- is_movement
- self_loop
- event_type_authentication
- time_delta

Additional:

edge_event_index
event_timestamps
node_type
edge_type

Each event produces two edges.

---

# 18. TEMPORAL GAT

M4.3 implements a Temporal GAT.

Approximate parameter count:

10,285

Architecture:

input normalization
→ GAT layer
→ ELU
→ GAT layer
→ target-event classifier

Uses chronological blocks and bounded historical context.

W1 result:

PR-AUC:
0.0775

Precision:
0.20

Recall:
0.3636

F1:
0.2581

TP:
4

FP:
16

TN:
200809

FN:
7

The pure graph model performs substantially worse than XGBoost.

Current hypothesis:

The graph representation aggregates information and loses some pair-specific temporal information retained by the 17 temporal/behavioral features.

---

# 19. MODEL PROGRESSION

The intended progression is:

M4.0

Temporal/behavioral baseline

↓

M4.2

Graph representation

↓

M4.3

Temporal GNN

↓

M4.4

Hybrid graph + temporal/behavioral model

↓

M4.5

Explainability

↓

M4.6

Cross-window generalization

↓

M4.7

Ablation

↓

M5

Final research evaluation

The exact implementation may evolve based on experimental results.

---

# 20. HYBRID MODEL

M4.4 investigates whether graph embeddings can complement the existing 17 temporal/behavioral features.

Conceptually:

Graph representation
+
GNN embeddings
+
temporal/behavioral features
→
classifier

The implementation should be designed by inspecting the existing architecture rather than blindly introducing a parallel pipeline.

---

# 21. EXPLAINABILITY

The project should explain suspicious predictions.

Possible approaches include:

- feature importance
- SHAP-style analysis
- model-specific attribution
- graph/attention analysis where scientifically justified

The explainability method should be appropriate for the final selected model.

---

# 22. ABLATION

The final study should compare information sources.

Potential comparison:

1. temporal/behavioral only
2. graph only
3. hybrid

The purpose is to determine what information actually contributes to detection.

---

# 23. FINAL EVALUATION

The final evaluation should establish:

- best-performing model
- W1 performance
- W2 generalization
- explainability
- ablation results
- limitations
- computational requirements
- research conclusions

---

# 24. EXPECTED RESEARCH CONTRIBUTION

Do NOT claim:

"first GNN for lateral movement detection"

or similar unsupported novelty claims.

The graph-based lateral movement detection area is already well studied.

The contribution should instead be framed around the actual experimental study, such as:

- real LANL evaluation
- temporal graph representation
- comparison with strong temporal baseline
- hybrid representation
- cross-window generalization
- explainability
- empirical analysis of graph information

The final contribution must be determined by actual results.

---

# 25. RESOURCE LIMITS

Hardware:

RTX 2050
4 GB VRAM

Therefore:

- do not load entire LANL auth data into memory
- use streaming
- use bounded windows
- use bounded histories
- use memory-safe batches
- avoid repeated full-dataset runs

Full-dataset processing should only occur when explicitly required for final research evaluation.

---

# 26. SYNTHETIC DATA POLICY

Synthetic data is allowed for:

- unit tests
- fixtures
- deterministic software validation
- debugging

Synthetic data is prohibited for:

- research metrics
- final model comparisons
- research conclusions
- claims about LANL behavior

---

# 27. FINAL PROJECT OUTPUT

The finished system should contain:

1. LANL ingestion
2. canonical event representation
3. temporal graph representation
4. temporal behavioral features
5. baseline model
6. graph neural model
7. hybrid model
8. explainability
9. cross-window evaluation
10. ablation
11. final evaluation
12. reproducible tests and experiment scripts

The project should remain understandable, reproducible, and scientifically defensible.

---

# 28. M4.4 HYBRID GNN RESULTS (W1)

Status: COMPLETE (2026-09-02)

## 28.1 Question

Do graph embeddings add value to the 17 temporal/behavioural features?

Answering this required a third arm. Comparing the hybrid against XGBoost
confounds the representation with the model family (MLP head vs gradient-boosted
trees), so a `behavioral_only` neural control was added: the identical network
with the GAT branch removed. The graph's contribution is `hybrid` minus
`behavioral_only`, and nothing else.

The formal ablation across information sources remains M4.7.

## 28.2 Implementation

Extended `ml/models/gnn.py` rather than adding a parallel pipeline.

`EventGraphBatch` gained a `behavioral` array of shape (targets, 17). It is
filled by the existing M3.4 `TemporalGraphFeatureExtractor`, run inside the same
single streaming pass that builds the graph batches, over every event including
the context region. A test asserts the produced values equal a standalone M3.4
pass exactly, so the hybrid and the M4.0 baseline provably consume the same
numbers.

One model class now covers three variants:

- `graph_only`      — M4.3 unchanged (10,285 params, verified identical)
- `behavioral_only` — GAT branch removed (881 params)
- `hybrid`          — graph embeddings + the 17 features (10,863 params)

The GAT encoder is identical wherever it is present.

## 28.3 Train-only preprocessing

`BehavioralScaler` implements RESEARCH_CONSTRAINTS section 13:

1. `signed_log1p(v) = sign(v) * log1p(|v|)` — fixed, learns nothing. Chosen over
   `log1p(clamp(min=0))` because the M3.4 time deltas use -1.0 for "no prior
   event", which clamping would collapse onto a genuine zero-second gap.
2. Standardisation with mean/std fitted on TRAIN batches ONLY (892,424 events),
   then frozen for validation and test.

The scaler refuses non-TRAIN batches, refuses refitting, and refuses use before
fitting. A test confirms the statistics are unchanged when validation/test
events are absent from the stream.

## 28.4 Configuration

Window: W1 (context 760506-764106, emission 764106-771306). W2 not read.

157 chronological blocks, block_size 8192, max_history_events 20000,
hidden_dim 32, heads 4, dropout 0.0, lr 0.005, Adam, 20 epochs, seed 0,
BCEWithLogitsLoss with pos_weight computed from TRAIN only.

Events: 645,665 context + 1,279,764 emitted.
Train 892,424 (69 positive) / validation 186,504 (15) / test 200,836 (11).

Preprocessing 174.1s CPU. Batch cache 450.9 MB (363.9 MB in M4.3, +87 MB for the
17-feature block). Peak process RSS 1039 MB. Device CUDA; peak VRAM 212.1 MB of
4096 for the graph variants, 32.0 MB for `behavioral_only`.

## 28.5 W1 TEST RESULTS — canonical deterministic run

Threshold selected on validation, then frozen. Run with `--deterministic`.

| model | params | PR-AUC | ROC-AUC | P | R | F1 | TP | FP | TN | FN | pred+ | threshold | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost:standard (M4.0) | n/a | 0.9517 | 0.999997 | 0.7857 | 1.0000 | 0.8800 | 11 | 3 | 200822 | 0 | 14 | 0.015060 | 6.5 |
| gnn:behavioral_only | 881 | 0.9088 | 0.999993 | 0.8333 | 0.9091 | 0.8696 | 10 | 2 | 200823 | 1 | 12 | 0.996246 | 17.2 |
| gnn:hybrid | 10863 | 0.8444 | 0.999973 | 0.6000 | 0.8182 | 0.6923 | 9 | 6 | 200819 | 2 | 15 | 0.999382 | 263.0 |
| gnn:graph_only (M4.3 arch) | 10285 | 0.1403 | 0.995282 | 0.1000 | 0.5455 | 0.1690 | 6 | 54 | 200771 | 5 | 60 | 0.999047 | 262.0 |

Validation PR-AUC at the selected epoch: behavioral_only 0.9261 (epoch 17),
hybrid 0.7350 (epoch 20), graph_only 0.1404 (epoch 20).

## 28.6 W1 TEST RESULTS — first (non-deterministic) run, preserved

Same code and configuration, run before `--deterministic` existed. Preserved per
RESEARCH_CONSTRAINTS section 23; the difference between the two runs is the
evidence for section 29.

| model | PR-AUC | P | R | F1 | TP | FP | TN | FN | pred+ |
|---|---|---|---|---|---|---|---|---|---|
| gnn:behavioral_only | 0.9088 | 0.8333 | 0.9091 | 0.8696 | 10 | 2 | 200823 | 1 | 12 |
| gnn:hybrid | 0.7765 | 0.4762 | 0.9091 | 0.6250 | 10 | 11 | 200814 | 1 | 21 |
| gnn:graph_only | 0.0333 | 0.0370 | 0.0909 | 0.0526 | 1 | 26 | 200799 | 10 | 27 |

`behavioral_only` is bitwise identical across both runs. Only the graph-bearing
variants moved.

## 28.7 Conclusion

**Graph embeddings did not add value to the 17 temporal/behavioural features on
W1. They reduced performance.**

Both runs agree on the ordering:

- deterministic: hybrid 0.8444 < behavioral_only 0.9088
- non-deterministic: hybrid 0.7765 < behavioral_only 0.9088

The cost is precision. Hybrid and behavioral_only recover a similar number of
attacks (TP 9-10 of 11), but the hybrid raises 6 false positives against the
control's 2 at the frozen threshold. The graph branch added noise, not signal.

The 17 features remain the dominant information source. XGBoost on those
features (PR-AUC 0.9517) is still the strongest model overall; the neural
control on the same features reaches 0.9088, so about 0.04 PR-AUC of the gap
between the hybrid and the baseline is attributable to model family and the
remainder to the graph branch.

This is a negative result for RQ3 on W1 and is reported as such. It is
consistent with the M4.3 hypothesis in section 18: the M4.2 node features
aggregate history into per-node degree/activity counts, discarding the exact
pair-specific history (`user_destination_count`,
`time_since_user_destination_event`) that the M3.4 features retain.

## 28.8 Limitations

1. **11 test positives.** Differences of one or two true positives move F1
   substantially. No result here is statistically definitive.
2. **Validation instability.** Validation PR-AUC for the graph-bearing variants
   swings widely across epochs, so validation-based epoch selection is itself a
   noisy estimator for those variants. `behavioral_only` converges smoothly.
3. **Single configuration.** One hyperparameter setting per variant, fixed
   before results were seen and not tuned. A larger graph model, richer node
   features, or longer training might change the outcome; that was deliberately
   not explored, to avoid result chasing.
4. **Deterministic CUDA costs about 3.7x training time** (262s vs 70s) for the
   graph variants. `behavioral_only` is unaffected.
5. **W1 only.** Cross-window behaviour of these models is M4.6.

---

# 29. RUN-TO-RUN DETERMINISM FINDING (M4.4)

Discovered while reconciling two `graph_only` runs.

**The GAT branch is not run-to-run reproducible on CUDA by default.** The
scatter reduction inside `GATConv` accumulates in a non-deterministic order on
GPU. Measured directly, on a fixed seed and fixed data:

| variant | CPU reproducible | CUDA reproducible (default) |
|---|---|---|
| graph_only | yes | **no** |
| behavioral_only | yes | yes |

Per-epoch loss differences start around 1e-9 and compound. Because the
graph-bearing variants have unstable validation PR-AUC across epochs,
validation-based epoch selection can land on a different epoch, which changes
the reported test metrics materially.

Observed `graph_only` W1 test PR-AUC across three runs of identical code and
configuration: **0.0775** (M4.3, section 18), **0.0333** (section 28.6),
**0.1403** (section 28.5, deterministic).

This does not invalidate the M4.3 conclusion — the pure graph model is far below
the baseline in all three runs — but the specific figure in section 18 should be
read as one sample from a wide distribution, not a stable point estimate.

Mitigation, implemented: `scripts/run_gnn_experiment.py --deterministic` sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`
before CUDA initialises. Verified to produce bitwise-identical runs. A CPU
reproducibility test covering all three variants is pinned in
`tests/test_hybrid_gnn.py`.

Recommendation for M4.5 onward: run graph-bearing experiments with
`--deterministic`, and treat any single-run graph metric with caution.


---

# 30. M4.5 EXPLAINABILITY RESULTS (W1)

Status: COMPLETE (2026-09-11)

## 30.1 Model selected

M4.5 explains "the best model from M4.4". Selection used VALIDATION PR-AUC only;
test was not consulted.

| candidate | validation PR-AUC | source |
|---|---|---|
| xgboost:standard | **0.9332** | this run |
| gnn:behavioral_only | 0.9261 | section 28.5 |
| xgboost:class_weighted | 0.7866 | this run |
| gnn:hybrid | 0.7350 | section 28.5 |
| gnn:graph_only | 0.1404 | section 28.5 |

Selected: **standard M4.0 XGBoost**.

Reproduction check: retraining it exactly as M4.0 did reproduced section 14
exactly — validation-selected threshold 0.015060, test TP 11 / FP 3 /
TN 200822 / FN 0, PR-AUC 0.951671.

## 30.2 Method

Exact TreeSHAP (Lundberg et al.) via XGBoost's native `pred_contribs`. Exact
Shapley values from the tree structure: no sampling, no approximation. No new
dependency — `shap` is not installed and is not needed. Contributions are
additive in log-odds (margin) space. The only background distribution is the
training cover stored in the trees, so no validation or test statistic enters.

Graph/attention attribution was not used: the graph models were not selected
(section 28), so attention analysis would explain a model the project does not
use.

Files: `ml/evaluation/explainability.py`, `scripts/run_explainability.py`,
`tests/test_explainability.py` (26 tests).

Label separation: the explainer takes `(model, X)` only; its signature admits no
labels. Labels appear in one function, which groups finished attributions into
TP/FP/FN/TN for reporting. Nothing produced feeds back into the model,
threshold or features.

## 30.3 Attribution validity on real W1 data

| check | result |
|---|---|
| local accuracy, max abs(sum(phi) + bias - margin) | 2.10e-5 validation, 2.05e-5 test (float32 rounding) |
| decision consistency, max abs(sigmoid(margin) - score) | 7.0e-8 — the explained quantity is the thresholded score |
| bias (expected log-odds over training cover) | -10.0226 |
| dummy property | `movement_edge` receives exactly 0 on every event |
| ranking stability, Spearman(validation, test) of mean abs(phi) | 0.9975 |

`movement_edge` is exactly `1 - source_equals_destination`, so the trees never
need it. It is a redundant feature, and TreeSHAP correctly gives it zero.

Deletion (faithfulness) check. Flagged events only; the chosen features are
replaced with TRAIN-only medians; values are mean margin drop in log-odds;
random choice is seeded (0).

| split | k | events | drop: top-k phi | drop: random-k | drop: bottom-k | top > random |
|---|---|---|---|---|---|---|
| validation | 1 | 18 | 10.341 | 0.556 | 0.000 | 100% |
| validation | 2 | 18 | 11.729 | 3.625 | 0.019 | 100% |
| validation | 3 | 18 | 13.078 | 4.627 | -0.133 | 100% |
| test | 1 | 14 | 10.358 | 1.113 | 0.000 | 100% |
| test | 2 | 14 | 11.992 | 2.623 | 0.000 | 100% |
| test | 3 | 14 | 12.970 | 3.894 | -0.457 | 100% |

Removing the single top-attributed feature removes about 10 log-odds —
enough on its own to drop every flagged event below the threshold. The
attributions point at what actually drives the score.

## 30.4 Global importance (TEST, 200,836 events)

| rank | feature | mean abs(phi) | share |
|---|---|---|---|
| 1 | source_unique_destination_count | 2.2523 | 25.7% |
| 2 | source_destination_count | 1.6310 | 18.6% |
| 3 | destination_recent_event_count | 0.9318 | 10.6% |
| 4 | user_destination_count | 0.9134 | 10.4% |
| 5 | user_recent_event_count | 0.7587 | 8.6% |
| 6 | user_recent_unique_destination_count | 0.4820 | 5.5% |
| 7 | user_destination_seen | 0.4323 | 4.9% |
| 8 | user_unique_destination_count | 0.4294 | 4.9% |
| 9-16 | remaining eight features | 0.04-0.23 each | 13.3% combined |
| 17 | movement_edge | 0.0000 | 0.0% |

## 30.5 What distinguishes positives

Mean phi on TEST, grouped by class. Labels were used only to group finished
attributions.

| feature | positives | negatives |
|---|---|---|
| source_unique_destination_count | +4.872 | -1.699 |
| source_destination_count | +1.503 | -1.595 |
| source_destination_seen | +1.399 | -0.056 |
| user_unique_destination_count | +1.037 | -0.239 |
| destination_unique_source_count | +0.883 | +0.062 |

The model flags an event when its source host has already reached an unusually
large number of distinct destinations, and is now contacting a destination it
has never reached before. This is the classic lateral-movement fan-out
signature.

## 30.6 XGBoost gain importance is misleading for this model

Spearman(gain, TreeSHAP mean abs(phi)) = **0.10**. Gain ranks
`source_equals_destination` first (52% of total gain) and
`source_destination_seen` second; TreeSHAP ranks them 14th and 13th.

Gain measures loss reduction at splits. It is dominated by a few early splits on
binary features that separate the easy bulk of benign traffic, and it does not
measure contribution to any prediction. The M4.0 gain ranking
(`feature_importance_ranking`) must not be quoted as an explanation of what
drives detections.

## 30.7 Local explanations and identities

Every flagged test event (TP 11, FP 3, FN 0) has
`source_unique_destination_count` as its largest contribution, between +4.1 and
+5.2 log-odds.

Identities were recovered by a one-off bounded diagnostic that streamed only
auth region [770380, 771140], kept in the session scratchpad rather than the
repository:

- **11/11 TP:** source host **C17693**; users U7375 (6), U66 (3), U4448 (2); 11
  distinct destinations. The fan-out count rises monotonically from 129 to 137
  across the test period.
- **FP t=770435:** U66@DOM1, C17693 -> C1881
- **FP t=770753:** U4448@DOM1, C17693 -> C3774
- **FP t=770607:** C3758$@DOM1 (machine account), C1521 -> C3758; that source has
  a fan-out of 308.

Two of the three FPs come from the attacker's source host and use compromised
accounts that appear in labeled redteam records. Under the exact-match
ground-truth rule (section 7; RESEARCH_CONSTRAINTS section 4) they remain false
positives and are **not relabeled**. They are, however, identity-indistinguishable
from labeled attack traffic, so W1 precision may understate practical precision.
This is an observation, not a label change. The third FP is a genuinely benign
high-fan-out host, flagged by the same feature.

## 30.8 Central finding: W1 is a single-source campaign

From `redteam.txt.gz` alone:

| split | redteam records | source hosts | distinct users | distinct destinations |
|---|---|---|---|---|
| train | 76 | C17693 only | 13 | 60 |
| validation | 15 | C17693 only | 5 | 15 |
| test | 11 | C17693 only | 3 | 11 |

**All 102 W1 redteam records originate from the single source host C17693.**
The 76 train records map to 69 matched events; unmatched records are preserved
per section 7.

Consequences:

1. W1 measures detection of one attacker host's sweep. The near-perfect W1
   metrics (PR-AUC 0.95) are best read as "the model learned C17693's fan-out
   in training and recognised its continuation in test". They are not evidence
   of general lateral-movement detection.
2. Reliance on source-host fan-out is appropriate for this campaign. It is also
   exactly the kind of campaign-specific dependence that may not transfer.
3. **Hypothesis for M4.6 — not tested here; W2 was not read in M4.5:** the M4.1
   collapse on W2 (PR-AUC 0.0843) arises because W2 attack traffic differs in
   source host and/or fan-out profile. M4.6 should compare W1 and W2 attacker
   identity and fan-out profiles (`ml_window.identity_overlap` already exists)
   alongside the frozen-model evaluation.

This also sharpens section 28: the neural/hybrid comparisons rest on 11 test
events from one host, so small-sample caution applies doubly.

## 30.9 Runtime and memory

W1 build 150.0s. XGBoost retrain (both variants) 12.4s. TreeSHAP over 387,340
events 27.1s on CPU. Peak RSS 812 MB. No GPU needed. The diagnostic identity
stream took about 2 minutes.

## 30.10 Limitations

1. **Correlated features share credit.** Many of the 17 features are
   near-duplicates (count / seen / time-since for the same pair). Path-dependent
   TreeSHAP divides credit among correlated features in a model-dependent way,
   so the order within a correlated group should not be over-interpreted.
2. **Log-odds space.** Contributions add in log-odds, not probability. Against a
   bias of -10, a +5 contribution is decisive.
3. **The deletion check** replaces features jointly with TRAIN medians. It is not
   additive, and it covers only 14 test and 18 validation flagged events. It is a
   sanity check of direction and magnitude, not a proof.
4. **One campaign** (section 30.8).
5. **Explanations describe the model, not the attack.** They are only as
   meaningful as the features themselves.

## 30.11 Conclusion

**RQ5 (W1):** the selected detector's suspicious predictions are driven by
source-host fan-out — the number of distinct destinations the source host has
already reached — reinforced by first contact between source and destination
and by user fan-out. The attributions are exact, stable across splits, and
faithful under deletion.

W1's positives, however, all come from a single attacker host. The explanation
therefore also exposes the limit of the W1 result: the model has learned one
host's sweep. Whether that generalises is the M4.6 question.


---

# 31. M4.6 CROSS-WINDOW FINAL GENERALIZATION RESULTS

Status: COMPLETE (2026-09-13)

## 31.1 Protocol

Model: the standard M4.0 XGBoost, selected on validation PR-AUC in M4.5
(section 30.1).

RESEARCH_CONSTRAINTS section 9, enforced in code (`ml/evaluation/cross_window.py`):

1. Fit on W1 TRAIN only (892,424 events, 69 positive).
2. Select the threshold on W1 VALIDATION only (186,504 events, 15 positive):
   0.015060.
3. Freeze. `FrozenDetector` is immutable and stores a SHA-256 digest of the
   serialised booster; the digest is re-verified before and after every
   evaluation. It was unchanged at the end of the run.
4. Evaluate on every emitted W2 event. W2 is never used to fit, tune,
   re-threshold or select; W2 labels are used only for metrics and for grouping
   finished results.

Window guard: W2's whole read region, context included (from 1067648), must
start after W1's emission ends (771306). Verified.

W1 was released from memory before W2 was opened.

Files: `ml/evaluation/cross_window.py`, `scripts/run_cross_window.py`,
`tests/test_cross_window.py` (21 tests).

## 31.2 Reproducibility

- W1 and W2 selected by the M3.6 rule match sections 12 and 13 exactly.
- W1 test reproduces section 14 exactly.
- **W2 reproduces the M4.1 result in section 16 exactly** (TP 36, FP 430,
  TN 1,832,351, FN 40, PR-AUC 0.0843, ROC-AUC 0.9970). M4.1 was originally run
  ad hoc with no code in the repository; it is now reproducible with
  `scripts/run_cross_window.py`.
- W2: 1,832,857 emitted events, 76 matched positives, 92 redteam records in
  range (16 records do not match an authentication event exactly and are
  preserved per section 7).

## 31.3 W1 versus W2 at the frozen threshold (0.015060)

| window | events | positives | PR-AUC | ROC-AUC | P | R | F1 | TP | FP | TN | FN | pred+ | FP per hour |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1 test | 200,836 | 11 | **0.9517** | 0.999997 | 0.7857 | 1.0000 | 0.8800 | 11 | 3 | 200,822 | 0 | 14 | 10.0 |
| W2 emitted | 1,832,857 | 76 | **0.0843** | 0.997022 | 0.0773 | 0.4737 | 0.1328 | 36 | 430 | 1,832,351 | 40 | 466 | 215.0 |

FP per hour uses the evaluated span: 1,080 s for W1 test and 7,200 s for W2.

**The selected model does not generalise from W1 to W2.** PR-AUC falls from
0.9517 to 0.0843. Recall halves (1.00 to 0.47) while the alert rate rises from
10 to 215 per hour. The W2 ROC-AUC of 0.997 is inflated by roughly 24,000
negatives per positive and is not evidence of useful performance
(RESEARCH_CONSTRAINTS section 15).

## 31.4 Testing the M4.5 hypothesis (section 30.8)

The hypothesis: W2 attack traffic differs from W1 in source host and/or fan-out
profile.

### Attacker identity (redteam.txt.gz)

| identity | W1 | W2 | shared | Jaccard |
|---|---|---|---|---|
| source hosts | 1 | 1 | 1 (C17693) | **1.000** |
| users | 17 | 24 | 4 (U1480, U162, U4448, U66) | 0.108 |
| destinations | 78 | 39 | 11 | 0.104 |

**All 92 W2 redteam records also originate from C17693.** The "different source
host" part of the hypothesis is **refuted**. W1 and W2 are the same attacker
host, about 3.5 days apart, using largely different accounts and destinations.

### Fan-out profile (`source_unique_destination_count`)

| group | n | min | p25 | median | p75 | p99 | max |
|---|---|---|---|---|---|---|---|
| W1 train positives | 69 | 25 | 43 | 85 | 101 | 118 | 118 |
| W1 train negatives | 892,355 | 0 | 5 | 6 | 12 | 79 | 5,183 |
| W1 test positives | 11 | 129 | 130 | 133 | 134 | 137 | 137 |
| W2 positives | 76 | 21 | 38 | 58 | 70 | 90 | 90 |
| W2 TP | 36 | 30 | 40 | 56 | 67 | 86 | 86 |
| W2 FN | 40 | 21 | 37 | 63 | 81 | 90 | 90 |
| W2 FP | 430 | 22 | 25 | 31 | 45 | 3,092 | 4,565 |
| W2 negatives | 1,832,781 | 0 | 6 | 8 | 12 | 65 | 6,410 |

W2 attack fan-out is lower than W1's (median 58 vs 85 in W1 train). But within
W2, detected and missed attacks have similar fan-out (median 56 vs 63). **Fan-out
does not explain which W2 attacks were missed**, so the fan-out part of the
hypothesis is **not supported as the mechanism**.

## 31.5 What actually caused the misses

Mean TreeSHAP contribution (log-odds) of the frozen model on W2. Local accuracy
error 8.65e-6.

| feature | W2 TP | W2 FN | W2 FP |
|---|---|---|---|
| source_destination_count | +1.405 | **-1.248** | +1.325 |
| source_destination_seen | +1.451 | +0.140 | +0.922 |
| source_unique_destination_count | +2.789 | +1.881 | +2.574 |
| user_recent_event_count | -0.537 | **-1.149** | +0.602 |
| user_destination_seen | +0.390 | -0.018 | +0.035 |
| destination_recent_event_count | +1.005 | +0.620 | +0.878 |

The missed attacks share two behaviours the W1 training attacks lacked:

1. **Repeat contacts.** Many FNs re-authenticate to a destination the source has
   already reached: `source_destination_count` ranges from 4 to 18, contributing
   -2.1 to -4.0 log-odds. One burst reached counts 14, 15, 16 and 17 within 16
   seconds. The W1 test attacks were almost all first contacts
   (`source_destination_seen=0`).
2. **Busy accounts.** Many FNs use accounts with high recent activity
   (`user_recent_event_count` 129 to 1,091), contributing -2.4 to -4.2 log-odds.
   One FN used an account with 7,285 recent events but that feature contributed
   only -0.88 there. The model learned "busy account" as a benign signal.

**Mechanism of the W2 recall loss:** the model learned a narrow W1 signature —
*first contact, from a fanning-out source, by a quiet account*. The same attacker
host in W2 partly behaves differently (repeat contacts, busy accounts), and those
events score near zero (median W2 attack score 0.0087, below the 0.015060
threshold).

This is a **same-attacker behavioural shift**, not a new-attacker shift. It is a
stronger negative result than the M4.5 hypothesis anticipated: the model fails to
generalise even within a single campaign.

## 31.6 False positives

The 430 W2 FPs match the learned signature: positive contributions from source
fan-out (+2.57), pair novelty (`source_destination_count` +1.33,
`source_destination_seen` +0.92), and quiet accounts
(`user_recent_event_count` +0.60). Their fan-out is mostly moderate (median 31),
with a tail of very high-fan-out benign hosts (p99 3,092).

The alert rate rises about 21x (10 to 215 per hour), while W2's event volume is
only 1.43x W1's (255 vs 178 events per second). Volume alone therefore does not
explain the FP increase. The cause is **not established** by this experiment and
is left open. W1's FP rate itself rests on only 3 events.

## 31.7 Structural finding: cumulative features are position-dependent

Confirmed from `ml/preprocessing/features.py`: the unique-count and pair-count
features (`source_unique_destination_count`, `user_unique_destination_count`,
`destination_unique_source_count`, `user_destination_count`,
`source_destination_count`) **never expire**. They accumulate from the moment the
extractor starts — each window's context start — and only the `*_recent_*`
features use the 3,600 s window.

Consequences:

- The same attacker behaviour yields different absolute values depending on how
  far into the stream it occurs. C17693's fan-out climbs monotonically within each
  window: W1 train 25 to 118, W1 test 129 to 137, W2 21 to 90. The first W2 attack,
  at the very start of W2's emission, had fan-out 21 and was missed with a -4.42
  contribution from that feature alone.
- Trees split on absolute counts, so learned thresholds are tied to stream
  position within a bounded window. A model trained on 3-hour windows would see
  very different magnitudes on an unbounded or full-dataset stream.

This does not invalidate earlier results, which all used the same windowing. It
is a limitation of the M3.4 feature design under bounded windows and must be
carried into M5.

## 31.8 Runtime and memory

W1 build 152.8 s; fit 7.2 s; W2 build 232.3 s; TreeSHAP negligible. Total 395.5 s.
Peak RSS 1,015 MB (290 MB after releasing W1). CPU only.

## 31.9 Limitations

1. **One attacker host in both windows.** "Cross-window" here means one campaign
   observed at two times, not generalisation across attackers. The LANL redteam
   data offers no other source host in these windows.
2. **Only two windows.** A single W1-to-W2 transfer is one observation of
   generalisation, not a distribution.
3. **FN mechanism is attributional.** TreeSHAP shows which features pushed missed
   attacks down. It does not prove that changing those behaviours would have
   produced detection.
4. **Source host attribution** for W2 positives used an unambiguous timestamp
   join to redteam records; all 76 were unambiguous.
5. **FP rate cause** is unresolved (section 31.6).
6. **Feature non-stationarity** (section 31.7) confounds part of the W1/W2
   fan-out comparison.

## 31.10 Conclusion

**RQ4 (selected model):** the model does **not** generalise from W1 to W2. PR-AUC
falls from 0.9517 to 0.0843, recall from 1.00 to 0.47, and the false-positive
rate rises from 10 to 215 per hour. This holds even though both windows contain
the *same* attacker source host.

The M4.5 hypothesis was only partly right. The attacker did not change host. What
changed was behaviour — repeat contacts and busy accounts — which falls outside
the narrow signature learned from W1. Cumulative count features that grow with
stream position add a structural reason why absolute thresholds learned in one
window transfer poorly.

Implications for later modules (recorded, not acted on):

- **M4.7** should consider evaluating each information source (behavioural, graph,
  hybrid) under this same frozen W1-to-W2 protocol. Whether the graph variants
  transfer differently is untested, and within-window ranking alone may not
  predict cross-window behaviour.
- **M5** must report W1 performance only alongside the W2 collapse, the
  single-host nature of both windows, and the feature non-stationarity.


---

# 32. M4.7 ABLATION RESULTS (PARTIAL — W2 COMPARATOR BLOCKED)

Status: IN PROGRESS (2026-09-13). The neural ablation below is complete and
valid. The XGBoost W2 comparator is **not** valid yet (section 33), so the final
M4.7 conclusion is withheld until the W2 correction is approved and run.

## 32.1 Design

Purpose (MODULES.md): determine the value of each information source.

Three information sources reach the neural classifier head:

- **G** — graph embeddings from the GAT over the pre-cutoff history graph
- **B** — the 17 M3.4 temporal/behavioural features
- **T** — the 7 target-event attributes

The M4.4 variants (G+T, B+T, G+B+T) could not value the sources cleanly: all of
them include T, and three of T's attributes are graph-derived "seen before"
flags. Two ablation-only variants complete the design, and M4.4's variants are
unchanged:

| variant | inputs | new in M4.7 |
|---|---|---|
| target_only | T | yes |
| behavioral_no_target | B | yes |
| behavioral_only | B+T | no (M4.4) |
| graph_only | G+T | no (M4.3/M4.4) |
| hybrid | G+B+T | no (M4.4) |

Each source's marginal value is the difference between two variants that differ
in exactly that one input block (tested in `tests/test_ablation.py`).

Protocol:

- The M4.4 configuration is used unchanged, with **no tuning**.
- **3 seeds** per variant (0, 1, 2), because section 29 showed single graph runs
  are unreliable. Seeds estimate variance; they were not searched over.
- **Deterministic CUDA** throughout.
- Each model is trained on W1 TRAIN; the epoch and threshold are selected on W1
  VALIDATION. The behavioural scaler is fitted on W1 TRAIN only and was verified
  unchanged at the end of the run.
- All 15 models were trained with **only W1 in memory**. W1 was released before W2
  was built. W2 batches were relabelled as test data, so the scaler cannot fit on
  them.
- Every frozen model was then scored on every emitted W2 event.

Files: `ml/evaluation/ablation.py`, `scripts/run_ablation.py`,
`tests/test_ablation.py` (19 tests). `ml/models/gnn.py` gained `use_target`,
the two variants, `VARIANT_INPUTS`, and `enable_deterministic_cuda()`, which
`scripts/run_gnn_experiment.py` now imports instead of defining its own copy.

Patch safety: reconstructing the pre-patch `gnn.py` from the exact edit strings
and comparing on CPU gave identical parameter keys, key order, seed-0 initial
weights and forward outputs for all three M4.4 variants.

W1: 157 batches; split counts match section 12 exactly.
W2: 267 batches; **2,174,232 events, 92 positives** — the full W2 emission window
(section 33).

Runtime 2,449 s. Peak RSS 1,874 MB. Peak VRAM 871 MB.

## 32.2 W1 results (valid)

Per seed: W1 test PR-AUC, with validation PR-AUC at the selected epoch in
brackets.

| variant | inputs | params | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| target_only | T | 303 | 0.0079 (0.0025) | 0.0072 (0.0024) | 0.0088 (0.0023) |
| behavioral_no_target | B | 643 | 0.9403 (0.9300) | 0.9018 (0.9077) | 0.7709 (0.9404) |
| behavioral_only | B+T | 881 | 0.9088 (0.9261) | 0.8349 (0.8831) | 0.9222 (0.9321) |
| graph_only | G+T | 10,285 | 0.1022 (0.1385) | 0.2049 (0.0688) | 0.0713 (0.1418) |
| hybrid | G+B+T | 10,863 | 0.9842 (0.6972) | 0.6867 (0.7125) | 0.7682 (0.6625) |

Seed summaries, mean [min, max]:

| variant | W1 validation PR-AUC | W1 test PR-AUC |
|---|---|---|
| target_only | 0.0024 | 0.0080 [0.0072, 0.0088] |
| behavioral_no_target | 0.9260 | 0.8710 [0.7709, 0.9403] |
| behavioral_only | 0.9138 | 0.8886 [0.8349, 0.9222] |
| graph_only | 0.1164 | 0.1261 [0.0713, 0.2049] |
| hybrid | 0.6907 | 0.8130 [0.6867, 0.9842] |
| xgboost:standard (section 14) | 0.9332 | 0.9517 |

## 32.3 Frozen W2 results, neural variants (valid, full W2)

Base rate on W2 is 92 / 2,174,232 = 4.2e-5. FP per hour uses the 2-hour emission
window.

| variant | inputs | W2 PR-AUC mean [min, max] | W2 recall mean | W2 FP/h mean [min, max] |
|---|---|---|---|---|
| target_only | T | 0.0009 [0.00087, 0.00095] | 0.094 | 2,923 [2,841, 3,056] |
| behavioral_no_target | B | 0.1451 [0.1269, 0.1547] | 0.301 | 90 [63, 105] |
| behavioral_only | B+T | 0.1108 [0.0618, 0.1388] | 0.315 | 130 [117, 140] |
| graph_only | G+T | 0.0320 [0.0163, 0.0462] | 0.246 | 547 [150, 1,320] |
| hybrid | G+B+T | 0.1306 [0.0764, 0.2173] | 0.573 | 1,551 [272, 3,124] |

**The XGBoost W2 figure in section 31.3 (PR-AUC 0.0843) is NOT comparable with this
table.** It was computed on a truncated W2 (section 33) and is deliberately left
out.

## 32.4 Marginal value of each source

Difference of seed means, in PR-AUC.

| contribution | comparison | W1 test | W2 |
|---|---|---|---|
| graph, given target | graph_only - target_only | +0.118 | +0.031 |
| graph, given behavioural+target | hybrid - behavioral_only | -0.076 | +0.020 |
| behavioural, given target | behavioral_only - target_only | **+0.881** | **+0.110** |
| behavioural, given graph+target | hybrid - graph_only | **+0.687** | **+0.099** |
| target, given behavioural | behavioral_only - behavioral_no_target | +0.018 | -0.034 |

## 32.5 Findings (neural ablation)

1. **The behavioural features are the only information source with a large,
   consistent value.** Their marginal contribution is large in both contexts and
   on both windows. In every seed on both windows, every variant containing B
   beats every variant without it: on W1 the worst B-containing model scored
   0.687 against the best non-B model's 0.205; on W2, 0.062 against 0.046.
2. **The graph adds nothing distinguishable from seed noise once the behavioural
   features are present.** Hybrid vs behavioral_only: -0.076 on W1 and +0.020 on
   W2, and the seed ranges overlap on both windows. Alone, the graph carries
   weak signal (+0.118 on W1 over target_only), far below B.
3. **The 7 target attributes add nothing.** Alone they are near random (W1
   0.008). Added to B they give +0.018 on W1 (within noise) and -0.034 on W2.
   The graph-derived "seen before" flags in T do not help.
4. **Graph-bearing variants are the least stable.** Hybrid W1 test spans 0.687 to
   0.984 across seeds. graph_only spans 0.071 to 0.205.
   - Hybrid seed 0's 0.984 exceeds XGBoost's W1 0.9517, but it is one draw that
     does not reproduce (section 34). Its validation PR-AUC (0.697) is below
     every behavioural variant's in every seed (0.883 to 0.940). Under the
     validation-based selection rule it would not be chosen, so it is not
     evidence that hybrid is better.
5. **Validation-selected thresholds do not transfer to W2 for graph-bearing
   models.** Hybrid's W2 alert rate ranges from 272 to 3,124 FP/h across seeds.
   Its high mean W2 recall (0.57) comes from over-alerting, not better ranking:
   its W2 PR-AUC seed range (0.076 to 0.217) overlaps both behavioural
   variants'. behavioral_no_target has the
   lowest and most stable W2 alert rate (63 to 105 FP/h).
6. **All variants collapse from W1 to W2,** consistent with section 31. Even the
   best W2 mean PR-AUC (0.145, behavioral_no_target) is roughly 6x below the
   corresponding W1 value.

## 32.6 Seed-0 reproduction of section 28.5

| variant | section 28.5 (M4.4) | M4.7 seed 0 | reproduced |
|---|---|---|---|
| behavioral_only | 0.908838 | 0.908838 | **YES** (same threshold 0.996246, epoch 17, TP/FP/FN) |
| graph_only | 0.140273 | 0.102192 | **NO** |
| hybrid | 0.844415 | 0.984160 | **NO** |

The patch was ruled out (section 32.1). The cause is analysed in section 34.

---

# 33. DEFECT: W2 WAS SILENTLY TRUNCATED (affects sections 13, 16, 31)

Discovered 2026-09-13 during M4.7, when the graph pipeline's W2 counts did not
match section 13.

## 33.1 Cause

`build_window_dataset` (M3.6) passes `max_events=DEFAULT_MAX_WINDOW_EVENTS`
(3,000,000) to `MLDataset.from_event_source`. That loop counts **context events
toward the cap** and stops with a silent `break` — no error and no warning. The
existing test `test_window_build_is_bounded_by_max_events` pins this stopping
behaviour.

The guard's documented purpose is to stop a mis-specified window from turning
into a full-dataset run. It was never meant to shorten a correctly specified
window.

## 33.2 Evidence (direct count of W2, CPU stream)

| quantity | value |
|---|---|
| W2 context events | 1,167,143 |
| W2 emitted events, full window | **2,174,232** |
| total read | 3,341,375 (above the 3,000,000 cap) |
| emitted events inside the cap | **1,832,857** — exactly the section 13 figure |
| timestamp of event #3,000,000 | 1077310 (W2 emission ends at 1078448) |
| redteam records after that timestamp | **16** |
| W2 redteam records at or before t=1077293 (last tabular positive) | 76 |

The truncated build dropped the last ~19 minutes of W2's 2-hour emission window:
341,375 events and 16 attacks.

## 33.3 Consequences for the record

- **Section 13** ("Emitted 1,832,857; exact positives 76 / 92 redteam records") is
  a truncation artifact. The defined W2 window contains **2,174,232 emitted
  events, and all 92 redteam records match an authentication event exactly.**
- **Section 31.2's statement that 16 records "do not match an authentication event
  exactly" is wrong.** They were simply outside the truncated data.
- **Sections 16 (M4.1) and 31 (M4.6)** evaluated the frozen XGBoost on the first
  84% of W2 only. Their numbers describe that truncated set, not the defined W2.
  - Because cumulative fan-out grows with stream position (section 31.7), the
    missing tail attacks are plausibly the most detectable for the frozen model,
    so the corrected W2 result may differ materially.
  - The identity-overlap analysis in section 31.4 used all 92 redteam records and
    is unaffected. The fan-out profiles and TP/FN analysis used only 76 positives
    and are affected.
- **W1 is unaffected.** It reads 1,925,429 events, below the cap, so every W1
  result (M3.6 to M4.5, and the M4.7 W1 ablation) stands.
- **The M4.7 neural W2 results (section 32.3) are on the full W2.** The graph
  pipeline has no event cap.

All earlier sections are left unchanged, per RESEARCH_CONSTRAINTS section 23.
Corrections will be appended once the fix is approved and run.

## 33.4 Proposed correction (awaiting user approval)

1. Make exceeding `max_events` in a window build **raise** instead of truncating,
   and raise `DEFAULT_MAX_WINDOW_EVENTS` enough to cover the full defined W2
   (3,341,375 events). Update the M3.6 test that pins silent truncation.
2. Re-run `scripts/run_cross_window.py` (M4.6) on the full W2, and append corrected
   results with the truncated figures preserved and explained.
3. Complete M4.7 with a valid full-W2 XGBoost comparator.


---

# 34. AMENDMENT TO SECTION 29: DETERMINISTIC CUDA DOES NOT MAKE GAT TRAINING REPRODUCIBLE ON REAL DATA

Discovered 2026-09-13 while reconciling the M4.7 seed-0 reproduction failures
(section 32.6).

## 34.1 What section 29 claimed

Section 29 said `--deterministic` was "verified to produce bitwise-identical
runs". That verification used a tiny in-memory fixture (60 events), with two runs
in identical order. It was never checked on real W1 data.

## 34.2 Diagnostics

All diagnostics used identical code (the M4.7 patch was verified identical on CPU,
section 32.1), identical W1 batches (behavioral_only reproduced exactly),
deterministic CUDA, and seed 0.

**(A) Real W1, M4.4 conditions.** A fresh process trained graph_only first, the
exact M4.4 order (`scripts/run_gnn_experiment.py --deterministic --variants
graph_only`):

- **Epochs 1-3 are identical** to the M4.4 deterministic run: train loss 1.09230,
  0.78747, 0.38287; validation PR-AUC 0.002239, 0.007126, 0.094998.
- **The run diverges afterwards:**

| quantity | M4.4 deterministic run | diagnostic (A) |
|---|---|---|
| best epoch | 20 | 4 |
| validation PR-AUC | 0.1404 | 0.1274 |
| threshold | 0.999047 | 0.999956 |
| W1 test PR-AUC | 0.1403 | 0.0016 |
| W1 test TP / FP / FN | 6 / 54 / 5 | 0 / 26 / 11 |

**(B) Synthetic fixture** (3,000 events, 256-event blocks, 3 epochs). Four fresh
processes: graph_only trained first (twice), and after nine graph-free models
(twice). **All four final-weight digests and every per-epoch loss were
identical.**

## 34.3 Conclusions

1. **Process history is not the cause.** Diagnostic (B) shows no order dependence,
   and (A) diverges even in M4.4's exact order.
2. **The M4.7 patch is not the cause** (section 32.1).
3. **On real W1 data, deterministic-mode GAT training on this GPU is not
   reproducible across processes.** Runs agree bitwise for the first few epochs
   and then drift. The drift is amplified by unstable validation PR-AUC and
   validation-based epoch selection. The small fixture section 29 relied on does
   not trigger it. The kernel responsible was not identified; it must be one that
   PyTorch's deterministic mode does not flag.
4. **Graph-free variants are reproducible.** behavioral_only reproduced section
   28.5 exactly on real data.

Observed W1 test PR-AUC of graph_only, seed 0, identical configuration, across
five runs: **0.0775** (M4.3), **0.0333** (M4.4 default), **0.1403** (M4.4
deterministic), **0.1022** (M4.7 ablation), **0.0016** (diagnostic A).

## 34.4 Consequences

- **Every graph-bearing figure in the record** (sections 18, 28, 32) is a single
  draw from a run-to-run distribution, even at a fixed seed. Draw conclusions only
  from ranges and orderings that hold consistently. The M4.7 findings in section
  32.5 are of that kind; the individual graph-bearing cells are not reproducible.
- **Section 28.5's "canonical deterministic run" is not canonical.** Section
  28.7's conclusion (hybrid < behavioral_only on W1) is still supported by the
  M4.7 seed ranges, but its specific figure of 0.8444 is not stable.
- **Keep `--deterministic` and `enable_deterministic_cuda()`.** They make
  graph-free runs and small graphs reproducible. They must not be described as
  guaranteeing reproducible GAT training on real data. M4.7 ran with the flag on
  throughout.
- **If exact graph-bearing figures are needed in M5,** CPU training is the
  candidate route. It is reproducible on the unit-test fixture, but unverified on
  real data, and would need its own real-data check before being relied on.


---

# 35. CORRECTED CROSS-WINDOW RESULT ON THE FULL W2 (supersedes sections 16 and 31.3)

Status: COMPLETE (2026-09-17). Approved correction of the defect in section 33.

## 35.1 The fix

`ml/preprocessing/ml_window.py`:

- A window build that reaches `max_events` now raises the new
  `WindowTruncationError` instead of returning a silently shortened dataset.
  Equality counts as truncation: a stream that stopped exactly at the ceiling
  cannot be shown to have covered the window.
- `DEFAULT_MAX_WINDOW_EVENTS` raised from 3,000,000 to 4,000,000, which covers
  the largest defined window (W2 reads 3,341,375 events).
- `MLDataset` semantics are unchanged; it still processes a bounded prefix, which
  M3.5 depends on. Only the window build demands completeness.
- `tests/test_ml_window.py`: the test that pinned silent truncation was replaced
  by one asserting the build refuses to truncate, plus one asserting a build below
  the ceiling still succeeds.

Full suite after the fix: 487 passed, 1 skipped.

## 35.2 Corrected W2 evaluation

Same frozen detector as M4.6 -- fitted on W1 TRAIN, threshold 0.015060 selected on
W1 VALIDATION, same model digest `0e25fdf1c66b15e7...`, verified unchanged at the
end of the run. W1 test reproduced section 14 exactly. Only the W2 data changed.

| quantity | truncated W2 (sections 16, 31.3) | **corrected full W2** |
|---|---|---|
| events | 1,832,857 | **2,174,232** |
| positives | 76 | **92** |
| PR-AUC | 0.0843 | **0.1257** |
| ROC-AUC | 0.9970 | 0.997528 |
| Precision | 0.0773 | 0.0817 |
| Recall | 0.4737 | 0.4891 |
| F1 | 0.1328 | 0.1400 |
| TP | 36 | 45 |
| FP | 430 | 506 |
| TN | 1,832,351 | 2,173,634 |
| FN | 40 | 47 |
| predicted positive | 466 | 551 |
| FP per hour | 215.0 | 253.0 |

**The 16 attacks that truncation had removed: 9 detected, 7 missed.** Their 56%
detection rate is above the 47% of the rest of W2, which matches section 31.7:
cumulative fan-out grows with position in the window, so late attacks look more
anomalous to a model trained on high fan-out values.

W1 test is unchanged: PR-AUC 0.9517, TP 11 / FP 3 / TN 200,822 / FN 0, 10.0 FP/h.

## 35.3 What the correction changes, and what it does not

**Changed** (all figures above): W2 counts, every W2 metric, and the alert rate.
The alert-rate gap between windows widens slightly, from 21.5x to **25.3x** (10.0
to 253.0 FP/h), while W2's event volume is only 1.70x W1's, so volume still does
not explain it (section 31.6 remains open).

**Unchanged:**

- **Attacker identity** (section 31.4). All 92 W2 redteam records still come from
  the single source host C17693; source-host Jaccard with W1 is 1.000. The
  analysis already used all 92 records.
- **Fan-out still fails to separate hits from misses.** Corrected profile for
  `source_unique_destination_count`:

| group | n | min | p25 | median | p75 | p99 | max |
|---|---|---|---|---|---|---|---|
| W2 positives | 92 | 21 | 47 | 63 | 83 | 97 | 97 |
| W2 TP | 45 | 30 | 51 | 63 | 84 | 96 | 96 |
| W2 FN | 47 | 21 | 46 | 65 | 83 | 97 | 97 |
| W2 FP | 506 | 22 | 25 | 32 | 48 | 3,091 | 4,565 |
| W2 negatives | 2,174,140 | 0 | 6 | 8 | 13 | 69 | 6,456 |

- **The mechanism of the misses** (section 31.5). Corrected mean TreeSHAP
  contributions, in log-odds:

| feature | W2 TP | W2 FN | W2 FP |
|---|---|---|---|
| source_destination_count | +1.435 | **-1.301** | +1.316 |
| source_destination_seen | +1.418 | +0.113 | +0.931 |
| source_unique_destination_count | +3.272 | +2.176 | +2.615 |
| user_recent_event_count | -0.432 | **-1.137** | +0.612 |
| user_unique_destination_count | +1.068 | +0.750 | +0.734 |

  Misses are still repeat contacts to already-reached destinations and activity by
  busy accounts -- behaviour outside the narrow W1 signature.
- **Section 31.7** (cumulative features are position-dependent) and section 31.10's
  conclusion. The collapse is now 0.9517 to 0.1257 rather than to 0.0843.

**Corrected statements of record:**

- Section 13's W2 counts (1,832,857 emitted; "76 / 92" exact positives) are a
  truncation artifact. The defined W2 holds **2,174,232 emitted events, and all 92
  redteam records match an authentication event exactly.**
- Section 31.2's claim that 16 records "do not match an authentication event
  exactly" is withdrawn.
- Section 16 (M4.1) and section 31.3 describe the truncated W2 and are superseded
  by this section. They are preserved per RESEARCH_CONSTRAINTS section 23.

Runtime 454.5 s; peak RSS 1,099 MB; CPU only.

---

# 36. M4.7 COMPLETION: FINAL ABLATION WITH THE CORRECTED W2

Status: COMPLETE (2026-09-17). Completes section 32, whose W2 comparator was
blocked by the section 33 defect.

The neural ablation in section 32 already used the full W2 (2,174,232 events, 92
positives), so those figures stand unchanged. Only the XGBoost comparator was
missing; section 35 supplies it.

## 36.1 Final comparison

Neural rows are the mean over 3 seeds with [min, max]. XGBoost is deterministic
and exact. All W2 figures are on the corrected full W2 at each model's own
validation-selected threshold.

| model | inputs | W1 test PR-AUC | W2 PR-AUC | W2 recall | W2 FP/h |
|---|---|---|---|---|---|
| xgboost:standard | B | **0.9517** | 0.1257 | 0.489 | 253 |
| behavioral_only | B+T | 0.8886 [0.835, 0.922] | 0.1108 [0.062, 0.139] | 0.315 | 130 |
| behavioral_no_target | B | 0.8710 [0.771, 0.940] | **0.1451** [0.127, 0.155] | 0.301 | **90** |
| hybrid | G+B+T | 0.8130 [0.687, 0.984] | 0.1306 [0.076, 0.217] | 0.573 | 1,551 |
| graph_only | G+T | 0.1261 [0.071, 0.205] | 0.0320 [0.016, 0.046] | 0.246 | 547 |
| target_only | T | 0.0080 [0.007, 0.009] | 0.0009 | 0.094 | 2,923 |

## 36.2 Value of each information source

1. **Temporal/behavioural features: the only source with real value.** Adding them
   is worth +0.881 PR-AUC on W1 and +0.110 on W2 over target attributes alone, and
   +0.687 / +0.099 over graph plus target. In every seed and on both windows, every
   model containing them beat every model without them.
2. **Graph: no value once the behavioural features are present.** Hybrid minus
   behavioral_only is -0.076 on W1 and +0.020 on W2, with overlapping seed ranges
   on both. Alone the graph is weak (W1 0.126), it is the least stable arm
   (section 34), and it alerts far more often (547 to 1,551 FP/h against 90 to 130
   for behavioural variants).
3. **Target attributes: no value.** Near random alone (W1 0.008); +0.018 on W1
   (inside noise) and -0.034 on W2 when added to the behavioural features. The
   graph-derived "seen before" flags they contain do not help.
4. **Model family matters less than representation, and only on W1.** XGBoost beats
   the best neural behavioural variant by 0.063 PR-AUC on W1 (0.9517 vs 0.8886),
   but on W2 it lands inside the neural behavioural range (0.1257 against 0.111 to
   0.145). No representation and no model family generalises.
5. **Nothing rescues cross-window generalisation.** The best W2 PR-AUC of any model
   is 0.145, roughly six times below its own W1 figure. Hybrid reaches the highest
   W2 recall (0.573) only by alerting 1,551 times an hour, and its W2 PR-AUC is
   indistinguishable from the behavioural variants'.

## 36.3 Answer to RQ2 and RQ3

**RQ2 (does a temporal graph representation improve detection?) -- No.** Alone it is
far weaker than temporal/behavioural features on both windows.

**RQ3 (does combining graph with behavioural features improve performance?) -- No.**
The combination is no better than the behavioural features alone, within seed
noise, on either window, and it costs an order of magnitude in false alerts.

## 36.4 Limitations

1. **Graph-bearing figures are not reproducible** (section 34). Their means and
   ranges are usable; individual cells are not.
2. **Three seeds** give a coarse variance estimate.
3. **11 W1 test positives and 92 W2 positives**, all from one attacker host
   (sections 30.8 and 31.4), so these are two observations of one campaign.
4. **One configuration per variant**, fixed in advance and never tuned. A larger or
   differently designed graph model might do better; that was not explored.
5. **Cumulative features remain position-dependent** (section 31.7).

## 36.5 Recommendation for M5

Report the behavioural feature set as the practical detector, XGBoost as its
strongest realisation on W1, and the cross-window collapse as the central result.
The graph representation did not pay for itself on this data, and that should be
stated as a finding rather than omitted. Carry the section 33 defect, the section
34 reproducibility limit and the single-campaign caveat into the final write-up.


---

# 37. END-TO-END VERIFICATION (2026-09-18)

## 37.1 Scope

Every entry point in `scripts/` was run on real data under a bounded
configuration, and every deterministic output was compared with the record.

- No full-dataset run was performed.
- Raw LANL files are unchanged (auth.txt.gz 7,626,505,158 B; redteam.txt.gz
  4,846 B; timestamps unchanged since August).
- All runs piped their output, which is the condition that used to trigger the
  Windows encoding crash.

## 37.2 Issues found and fixed

| file | issue | fix |
|---|---|---|
| `scripts/run_lanl_baseline.py` | `--limit` defaulted to the **whole 7.6 GB dataset**, and `--smoke-test` then "proceeded to full run"; it scanned all 749 redteam records per event (O(N x 749)); it printed a non-ASCII check mark, crashing when output was redirected (the crash preserved in `baseline_results.txt`) | Default bound is the 1,000,000-event M3.5 development bound; a full run needs `--full-dataset`; `--limit <= 0` is rejected; exact-key index that preserves duplicate records (749 records, 715 keys); ASCII output |
| `scripts/project_health_check.py` | Still asserted that PyTorch and PyG were **not** installed -- an M0 rule overturned in M4.3 -- so it reported 2 FAILs | Verifies the versions pinned in `requirements-gnn.txt`, now a required file |
| `scripts/smoke_test_lanl_adapter.py` | Non-ASCII output crashed when piped or redirected | ASCII output |
| `scripts/inspect_lanl_dataset.py` | Hard-coded output overwrote `docs/lanl_inspection_results.json` | `--output` option (default unchanged); writes UTF-8 |
| `scripts/run_ablation.py` | Still quoted the truncated W2 after section 35: its W2 count check printed "NO", and its XGBoost comparator row showed 0.0843 / 0.4737 / 215 | Corrected to (2,174,232, 92) and the section 35.2 figures 0.1257 / 0.4891 / 253.0 |
| `scripts/run_gnn_experiment.py` | Banner said M4.3 | Now says M4.3/M4.4 |

The index change in `run_lanl_baseline.py` was verified on real data. On 159,499
W1 attack-period events, including 19 redteam matches, it returned exactly the
same records as the original per-event scan, with 0 mismatches. The default
1,000,000-event prefix contains no redteam activity, so it could not test this.

## 37.3 Reproduction matrix

| entry point | module | compared with | result |
|---|---|---|---|
| `pytest -q` | all | -- | 487 passed, 1 skipped |
| `pip check` | M0 | -- | no broken requirements |
| `package_check.py` | M0 | -- | PASS 12/12 |
| `project_health_check.py` | M0 | -- | PASS 10/10 |
| `demo_temporal_graph.py` | M2.3 | render determinism | deterministic; pixel-identical |
| `inspect_lanl_dataset.py` | M1/M3 | `docs/lanl_inspection_results.json` | identical |
| `smoke_test_lanl_adapter.py` | M3.1 | -- | PASS (1k events: 1,455 nodes / 2,000 edges; 10k: 4,165 / 20,000) |
| `run_lanl_baseline.py` | M3.3 | -- | 1,000,000 events in 146 s; 0 redteam in the prefix, as expected |
| `smoke_test_graph_data.py` | M4.2 | M4.2 record | 10,461 nodes / 100,000 edges; all checks pass |
| `run_gnn_experiment.py`, XGBoost | M4.0 | sections 14, 15 | **exact**: 0.951671, TP 11 / FP 3 / FN 0 at 0.015060; class-weighted 0.858830, TP 9 / FP 2 / FN 2 |
| `run_gnn_experiment.py`, behavioral_only | M4.4 | section 28.5 | **exact**: 0.908838 at 0.996246, epoch 17 |
| `run_explainability.py` | M4.5 | section 30 | **exact**: validation PR-AUC 0.9332; local accuracy 2.10e-5 / 2.05e-5; Spearman 0.9975 / 0.1005; deletion k=1 test drop 10.358 |
| `run_cross_window.py` | M4.6 | section 35 | **exact**: W2 2,174,232 / 92; PR-AUC 0.1257; TP 45 / FP 506 / FN 47; digest unchanged |
| `run_ablation.py`, seed 0, graph-free | M4.7 | section 32 | **exact**: target_only W1 0.007945 / W2 0.000866; behavioral_no_target W1 0.940307 / W2 0.153842 |

Graph-bearing variants were not re-run. Section 34 established that they are not
reproducible run to run, so a re-run could not confirm any specific figure.

## 37.4 Incident during verification

The first harness for the demo check redirected the output paths on the dictionary
that `runpy.run_path` returns. That is a copy of the module globals, so the redirect
did not reach `main()`.

The demo therefore wrote to its normal location and overwrote
`experiments/figures/synthetic_multihop_graph.png` and
`synthetic_multihop_timeline.png`. These are gitignored, generated M2.3 figures.
The August originals cannot be recovered. The corrected harness shows rendering is
deterministic and the current files are pixel-identical to what the code produces.
No data or source file was affected.

## 37.5 Open items needing a user decision (left unchanged)

1. **`README.md` is stale.** It describes the M0 state ("no application or ML code
   has been written yet"). It also uses a different title and scope --
   uncertainty, risk scoring, attack-path reconstruction, backend and dashboard --
   none of which is on the MODULES.md roadmap. Rewriting it is an M6/M7 framing
   decision.
2. **`CLAUDE.md` and `MODULES.md` tell readers to read `PROJECT_SPEC.md`, which does
   not exist.** The specification lives inside `PROJECT_STATE.md`, whose first line
   is `# PROJECT_SPEC.md`.
3. **`baseline_results.txt`** is an untracked UTF-16 crash log from the pre-fix
   baseline script. It is now obsolete.
4. **Empty scaffold directories** from M0 (`ml/uncertainty`, `ml/risk`,
   `ml/path_analysis`, `backend/*`, `frontend`, `notebooks`, `checkpoints`) match the
   old README scope, not the current roadmap. The health check requires them.
5. **`docs/*.md`** (M0/M1) were not reviewed for currency.


---

# 38. SOC ANALYST DASHBOARD (2026-09-19)

Status: BUILT, at the user's request. It is not a MODULES.md module: it belongs to
M6 ("final visualisations") and was built ahead of M5 because the user asked for
it. No detection or ML logic was modified; everything served comes from the
existing research code in `ml/`.

## 38.1 What existed before

- `frontend/` and `backend/` held only `.gitkeep` placeholders.
- No `package.json` existed anywhere.
- No outputs were persisted: every experiment script ends "Nothing was written to
  disk".

The dashboard stack was therefore built new: a Next.js frontend and an aiohttp
backend. aiohttp was already installed, so the backend adds **no new Python
dependency**.

## 38.2 Architecture

1. **Offline export** (`scripts/build_dashboard_data.py`,
   `backend/services/export.py`). It calls the existing pipeline unmodified:
   - M3.6 windows;
   - `fit_frozen_detector` (standard M4.0 XGBoost: W1 TRAIN, threshold on W1
     VALIDATION, frozen);
   - `build_window_dataset` and `evaluation_window` for W2 features, labels and
     frozen scores.

   A second bounded W2 stream recovers user and host identities. Each of the
   2,174,232 rows is checked against the dataset's `event_id` before it is
   accepted. Output goes to `data/processed/dashboard/` (gitignored, 75 MB):
   `model.ubj`, `w1.npz`, `w2.npz`, `manifest.json`. Runtime is 536 s.
2. **Backend API** (`backend/`, aiohttp, bound to 127.0.0.1:8000;
   `scripts/run_dashboard_api.py`).
   - At start-up it re-scores 1,063 stored rows (all 551 alerts plus an even
     spread) with the loaded model, and **refuses to serve** if any score differs.
   - It serves a replay clock with an SSE stream, alerts with triage, per-event
     TreeSHAP, investigation context, graph slices, experiments, pipeline
     statistics and the Demo Lab.
3. **Frontend** (`frontend/`: Next.js 15.5, React 19, TypeScript, Tailwind 3,
   cytoscape). Seven pages: Overview, Network Graph, Investigation, Alerts,
   Experiments, Pipeline, Demo Lab.

## 38.3 Provenance rules the dashboard enforces

- **"Live" is a replay** of the real, historical W2 events on an analyst-controlled
  clock (1x to 300x, seek, restart). The UI says REPLAY. Every replay-backed route
  is gated by the clock, so no alert, related event or graph edge after the replay
  position is ever served.
- **The served detector** is the frozen standard XGBoost, the model selected in
  M4.5. Neural and hybrid results are shown only as recorded values, cited to
  sections 15, 28.5, 32.4 and 36.1, with the section 34 reproducibility caveat.
- **Computed figures** (Overview and Experiments) are recomputed from the export's
  real scores. PR and ROC curves are exact step corners, one per positive.
- **Research ground truth** (the exact redteam match) is shown and labelled as
  unavailable to a production SOC.
- **Heuristics are labelled.** Severity is a fixed presentation band on the score,
  not a model output. The movement trace is a heuristic that follows the alert's
  account; the project has no path-reconstruction model. The source fan-out view
  lists real first-contact events.
- **Demo Lab events are synthetic** (RESEARCH_CONSTRAINTS section 3).
  - They run through the real components -- M2.0 CanonicalEvent, M3.4 extractor,
    M2.1 TemporalGraph, the frozen model, TreeSHAP -- in an isolated session that
    shares nothing mutable with the replay store (tested).
  - They perform no network or system activity and are never counted in any alert
    or metric.
  - Every response carries `synthetic: true`, and the page shows a permanent banner.

## 38.4 Verification

- The export reproduces the record exactly:
  - W1 test: TP 11 / FP 3 / FN 0, PR-AUC 0.9517 (section 14);
  - W2: TP 45 / FP 506 / FN 47, PR-AUC 0.1257 (section 35);
  - threshold 0.015060, model digest `0e25fdf1c66b15e7...`, as in M4.6.
- Start-up integrity: 1,063 rows re-scored identically; the model digest survives
  the UBJ save/load round trip.
- Tests: 33 new (`tests/test_dashboard_export.py`,
  `tests/test_dashboard_backend.py`), on synthetic mini-exports only. They cover
  integrity and tamper detection, replay gating, filters, triage, TreeSHAP local
  accuracy, graph aggregation, account trace, fan-out, exact curves, lab mechanics
  and isolation, API routes, CORS and SSE. Full suite: 520 passed, 1 skipped.
- Frontend: `tsc` clean; `next build` succeeds (8 routes, about 115 KB of
  first-load JS each).
- All seven pages were exercised in a browser against the live backend on real
  data:
  - Alerts filtered to redteam only return exactly 10 alerts, all from C17693.
  - A live alert arrived during the check and appeared in the filtered list.
  - The Investigation TreeSHAP for a C17693 alert reproduces bias + contributions =
    margin (5.45%).
  - The SSE stream reconnected on its own after a backend restart.
- Console errors appeared only while the backend was deliberately restarted.

## 38.5 Observation from the Demo Lab (synthetic; mechanics only)

In a default fan-out sweep (one source, 40 new destinations, 20 s apart) the
frozen model stays below the threshold for the first 22 hops, scoring 0.00007 to
0.005. It crosses at hop 22 (0.758) and reaches 0.97. TreeSHAP attributes this
mainly to `source_unique_destination_count` (+2.97 at fan-out 39).

This agrees with section 30, but it is synthetic, out-of-distribution evidence of
pipeline mechanics, not a detection result.

## 38.6 How to run

1. `.\.venv\Scripts\python.exe scripts/build_dashboard_data.py` -- once; about 9
   minutes.
2. `.\.venv\Scripts\python.exe scripts/run_dashboard_api.py` -- options: `--speed`,
   `--paused`.
3. `npm --prefix frontend install`, then `npm --prefix frontend run dev`, and open
   http://localhost:3000.

## 38.7 Limitations

1. The replay is historical LANL data, not live telemetry; only W2 is replayed.
2. It is a single-analyst local tool: no authentication, localhost only. Triage
   state persists in `data/processed/dashboard/triage.json`.
3. Graph views aggregate each window and cap nodes at 400. A 5-minute W2 window
   holds about 90,000 events and about 6,800 active hosts.
4. The movement trace is heuristic.
5. Neural and hybrid models are not served. They underperform, and graph-bearing
   runs are not reproducible (sections 34 and 36).
6. The README, `docs/architecture.md` and section 37.5's open items still describe
   the older M0 scope.
