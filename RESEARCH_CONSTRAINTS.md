# RESEARCH_CONSTRAINTS.md

## 1. CORE RULE

This is a research project, not merely a software project.

Implementation decisions may be changed or improved, but the established research methodology must not be changed silently.

If a methodological change is genuinely necessary, explain why and stop for user approval before changing it.

---

## 2. PRIMARY DATASET

All research evidence and final experimental results must come from the original LANL Comprehensive Multi-Source Cyber-Security Events dataset.

Current research files:

data/raw/lanl/auth.txt.gz
data/raw/lanl/redteam.txt.gz

The raw LANL files are read-only.

Never modify, overwrite, truncate, or replace them.

Do not substitute another dataset without explicit approval.

---

## 3. SYNTHETIC DATA

Synthetic data is permitted ONLY for:

- unit tests
- software fixtures
- deterministic test scenarios
- debugging
- validating algorithms before using real LANL data

Synthetic data must NEVER be used for:

- final metrics
- model comparison
- research conclusions
- claims about LANL behavior
- replacing missing LANL evidence

Always clearly distinguish synthetic tests from real LANL experiments.

---

## 4. GROUND TRUTH

LANL redteam records are the source of known attack ground truth.

Current matching rule:

Exact match on:

(timestamp, user, source_host, destination_host)

Do not change exact matching to:

- fuzzy matching
- partial matching
- nearest timestamp matching
- source/user-only matching
- manually inferred labels

Do not create additional attack labels simply to increase the number of positives.

If a redteam record does not exactly match an authentication event, preserve that fact and report it.

---

## 5. LABEL LEAKAGE

Redteam information must be used ONLY for evaluation labels.

Do not use redteam information as:

- input features
- graph features
- historical state
- training metadata
- model inputs
- threshold features
- preprocessing statistics

The model must not be given knowledge that an event is redteam activity.

---

## 6. TEMPORAL CAUSALITY

The project is explicitly temporal.

For an event occurring at time T, model features and graph history may use only information strictly earlier than T.

Never allow:

- future events
- future graph edges
- future node statistics
- future feature values
- later events in the same prediction history

to influence an earlier prediction.

Same-timestamp events must not become historical information for each other unless an explicitly justified ordering is defined by the methodology.

---

## 7. DATA SPLITTING

Evaluation must respect chronological ordering.

The general structure is:

TRAIN → VALIDATION → TEST

in time order.

Do not randomly shuffle events across train/validation/test.

Do not allow temporal overlap that causes future information to enter training.

---

## 8. TRAINING / VALIDATION / TEST

Training data is used to fit models.

Validation data may be used for:

- model selection
- threshold selection
- permitted configuration decisions

Test data is used ONLY for final evaluation.

Do not:

- tune hyperparameters using test results
- select thresholds using test results
- repeatedly modify the model based on test performance
- train on test data

Once the final threshold is selected from validation, freeze it before test evaluation.

---

## 9. W1 AND W2

W1 is the current primary development/evaluation window.

W2 is reserved for temporal cross-window generalization evaluation.

Do not train or tune on W2 when performing a generalization experiment.

For cross-window evaluation:

1. train using the designated training data
2. select configuration/threshold using allowed validation data
3. freeze the model and threshold
4. evaluate on W2

W2 results must not be used to improve the model before reporting the experiment.

---

## 10. CURRENT W1

W1:

Context:
760506–764106

Emission:
764106–771306

Current W1 dataset:

Context events:
645,665

Emitted events:
1,279,764

Chronological split:

Train:
892,424 events
69 positives

Validation:
186,504 events
15 positives

Test:
200,836 events
11 positives

Important:

Only 11 positives exist in the current W1 test set.

Therefore, test metrics must be interpreted cautiously and should not be presented as statistically definitive.

---

## 11. CURRENT W2

W2:

Context:
1067648–1071248

Emission:
1071248–1078448

Emitted events:
1,832,857

Exact positives:
76 / 92 redteam records

W2 is primarily intended for generalization evaluation.

---

## 12. FEATURE LEAKAGE

All temporal/behavioral features must be causal.

Feature calculations must use only information available before the target event.

Do not calculate statistics using the complete dataset and then apply them to earlier events.

Examples of prohibited leakage:

- future event counts
- future destination frequencies
- future node degrees
- future user behavior
- test-derived normalization
- validation/test-derived feature statistics

---

## 13. FEATURE SCALING / PREPROCESSING

If normalization, scaling, encoding statistics, thresholds, or similar learned preprocessing is required:

Calculate them using TRAIN only.

Apply the frozen values to:

- validation
- test
- W2

Never calculate preprocessing statistics from validation/test/W2 and feed them back into training.

---

## 14. GRAPH TEMPORALITY

The graph must respect the same causal temporal rules as the feature pipeline.

For an event at time T:

Only graph structure available before T may be used as historical context.

Do not construct a graph using future events and then use it to predict earlier events.

---

## 15. CLASS IMBALANCE

Lateral movement detection is highly imbalanced.

Primary ranking metric:

PR-AUC

Also report:

- Precision
- Recall
- F1
- TP
- FP
- TN
- FN
- ROC-AUC
- threshold
- predicted positives

ROC-AUC must not be presented as the sole or primary evidence of performance.

False positives are important because enterprise authentication datasets contain very large numbers of benign events.

---

## 16. BASELINE COMPARISON

The project must maintain a meaningful comparison between:

1. temporal/behavioral baseline
2. graph-based model
3. hybrid model where applicable

Existing results must not be overwritten simply because a later model performs better or worse.

Poor performance is a valid research result.

---

## 17. EXPERIMENT DISCIPLINE

For every experiment:

1. Define the hypothesis.
2. Use the appropriate real LANL data.
3. Keep the evaluation methodology fixed.
4. Run the experiment.
5. Record actual results.
6. Analyze the result.
7. Continue only if the next module requires it.

Do not perform unlimited hyperparameter tuning.

Do not repeatedly rerun expensive experiments without a scientific reason.

---

## 18. NO RESULT CHASING

Do not modify methodology simply because a result is poor.

Do not:

- change labels to improve recall
- change the evaluation window after seeing results
- change the threshold using test performance
- remove difficult samples
- selectively report favorable metrics
- hide false positives
- hide failed models
- claim success based on a single favorable metric

If a model performs poorly, report that result and investigate the scientific reason.

---

## 19. REPRODUCIBILITY

Experiments should be deterministic whenever practical.

Use:

- fixed random seeds where randomness exists
- deterministic preprocessing
- fixed window definitions
- fixed splits
- documented model configuration

Do not introduce uncontrolled randomness unnecessarily.

---

## 20. COMPUTE LIMITS

Development hardware:

RTX 2050
4 GB VRAM

Therefore prioritize:

- streaming
- bounded historical state
- bounded graph windows
- chronological blocks
- memory-safe batches
- efficient data structures

Do not repeatedly process the entire LANL dataset during development.

Do not load the entire authentication dataset into RAM.

A full-dataset experiment should only be performed when explicitly required by the research plan.

---

## 21. RAW DATA PROTECTION

Never modify:

data/raw/lanl/

Do not create derived files inside the raw-data directory.

Generated datasets, caches, models, or experiment artifacts must be stored elsewhere if they are actually necessary.

Avoid creating large persistent artifacts unless required.

---

## 22. RESEARCH ARTIFACTS

Avoid unnecessary:

- notebooks
- CSV dumps
- JSON dumps
- pickle files
- huge logs
- temporary datasets
- duplicated models
- screenshots
- generated reports

Prefer:

- source code
- focused tests
- reproducible scripts
- concise state/results updates

---

## 23. EXISTING RESEARCH RESULTS

Existing results in PROJECT_STATE.md are part of the research history.

Do not overwrite them.

If a later experiment contradicts an earlier result:

preserve both results and explain the difference.

---

## 24. NOVELTY CLAIMS

Do not claim that this project is the:

- first GNN
- first temporal graph
- first graph-based lateral movement detector
- first LANL graph approach

unless explicitly supported by a documented literature review.

The graph-based lateral movement detection field already contains substantial prior research.

The final contribution must be based on the actual experimental findings.

---

## 25. HONEST CONCLUSIONS

The final conclusion must answer what the experiments actually demonstrate.

Possible outcomes include:

- graph improves detection
- graph does not improve detection
- hybrid improves performance
- graph helps within-window but fails cross-window
- temporal features dominate graph structure
- performance is limited by sparse ground truth
- generalization is poor

Any of these can be a valid research conclusion.

Never force the project toward a predetermined positive result.