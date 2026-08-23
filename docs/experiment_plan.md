# Experiment Plan

This plan will be refined once the dataset and baselines are finalised. Sections below define what must be decided before any experiment is run.

## Dataset

- Candidate dataset(s): _TBD_
- Selection criteria: _TBD_ (e.g., availability, label quality, realism of authentication telemetry)
- Train/validation/test split strategy: _TBD_ (must respect time ordering; see *Temporal Evaluation*)
- Known limitations of the data and labels: _TBD_

## Baselines

Baseline families to be implemented before the proposed model:

1. Conventional ML baselines on non-graph features — specific models: _TBD_
2. Static graph-based baselines (graph analytics / static graph learning) — specific methods: _TBD_

Each baseline must use identical train/test partitions as the proposed model to ensure fair comparison.

## Proposed Model

- Temporal GNN architecture: _TBD_ (to be selected in a later module)
- Input graph construction choices: _TBD_
- Training configuration: _TBD_
- Uncertainty estimation method attached to detections: _TBD_

## Evaluation Metrics

Detection metrics:

- Precision, Recall, F1 — thresholds/aggregation level: _TBD_
- ROC-AUC / PR-AUC — applicability under class imbalance: _TBD_
- Alert-level metrics for analyst-facing evaluation: _TBD_

Path/analysis metrics:

- Attack-path reconstruction quality metric(s): _TBD_
- Explanation usefulness assessment: _TBD_

## Temporal Evaluation

- All evaluation must respect temporal ordering: models trained only on past data, evaluated on future data.
- Split strategy (e.g., chronological hold-out, rolling windows): _TBD_
- Handling of concept drift over time: _TBD_
- Reporting of performance as a function of time: _TBD_

## Uncertainty Evaluation

- Calibration metrics (e.g., ECE, reliability diagrams): _TBD_
- Whether uncertainty separates correct from incorrect detections: _TBD_
- Use of uncertainty in alert prioritisation experiments: _TBD_

## Ablation Studies

Planned ablations (each removes or replaces one component):

| # | Ablation | Question answered | Status |
|---|----------|-------------------|--------|
| 1 | _TBD_ | e.g., contribution of temporal structure vs. static graph | Not started |
| 2 | _TBD_ | e.g., contribution of uncertainty module | Not started |
| 3 | _TBD_ | e.g., sensitivity to graph-construction choices | Not started |

## Error Analysis

- Categorisation of false positives / false negatives by attack phase or event type: _TBD_
- Inspection of failure cases on multi-hop paths (where does reconstruction break?): _TBD_
- Behaviour under missing/noisy log data: _TBD_
- Documentation of findings per experiment run under `experiments/results/` with figures in `experiments/figures/`.
