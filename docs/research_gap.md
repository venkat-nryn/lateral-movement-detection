# Research Gap

This document outlines the gap this project addresses. It is intentionally conservative: no novelty claims are made here that have not been validated experimentally.

## Existing Approaches

- **Rule- and signature-based detection**: traditional intrusion-detection rules and authentication anomaly heuristics; often brittle and dependent on hand-crafted thresholds.
- **Conventional machine learning on log features**: per-event or per-host feature engineering followed by classifiers (e.g., tree ensembles, SVMs); typically ignores the relational structure between hosts and users.
- **Static graph-based methods**: graph analytics and graph learning applied to a single aggregated snapshot of host/user interactions; temporal ordering of events is largely discarded.
- **Temporal / dynamic graph learning (research literature)**: recent work applies temporal graph neural networks to security telemetry, but published results are frequently tied to specific datasets, evaluation protocols, or assumptions.

## Observed Limitations

- Static graph representations lose the ordering of events, which is central to how lateral movement unfolds over time.
- Conventional ML approaches treat hosts/events independently, missing multi-hop patterns that span several machines and credentials.
- Detection outputs are often presented without any measure of confidence, making it hard for analysts to prioritise alerts.
- Few approaches close the loop from detection to an interpretable reconstruction of the likely attack path.
- Comparability across studies is limited by inconsistent datasets, labels, and temporal evaluation protocols. *(To be confirmed against the literature review.)*

## Proposed Direction

This project proposes to study whether uncertainty-aware temporal graph learning can jointly provide:

1. improved multi-hop lateral-movement detection relative to conventional ML and static-graph baselines,
2. calibrated confidence estimates attached to detections,
3. risk scoring over hosts/users, and
4. reconstructed attack paths with human-readable explanations for analysts.

The emphasis is on a single end-to-end pipeline evaluated under a consistent temporal protocol rather than on any single component in isolation.

## Claims That Still Require Experimental Validation

- That temporal graph models outperform static-graph and conventional ML baselines on the chosen dataset(s). — **Not yet tested.**
- That uncertainty estimates produced by the system are well-calibrated and useful for alert prioritisation. — **Not yet tested.**
- That reconstructed attack paths align with ground-truth attack scenarios. — **Not yet tested.**
- That the combined pipeline provides practical value to analysts compared with baseline detections alone. — **Not yet tested.**

Each claim above will be revisited after the corresponding evaluation module is complete, and will either be supported by reported results or withdrawn.
