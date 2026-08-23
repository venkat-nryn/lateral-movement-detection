# High-Level Architecture

This document describes the high-level architecture of the system only. Detailed module designs will be documented in later modules as they are implemented.

## Pipeline Overview

```
Security Logs
      |
      v
Data Preprocessing
      |
      v
Temporal Graph Construction
      |
      v
Baseline Models / Temporal GNN
      |
      v
Detection
      |
      v
Uncertainty
      |
      v
Risk
      |
      v
Attack-Path Reconstruction
      |
      v
Explanation
      |
      v
Backend API
      |
      v
Analyst Dashboard
```

## Stage Descriptions

| Stage | Purpose | Planned Location |
|-------|---------|------------------|
| Security Logs | Raw host/network authentication and telemetry logs serving as input data. | `data/raw/` |
| Data Preprocessing | Cleaning, normalisation, entity resolution, and formatting of raw logs. | `ml/preprocessing/` |
| Temporal Graph Construction | Building time-ordered graphs from preprocessed events (nodes = hosts/users, edges = interactions over time). | `ml/graph/` |
| Baseline Models / Temporal GNN | Conventional ML baselines, static graph baselines, and the proposed temporal GNN model. | `ml/baselines/`, `ml/models/` |
| Detection | Producing detections of suspicious lateral-movement activity. | `ml/models/` / evaluation modules |
| Uncertainty | Estimating confidence/uncertainty for detections so analysts can gauge reliability. | `ml/uncertainty/` |
| Risk | Converting detections and uncertainty into host/user risk scores. | `ml/risk/` |
| Attack-Path Reconstruction | Reconstructing likely multi-hop attack paths across hosts. | `ml/path_analysis/` |
| Explanation | Human-readable justification of detections and reconstructed paths for analysts. | `ml/` (later module) |
| Backend API | Serving results (detections, uncertainty, risk, paths, explanations) to clients. | `backend/api/`, `backend/services/`, `backend/models/`, `backend/config/` |
| Analyst Dashboard | Frontend visualisation of detections, confidence, risk, and attack paths. | `frontend/` |

## Notes

- The exact interfaces between stages are intentionally undefined at this point and will be specified incrementally.
- Evaluation tooling (`ml/evaluation/`) supports all modelling stages but sits alongside the pipeline rather than in the main flow.
- Experiment configuration, results, and figures are managed under `experiments/`.
