# Uncertainty-Aware Temporal Graph Analytics for Lateral Movement Detection and Attack-Path Reconstruction

## Project Objective

This project investigates whether temporal graph learning can improve the detection and analysis of multi-hop lateral movement in enterprise networks compared with conventional machine-learning and static graph-based approaches. Beyond detection, the system is designed to quantify prediction uncertainty, assign risk scores, and reconstruct likely attack paths so that security analysts receive actionable, explainable output rather than raw alerts. The work spans the full pipeline: from raw security logs, through temporal graph construction and model development, to a backend API and analyst-facing dashboard.

## High-Level System Components

- **Data pipeline** (`data/`): storage for raw security logs, interim artifacts, and processed datasets.
- **ML pipeline** (`ml/`): preprocessing, temporal graph construction, baseline models, proposed temporal GNN models, uncertainty estimation, risk scoring, attack-path reconstruction, and evaluation.
- **Backend API** (`backend/`): API layer, services, models, and configuration that expose detection, uncertainty, risk, and attack-path results to clients.
- **Frontend** (`frontend/`): analyst dashboard for visualising detections, confidence, risk, and reconstructed attack paths.
- **Experiments** (`experiments/`): experiment configurations, result files, and generated figures.
- **Notebooks** (`notebooks/`): exploratory analysis and dataset inspection.
- **Tests** (`tests/`), **scripts** (`scripts/`), and **checkpoints** (`checkpoints/`) supporting reproducibility.

See `docs/architecture.md` for the high-level architecture diagram.

## Current Development Status

| Module | Description | Status |
|--------|-------------|--------|
| M0.1 | Project structure and foundational documentation | In progress / current |
| Later modules | Data ingestion, preprocessing, graph construction, baselines, models, uncertainty, risk, path analysis, evaluation, backend, frontend | Not started |

No datasets have been downloaded and no ML dependencies (e.g., PyTorch, PyTorch Geometric) have been installed yet.

## Incremental Development Statement

This project is being developed incrementally in small modules. Each module is completed and verified before the next one begins. This repository currently contains only the project skeleton and foundational documentation; no application or ML code has been written yet.
