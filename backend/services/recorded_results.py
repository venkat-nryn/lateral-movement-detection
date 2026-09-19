"""Recorded research results the export does not recompute, with citations.

The export trains and serves only the frozen standard XGBoost; its metrics are
recomputed in :mod:`backend.services.experiments`. The models below are too
expensive to retrain at dashboard start-up (the neural ablation alone takes
about 40 minutes on the RTX 2050), so their figures are quoted verbatim from
PROJECT_STATE.md. Each entry cites its section, and the UI labels every one as
*recorded*, never as live.

Graph-bearing neural results are not reproducible run to run (section 34), so
their single-run cells are reported with that caveat and the ablation reports
the three-seed mean and range.
"""

from __future__ import annotations

RECORDED_NOTE = (
    "Recorded from PROJECT_STATE.md, not recomputed by the dashboard. "
    "Graph-bearing neural results vary run to run (section 34)."
)

#: W1 test, single runs (sections 15 and 28.5).
W1_TEST_SINGLE_RUNS = [
    {
        "model": "XGBoost (class-weighted)",
        "inputs": "B",
        "section": "15",
        "threshold": 0.210157,
        "pr_auc": 0.8588,
        "roc_auc": 0.999986,
        "precision": 0.8182,
        "recall": 0.8182,
        "f1": 0.8182,
        "tp": 9, "fp": 2, "tn": 200823, "fn": 2,
        "predicted_positive": 11,
        "reproducible": True,
    },
    {
        "model": "Neural: behavioural only",
        "inputs": "B+T",
        "section": "28.5",
        "parameters": 881,
        "threshold": 0.996246,
        "pr_auc": 0.9088,
        "roc_auc": 0.999993,
        "precision": 0.8333,
        "recall": 0.9091,
        "f1": 0.8696,
        "tp": 10, "fp": 2, "tn": 200823, "fn": 1,
        "predicted_positive": 12,
        "reproducible": True,
    },
    {
        "model": "Hybrid GNN (graph + behavioural)",
        "inputs": "G+B+T",
        "section": "28.5",
        "parameters": 10863,
        "threshold": 0.999382,
        "pr_auc": 0.8444,
        "roc_auc": 0.999973,
        "precision": 0.6000,
        "recall": 0.8182,
        "f1": 0.6923,
        "tp": 9, "fp": 6, "tn": 200819, "fn": 2,
        "predicted_positive": 15,
        "reproducible": False,
    },
    {
        "model": "Temporal GAT (graph only)",
        "inputs": "G+T",
        "section": "28.5",
        "parameters": 10285,
        "threshold": 0.999047,
        "pr_auc": 0.1403,
        "roc_auc": 0.995282,
        "precision": 0.1000,
        "recall": 0.5455,
        "f1": 0.1690,
        "tp": 6, "fp": 54, "tn": 200771, "fn": 5,
        "predicted_positive": 60,
        "reproducible": False,
    },
]

#: M4.7 ablation, three seeds, on W1 test and the corrected full W2 (section 36.1).
ABLATION = [
    {"model": "Neural: behavioural only", "inputs": "B+T", "w1_pr_auc": 0.8886, "w1_range": [0.835, 0.922], "w2_pr_auc": 0.1108, "w2_range": [0.062, 0.139], "w2_recall": 0.315, "w2_fp_per_hour": 130},
    {"model": "Neural: behavioural, no target", "inputs": "B", "w1_pr_auc": 0.8710, "w1_range": [0.771, 0.940], "w2_pr_auc": 0.1451, "w2_range": [0.127, 0.155], "w2_recall": 0.301, "w2_fp_per_hour": 90},
    {"model": "Hybrid GNN", "inputs": "G+B+T", "w1_pr_auc": 0.8130, "w1_range": [0.687, 0.984], "w2_pr_auc": 0.1306, "w2_range": [0.076, 0.217], "w2_recall": 0.573, "w2_fp_per_hour": 1551},
    {"model": "Temporal GAT", "inputs": "G+T", "w1_pr_auc": 0.1261, "w1_range": [0.071, 0.205], "w2_pr_auc": 0.0320, "w2_range": [0.016, 0.046], "w2_recall": 0.246, "w2_fp_per_hour": 547},
    {"model": "Target attributes only", "inputs": "T", "w1_pr_auc": 0.0080, "w1_range": [0.007, 0.009], "w2_pr_auc": 0.0009, "w2_range": [0.0009, 0.0009], "w2_recall": 0.094, "w2_fp_per_hour": 2923},
]

#: Marginal value of each information source, PR-AUC difference of seed means (section 32.4).
MARGINAL_CONTRIBUTIONS = [
    {"contribution": "graph, given target", "w1": 0.118, "w2": 0.031},
    {"contribution": "graph, given behavioural + target", "w1": -0.076, "w2": 0.020},
    {"contribution": "behavioural, given target", "w1": 0.881, "w2": 0.110},
    {"contribution": "behavioural, given graph + target", "w1": 0.687, "w2": 0.099},
    {"contribution": "target, given behavioural", "w1": 0.018, "w2": -0.034},
]

INPUT_LEGEND = {
    "G": "graph embeddings (GAT over the pre-cutoff temporal graph)",
    "B": "the 17 temporal/behavioural features (M3.4)",
    "T": "7 target-event attributes",
}


def recorded_results() -> dict:
    return {
        "kind": "recorded",
        "note": RECORDED_NOTE,
        "source": "PROJECT_STATE.md",
        "input_legend": INPUT_LEGEND,
        "w1_test_single_runs": W1_TEST_SINGLE_RUNS,
        "ablation": {"section": "36.1", "seeds": 3, "rows": ABLATION},
        "marginal_contributions": {"section": "32.4", "rows": MARGINAL_CONTRIBUTIONS},
    }
