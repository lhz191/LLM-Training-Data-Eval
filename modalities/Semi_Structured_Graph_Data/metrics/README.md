# Semi-Structured & Graph Data — Layer 1 Metrics

Modality-level metrics for knowledge graphs, JSON structures, and log data.

## Graph Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **Valid_rule** | [graph_validity.py](graph_validity.py) | Rule-based graph validity rate (tree, cycle, planar, etc.) | LLM4GraphGen (Yao et al., 2024) |
| **MMD (Degree/Clustering/Orbits/Spectral)** | [graph_fidelity.py](graph_fidelity.py) | Maximum Mean Discrepancy between real and generated graph distributions | GGM-metrics (O'Bray et al.); GraphRNN (You et al., 2018) |
| **FCD** | [graph_fidelity.py](graph_fidelity.py) | Frechet ChemNet Distance for molecular graphs | Preuer et al., 2018 |
| **Novelty** | [graph_diversity.py](graph_diversity.py) | Fraction of generated graphs not isomorphic to prompt examples | Yao et al., 2024 |
| **Uniqueness** | [graph_diversity.py](graph_diversity.py) | Fraction of unique valid graphs (deduplicated by isomorphism) | Yao et al., 2024 |
| **sigma_SCR** | [graph_robustness.py](graph_robustness.py) | Syntactic Correctness Rate (parseable as valid graph) | Richardeau (Le Merrer & Tredan), 2025 |
| **GAD / GADcap** | [graph_robustness.py](graph_robustness.py) | Graph Atlas Distance (graph edit distance to canonical targets) | Richardeau, 2025 |
| **DL2** | [graph_robustness.py](graph_robustness.py) | Degree-distribution L2 deviation from reference | Richardeau, 2025 |

## JSON Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **ValidJSONRate** | [json_validity.py](json_validity.py) | Fraction of parseable JSON outputs | Survey Section 5.2.2 |
| **CorrectnessRate** | [json_validity.py](json_validity.py) | Fraction passing JSON schema validation | Survey Section 5.2.2 |
| **MMP** | [json_fidelity.py](json_fidelity.py) | Mean Match Percentage — field structure overlap with ground truth | Survey Section 5.2.2 |

## Log Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **PA / GA / FGA / PTA / RTA / FTA** | [log_validity.py](log_validity.py) | Parsing Accuracy, Grouping Accuracy, Template F1 | loghub-2.0 (Jiang et al., 2024, ISSTA'24) |
| **VP / VR / VF1** | [log_fidelity.py](log_fidelity.py) | Variable Precision / Recall / F1 for runtime variable extraction | LogBench (Li et al., 2024, IEEE TSE) |
| **BLEU / ROUGE** | [log_fidelity.py](log_fidelity.py) | N-gram overlap for template language fidelity | LogBench |
| **L-ACC** | [log_fidelity.py](log_fidelity.py) | Log Level Accuracy (exact match of severity level) | LogBench |
| **ScoreQualitative** | [log_privacy.py](log_privacy.py) | Human expert Likert-scale rating of anonymization effectiveness | Aghili et al., 2025; Pilan et al., 2022 |
