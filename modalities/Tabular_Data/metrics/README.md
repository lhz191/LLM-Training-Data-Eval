# Tabular Data — Layer 1 Metrics

Modality-level metrics for structured table data (synthetic tabular generation quality).

## Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **beta-Recall** | [diversity.py](diversity.py) | Coverage of real distribution by synthetic data | Alaa et al., 2022; TabSyn (Zhang et al., 2024a) |
| **CovGap** | [fairness.py](fairness.py) | Sub-group coverage gap (demographic parity in representation) | Survey Section 4.3 |
| **CondShift** | [fairness.py](fairness.py) | Label-conditional distribution shift per sub-group (TV distance) | Survey Section 4.3 |
| **KSComplement** | [fidelity.py](fidelity.py) | Numerical column distribution similarity (1 - KS statistic) | SDMetrics |
| **TVComplement** | [fidelity.py](fidelity.py) | Categorical column distribution similarity (1 - TVD) | SDMetrics |
| **CorrelationSimilarity** | [fidelity.py](fidelity.py) | Pairwise correlation preservation between columns | SDMetrics |
| **ContingencySimilarity** | [fidelity.py](fidelity.py) | Joint distribution preservation for categorical pairs | SDMetrics |
| **LogisticDetection** | [fidelity.py](fidelity.py) | Classifier two-sample test (synthetic vs real distinguishability) | SDMetrics |
| **DCR** | [privacy.py](privacy.py) | Distance to Closest Record (memorization risk) | Borisov et al., 2023; Fang et al., 2024 |
| **MIA** | [privacy.py](privacy.py) | Membership Inference Attack AUC (privacy leakage) | Borisov et al., 2023 |
| **CSTest** | [validity.py](validity.py) | Chi-squared test for categorical column distributions | SDMetrics |
| **BoundaryAdherence** | [validity.py](validity.py) | Whether numerical values stay within real data range | SDMetrics |
| **CategoryAdherence** | [validity.py](validity.py) | Whether categorical values are valid (seen in real data) | SDMetrics |
| **Violation Rate (VR)** | [validity.py](validity.py) | Fraction of constraint violations (custom rules) | Survey Section 4.2 |
