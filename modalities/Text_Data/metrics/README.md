# Text Data — Layer 1 Metrics

Modality-level metrics for pure text corpora (dialogue, instruction-response, summarization, etc.).

## Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **EDS** | [eds.py](eds.py) | Embedding Diversity Score — corpus-level semantic diversity via sentence embeddings | — |
| **Grammaticality Rate** | [GrammaticalityRate.py](GrammaticalityRate.py) | Fraction of grammatically acceptable text (RoBERTa-CoLA classifier) | Warstadt et al., 2019 (CoLA) |
| **PMI-FAITH** | [PMI_FAITH.py](PMI_FAITH.py) | Pointwise Mutual Information faithfulness — context-response relevance via log-probability | — |
| **Self-CosSim** | [selfcossim.py](selfcossim.py) | Self-cosine similarity — pairwise semantic similarity within corpus (lower = more diverse) | — |
| **Text Diversity** | [text_diver.py](text_diver.py) | Type-Token Ratio, Distinct-N, n-gram entropy | Li et al., 2016 (Distinct-N) |
| **Toxicity Ratio** | [toxical_ratio.py](toxical_ratio.py) | Fraction of toxic content via Perspective API (8 dimensions) | Perspective API (Jigsaw) |
| **USL** | [usl.py](usl.py) | Understandability, Sensibleness, Likability — LLM-based quality scoring | — |
| **USR** | [usr.py](usr.py) | Unreferenced Score for dialogue Response evaluation | Mehri & Eskenazi, 2020 |
| **RUBER** | [ruber.py](ruber.py) | Referenced + Unreferenced Bilingual Evaluation of Response | Tao et al., 2018 |
| **Instruction Following** | [instruction_following_eval/](instruction_following_eval/) | Instruction constraint adherence rate | — |
