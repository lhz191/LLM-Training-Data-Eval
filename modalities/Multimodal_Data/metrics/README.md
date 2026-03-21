# Multimodal Data — Layer 1 Metrics

Modality-level metrics for image-text, video-text, and audio-text multimodal data.

## Image Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **Inception Score (IS)** | [inception_score.py](inception_score.py) | Image quality and class diversity via Inception V3 | Salimans et al., 2016 |
| **CLIP-T (Prompt Fidelity)** | [image_prompt_fidelity.py](image_prompt_fidelity.py) | CLIP cosine similarity between image and text prompt | Radford et al., 2021 |
| **CLIP-I (Subject Fidelity)** | [subject_fidelity.py](subject_fidelity.py) | CLIP image-image similarity for subject preservation | Radford et al., 2021 |
| **Well-Formed Rate (WFR)** | [well_formed_rate.py](well_formed_rate.py) | Fraction of outputs conforming to schema | — |
| **C2PA Validation** | [validate_cpa.py](validate_cpa.py) | Content provenance verification (C2PA standard) | C2PA Spec |

## Video Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **VBench (Holistic Fidelity)** | [holistic_fidelity.py](holistic_fidelity.py) | Comprehensive video quality benchmark (16+ dimensions) | Huang et al., 2024b |
| **Frame Diversity** | [frame_diver.py](frame_diver.py) | Optical flow-based frame-to-frame diversity | — |
| **Semantic Diversity** | [semantic_diver.py](semantic_diver.py) | Inception V3 feature-space diversity across frames | — |
| **Object Consistency** | [object_consistency.py](object_consistency.py) | CLIP-based object identity consistency across frames | — |
| **Temporal Accuracy** | [temporal_accuracy.py](temporal_accuracy.py) | Temporal coherence and motion quality | — |
| **Cross-Modal Consistency (CMC)** | [cmc.py](cmc.py) | ViCLIP video-text alignment score | InternVid (Wang et al., 2024) |
| **Video CMC** | [video_cmc.py](video_cmc.py) | ViCLIP-based video-level cross-modal consistency | InternVid (Wang et al., 2024) |
| **T2V Safety Bench** | [T2VSafetyBench.py](T2VSafetyBench.py) | GPT-4 Vision safety evaluation for text-to-video | — |
| **Win Rate** | [win_rate.py](win_rate.py) | Pairwise comparison win rate from human annotations | — |

## Audio Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **CLAP Score** | [clap_score.py](clap_score.py) | Audio-text alignment via CLAP model | Wu et al., 2023 (CLAP) |
| **FAD** | [fad.py](fad.py) | Frechet Audio Distance between real and generated audio | Kilgour et al., 2019 |
| **Audio Consistency** | [audio_consistency.py](audio_consistency.py) | Temporal audio feature consistency | — |
| **Audio KL Divergence** | [audio_kl.py](audio_kl.py) | KL divergence between audio feature distributions | — |

## TODO: Real-World Data Quality Metrics

The metrics above are primarily designed for **generated/synthetic** multimodal data. For **real-world** (naturally captured) data, the following quality dimensions are not yet covered:

### Image
- [ ] **Blur detection** — Laplacian variance, FFT frequency analysis (Ref: CleanVision; "A Data-Centric Perspective on Image Data Quality", 2025)
- [ ] **Exposure detection** — overexposure / underexposure (Ref: VizWiz-QualityIssues)
- [ ] **Noise estimation** — ISO noise, compression artifacts
- [ ] **Resolution check** — minimum resolution thresholds (Ref: AIM 2024 UHD-IQA)
- [ ] **Annotation accuracy** — CLIP-based label verification (Ref: ClipGrader, 2025; VEIL, 2023)
- [ ] **Near-duplicate detection** — MinHash / SimHash (Ref: CleanPatrick, 2025; Fastdup)

### Video
- [ ] **Encoding artifact detection** — codec compression damage (Ref: LEHA-CVQAD, 2025)
- [ ] **Frame corruption** — dropped/corrupted frames
- [ ] **Frame rate stability** — variable frame rate detection

### Audio
- [ ] **SNR estimation** — signal-to-noise ratio (Ref: GOMPSNR, 2025; NeMo-Curator)
- [ ] **Sampling rate check** — verify declared vs actual bandwidth (Ref: URGENT 2024)
- [ ] **Transcription accuracy (WER/CER)** — for speech data (Ref: NeMo-Curator; QualiSpeech, ACL 2025)

## Relationship to Layer 2

See [image_to_report_eval/metrics/](../image_to_report_eval/metrics/) for task-specific metrics: format_check, validity, report_quality, duplication, diversity.
