"""
Multimodal Data Evaluation Runner (Layer 1)

运行 Layer 1 通用多模态指标（Multimodal_Data/metrics/ 下的指标）。
支持两种数据加载方式：
    1. 传统 JSONL 直读（video / image / audio 模式）
    2. Loader 加载（report 模式），自动适配为 Layer 1 metrics 期望的 dict 格式

Layer 2 任务特有指标（如 image_to_report_eval/metrics/）有各自独立的 run_full_test.py。

Usage:
    # 传统 video/image/audio 模式（从 JSONL 加载）
    python execute.py -f configs/test.yaml

    # report 模式（从 loader 加载，跑 image 相关的 Layer 1 指标）
    python execute.py -f configs/report_iu_xray.yaml
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import sys
import json

from configs.basic_cfg import get_cfg

sys.path.append(os.path.join(os.path.dirname(__file__), "metrics"))


# =============================================================================
# 数据加载
# =============================================================================

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: 第 {line_num} 行\n{line}") from e
    return data


def load_via_report_loader(config):
    """
    用 image_to_report_eval 的 loader 加载数据，
    转为 Layer 1 metrics 通用的 dict 格式:
        {"image_path": str, "caption": str, "sample_id": str, ...}

    image_path 使用拼好的完整路径。
    """
    report_eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_to_report_eval")
    if report_eval_dir not in sys.path:
        sys.path.insert(0, report_eval_dir)

    from loaders import IUXRayLoader, ShareGPT4VLoader

    rc = config.report
    if rc.dataset == 'iu_xray':
        loader = IUXRayLoader(rc.data_path, split=rc.split)
    elif rc.dataset == 'sharegpt4v':
        loader = ShareGPT4VLoader(rc.data_path)
    else:
        raise ValueError(f"Unknown report dataset: {rc.dataset}")

    image_base_dir = rc.image_base_dir
    max_samples = rc.max_samples if rc.max_samples > 0 else None

    data = []
    for i, sample in enumerate(loader.iterate()):
        if max_samples and i >= max_samples:
            break

        img_path = ""
        if sample.images:
            raw = sample.images[0]
            if os.path.isabs(raw):
                img_path = raw
            else:
                img_path = os.path.join(image_base_dir, raw)

        data.append({
            "sample_id": sample.sample_id,
            "id": sample.sample_id,
            "image_path": img_path,
            "caption": sample.report,
            "text": sample.report,
            "gen_text": sample.report,
            "prompt": sample.instruction.replace("<image>", "").strip(),
            "images": [
                os.path.join(image_base_dir, p) if not os.path.isabs(p) else p
                for p in sample.images
            ],
            "metadata": sample.metadata,
        })

    print(f"[Loader] {rc.dataset}: {len(data)} samples loaded")
    return data


# =============================================================================
# Text metrics (inline implementations, offline-safe)
# =============================================================================

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _compute_self_cos_sim(data, device):
    """Self-CosSim: average pairwise cosine similarity of all text embeddings."""
    from sentence_transformers import SentenceTransformer
    from itertools import combinations

    model_path = os.path.join(_MODELS_DIR, "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_path, device=str(device))
    texts = [item.get("gen_text", "") for item in data if item.get("gen_text")]
    if len(texts) < 2:
        return 1.0

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    sim_sum, count = 0.0, 0
    for i, j in combinations(range(len(embeddings)), 2):
        sim_sum += float(np.dot(embeddings[i], embeddings[j]))
        count += 1

    return sim_sum / count if count else 0.0


def _compute_grammaticality(data, device):
    """Grammaticality Rate via RoBERTa-CoLA."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F

    model_path = os.path.join(_MODELS_DIR, "roberta-base-CoLA")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model = model.to(device).eval()
    scores = []
    for item in data:
        text = item.get("text", "")
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = F.softmax(logits, dim=1)[0][1].item()
        scores.append(prob)

    return float(np.mean(scores)) if scores else 0.0


def _compute_text_diversity(data):
    """TTR, Distinct-N, n-gram entropy."""
    import math
    from collections import Counter
    from nltk.tokenize import word_tokenize

    def _ttr(tokens):
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    def _distinct_n(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    def _entropy(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total, 2) for c in counts.values())

    ttrs, d1s, d2s, ents = [], [], [], []
    for item in data:
        text = item.get("gen_text", "")
        if not text:
            continue
        tokens = word_tokenize(text.lower())
        ttrs.append(_ttr(tokens))
        d1s.append(_distinct_n(tokens, 1))
        d2s.append(_distinct_n(tokens, 2))
        ents.append(_entropy(tokens, 2))

    return {
        "TTR": float(np.mean(ttrs)) if ttrs else 0.0,
        "Distinct-1": float(np.mean(d1s)) if d1s else 0.0,
        "Distinct-2": float(np.mean(d2s)) if d2s else 0.0,
        "2-gram_Entropy": float(np.mean(ents)) if ents else 0.0,
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    config = get_cfg(args.config_file)

    outputs = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # report 模式用 loader，其余用 JSONL
    if config.mode == 'report':
        data = load_via_report_loader(config)
    else:
        data = load_jsonl(config.jsonl_path)

    if config.mode == 'video':
        from metrics.frame_diver import get_frame_diversity
        from metrics.object_consistency import get_object_consistency
        from metrics.semantic_diver import get_semantic_diversity
        from metrics.T2VSafetyBench import T2VSafetyBench
        from metrics.video_cmc import get_cmc
        from metrics.holistic_fidelity import run_VBench

        if config.metrics.frame_diversity:
            print("Measuring Frame Diversity...")
            outputs['frame_diversity'] = float(get_frame_diversity(data))

        if config.metrics.object_consistency:
            print("Measuring Object Consistency...")
            outputs['object_consistency'] = float(get_object_consistency(data, device))

        if config.metrics.semantic_diversity:
            print("Measuring Semantic Diversity...")
            outputs['semantic_diversity'] = float(get_semantic_diversity(data, device))

        if config.metrics.safety_bench:
            print("Measuring Safety using T2VBench...")
            outputs["Safety"] = T2VSafetyBench(config, data)

        if config.metrics.cross_modal_consistency:
            print("Measuring Cross Modal Consistency...")
            outputs["cmc"] = get_cmc(data, device)

        if config.metrics.vbench:
            print("Measuring via VBench...")
            outputs["VBench"] = run_VBench(config, data)

    elif config.mode in ('image', 'report'):
        if config.metrics.well_formed_rate:
            from metrics.well_formed_rate import get_well_formed_rate
            print("Measuring Well-Formed Rate...")
            outputs["well_formed_rate"] = get_well_formed_rate(data, config)

        if config.metrics.win_rate:
            from metrics.win_rate import get_win_rate
            print("Measuring Win Rate...")
            outputs["win_rate"] = get_win_rate(data)

        if config.metrics.validate_cpa:
            from metrics.validate_cpa import get_validate_cpa
            print("Measuring Validate CPA...")
            outputs["validate_cpa"] = get_validate_cpa(data, config)

        if config.metrics.safety_asr_rr:
            from metrics.safety_asr_rr import get_safety_asr_rr
            print("Measuring Safety ASR/RR...")
            outputs["safety_asr_rr"] = get_safety_asr_rr(data, config.safety_asr_rr)

        if config.metrics.inception_score:
            from metrics.inception_score import get_inception_score
            print("Measuring Inception Score...")
            outputs["inception_score"] = get_inception_score(data, config.inception_score)

        if config.metrics.prompt_fidelity:
            from metrics.image_prompt_fidelity import get_prompt_fidelity
            print("Measuring Prompt Fidelity (CLIP text-image alignment)...")
            outputs["prompt_fidelity"] = get_prompt_fidelity(data, device)

        if getattr(config.metrics, 'subject_fidelity', False):
            from metrics.subject_fidelity import get_subject_fidelity
            print("Measuring Subject Fidelity (CLIP-I)...")
            outputs["subject_fidelity"] = get_subject_fidelity(data, device)

        # ---- Text metrics (inline, avoid module-level downloads) ----
        if config.mode == 'report':
            if getattr(config.metrics, 'self_cos_sim', False):
                print("Measuring Self-CosSim (text semantic redundancy)...")
                outputs["self_cos_sim"] = _compute_self_cos_sim(data, device)

            if getattr(config.metrics, 'grammaticality_rate', False):
                print("Measuring Grammaticality Rate (RoBERTa-CoLA)...")
                outputs["grammaticality_rate"] = _compute_grammaticality(data, device)

            if getattr(config.metrics, 'text_diversity', False):
                print("Measuring Text Diversity (TTR / Distinct-N / Entropy)...")
                outputs["text_diversity"] = _compute_text_diversity(data)

    else:
        if config.metrics.clap_score:
            from metrics.clap_score import compute_clap_score
            print("Measuring Clap Score...")
            outputs['clap_score'] = compute_clap_score(data, device)
        if config.metrics.fad:
            print("Measuring FAD...")

    output_path = os.path.join(config.output_dir, "res.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, default="")
    args = parser.parse_args()
    main(args)
