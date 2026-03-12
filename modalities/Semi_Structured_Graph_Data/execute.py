"""
Semi-Structured Data Evaluation Runner (Graph + JSON + Log)

Survey §5.2 + §5.3 dimensions:
  Graph — Validity, Fidelity (MMD), Diversity (Novelty/Uniqueness), Robustness (σSCR/GAD/DL2)
  JSON  — Validity (ValidJSONRate/CorrectnessRate), Fidelity (MMP)
  Log   — Validity (PA/GA/FGA/PTA/RTA/FTA), Fidelity (VP/VR/VF1/BLEU/ROUGE/L-ACC/AOD), Privacy

Usage:
    python execute.py -f configs/test.yaml
"""

import argparse
import os
import json

import pandas as pd

from configs.basic_cfg import get_cfg

from metrics.graph_validity import compute_graph_validity, text_to_graph
from metrics.graph_fidelity import compute_graph_mmd
from metrics.graph_diversity import compute_graph_diversity
from metrics.graph_robustness import compute_gad, compute_dl2_batch
from metrics.json_validity import compute_json_validity
from metrics.json_fidelity import compute_json_fidelity
from metrics.log_validity import compute_log_validity
from metrics.log_fidelity import compute_log_fidelity
from metrics.log_privacy import compute_qualitative_score


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
                raise ValueError(f"JSON parse error at line {line_num}") from e
    return data


# ─── Graph helpers ────────────────────────────────────────────

def _parse_graph_field(item, field, fmt, directed=False):
    text = item.get(field, '')
    if not text:
        return None
    if isinstance(text, (dict, list)):
        text = json.dumps(text)
    G, _ = text_to_graph(str(text), fmt=fmt, directed=directed)
    return G


def _parse_graphs(data, field, fmt, directed=False):
    graphs = []
    for item in data:
        G = _parse_graph_field(item, field, fmt, directed)
        if G is not None:
            graphs.append(G)
    return graphs


# ─── Log helpers ──────────────────────────────────────────────

def _extract_field(data, field, default=''):
    return [item.get(field, default) for item in data]


def _extract_var_sets(data, field):
    result = []
    for item in data:
        v = item.get(field)
        if v is None:
            result.append(set())
        elif isinstance(v, list):
            result.append(set(str(x) for x in v))
        elif isinstance(v, str):
            result.append(set(v.split(',')) if v else set())
        else:
            result.append({str(v)})
    return result


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main(args):
    config = get_cfg(args.config_file)
    data = load_jsonl(config.jsonl_path)
    outputs = {}

    # ════════════════════════════════════════════════════════
    # GRAPH mode
    # ════════════════════════════════════════════════════════
    if config.mode == "graph":
        fmt = config.graph.format
        directed = config.graph.directed

        # --- Validity ---
        if config.metrics.graph_validity:
            print("[Graph/Validity] ...")
            outputs['graph_validity'] = compute_graph_validity(data, config)

        # parse generated + reference graphs
        syn_graphs = _parse_graphs(data, config.graph.graph_field, fmt, directed)
        ref_graphs = _parse_graphs(data, config.graph.ref_field, fmt, directed) if config.graph.ref_field else []

        # --- Fidelity (MMD) ---
        if config.metrics.graph_fidelity_mmd and ref_graphs:
            print("[Graph/Fidelity] MMD ...")
            outputs['graph_mmd'] = compute_graph_mmd(
                syn_graphs, ref_graphs,
                statistics=list(config.graph.mmd_statistics),
            )

        # --- Diversity ---
        if config.metrics.graph_diversity:
            print("[Graph/Diversity] Novelty & Uniqueness ...")
            example_graphs = []
            if config.graph.example_field:
                example_graphs = _parse_graphs(data, config.graph.example_field, fmt, directed)
            outputs['graph_diversity'] = compute_graph_diversity(
                syn_graphs,
                example_graphs=example_graphs if example_graphs else None,
            )

        # --- Robustness (GAD + DL2) ---
        if config.metrics.graph_robustness:
            # σSCR: parse success rate (already in validity as parseable_graph_rate)
            total = len(data)
            scr = len(syn_graphs) / total if total else 0.0
            outputs['graph_robustness'] = {'scr': scr, 'num_parsed': len(syn_graphs), 'num_total': total}

            # GAD (requires target graphs)
            if config.graph.target_field:
                target_graphs = _parse_graphs(data, config.graph.target_field, fmt, directed)
                if target_graphs and len(target_graphs) == len(syn_graphs):
                    print("[Graph/Robustness] GAD ...")
                    cap = config.graph.gad_cap if config.graph.gad_cap > 0 else None
                    gad_res = compute_gad(syn_graphs, target_graphs, cap=cap, timeout=config.graph.gad_timeout)
                    outputs['graph_robustness'].update(gad_res)

            # DL2 (requires reference graphs, one-to-one)
            if ref_graphs and len(ref_graphs) == len(syn_graphs):
                print("[Graph/Robustness] DL2 ...")
                dl2_res = compute_dl2_batch(syn_graphs, ref_graphs)
                outputs['graph_robustness'].update(dl2_res)

    # ════════════════════════════════════════════════════════
    # JSON mode
    # ════════════════════════════════════════════════════════
    elif config.mode == "json":
        if config.metrics.json_validity:
            print("[JSON/Validity] ValidJSONRate + CorrectnessRate ...")
            outputs['json_validity'] = compute_json_validity(data, config)

        if config.metrics.json_fidelity:
            print("[JSON/Fidelity] MMP ...")
            outputs['json_fidelity'] = compute_json_fidelity(data, config)

    # ════════════════════════════════════════════════════════
    # LOG mode
    # ════════════════════════════════════════════════════════
    elif config.mode == "log":
        lcfg = config.log_eval

        # --- Validity (PA/GA/FGA/PTA/RTA/FTA) ---
        if config.metrics.log_validity:
            gt_templates = _extract_field(data, lcfg.ref_template_field)
            pred_templates = _extract_field(data, lcfg.template_field)
            if any(gt_templates) and any(pred_templates):
                print("[Log/Validity] PA/GA/FGA/PTA/RTA/FTA ...")
                gt_df = pd.DataFrame({'EventTemplate': gt_templates})
                pred_df = pd.DataFrame({'EventTemplate': pred_templates})
                outputs['log_validity'] = compute_log_validity(gt_df, pred_df)
            else:
                outputs['log_validity'] = {'note': 'template fields missing'}

        # --- Fidelity (VP/VR/VF1 + BLEU/ROUGE + L-ACC/AOD) ---
        if config.metrics.log_fidelity:
            print("[Log/Fidelity] ...")
            fidelity_kwargs = {}

            pred_vars = _extract_var_sets(data, lcfg.variables_field)
            true_vars = _extract_var_sets(data, lcfg.ref_variables_field)
            if any(pred_vars) and any(true_vars):
                fidelity_kwargs['pred_var_list'] = pred_vars
                fidelity_kwargs['true_var_list'] = true_vars

            pred_texts = _extract_field(data, lcfg.text_field)
            ref_texts = _extract_field(data, lcfg.ref_text_field)
            if any(pred_texts) and any(ref_texts):
                fidelity_kwargs['pred_texts'] = pred_texts
                fidelity_kwargs['ref_texts'] = ref_texts

            pred_levels = _extract_field(data, lcfg.level_field)
            true_levels = _extract_field(data, lcfg.ref_level_field)
            if any(pred_levels) and any(true_levels):
                fidelity_kwargs['pred_levels'] = pred_levels
                fidelity_kwargs['true_levels'] = true_levels

            if fidelity_kwargs:
                outputs['log_fidelity'] = compute_log_fidelity(**fidelity_kwargs)
            else:
                outputs['log_fidelity'] = {'note': 'no fidelity fields found'}

        # --- Privacy (qualitative score) ---
        if config.metrics.log_privacy:
            ratings_raw = _extract_field(data, lcfg.privacy_ratings_field)
            ratings = [float(r) for r in ratings_raw if r]
            if ratings:
                print("[Log/Privacy] Qualitative Score ...")
                outputs['log_privacy'] = compute_qualitative_score(ratings)
            else:
                outputs['log_privacy'] = {'note': 'no privacy ratings found'}

    # ── Save ────────────────────────────────────────────────
    output_path = os.path.join(config.output_dir, "res.json")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_path}")
    for key, val in outputs.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, float):
                    print(f"  {key}.{k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semi-Structured Data Evaluation")
    parser.add_argument('-f', '--config-file', type=str, default="")
    args = parser.parse_args()
    main(args)
