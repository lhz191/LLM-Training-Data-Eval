"""
Config for Semi-Structured Data Evaluation (Graph + JSON + Log)

Survey dimensions:
  Graph — Validity, Fidelity (MMD/FCD), Diversity (Novelty/Uniqueness), Robustness (σSCR/GAD/DL2)
  JSON  — Validity (ValidJSONRate/CorrectnessRate), Fidelity (MMP)
  Log   — Validity (PA/GA/FGA/PTA/RTA/FTA), Fidelity (VP/VR/VF1/BLEU/ROUGE/L-ACC/AOD), Privacy
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

_C.jsonl_path = ''
_C.output_dir = ''
_C.mode = "graph"  # ["graph", "json", "log"]

# ── Metric toggles ───────────────────────────────────────────
_C.metrics = CN()

# Graph
_C.metrics.graph_validity = True
_C.metrics.graph_fidelity_mmd = True
_C.metrics.graph_diversity = True
_C.metrics.graph_robustness = True

# JSON
_C.metrics.json_validity = True
_C.metrics.json_fidelity = True

# Log
_C.metrics.log_validity = True
_C.metrics.log_fidelity = True
_C.metrics.log_privacy = False        # 需要人工打分数据

# ── Graph-specific configs ───────────────────────────────────
_C.graph = CN()
_C.graph.graph_field = "graph"         # JSONL field: generated graph text
_C.graph.ref_field = "ref_graph"       # JSONL field: reference graphs (for MMD/DL2)
_C.graph.example_field = ""            # JSONL field: example graphs in prompt (for Novelty)
_C.graph.target_field = ""             # JSONL field: target graph (for GAD)
_C.graph.format = "edge_list"          # "edge_list" | "adjacency" | "networkx_json"
_C.graph.directed = False
_C.graph.mmd_statistics = ["degree", "clustering", "orbits", "spectral"]
_C.graph.gad_cap = 0.0                 # GADcap 截断值, 0 表示不截断
_C.graph.gad_timeout = 30.0            # GED 超时秒数

# ── JSON-specific configs ────────────────────────────────────
_C.json_eval = CN()
_C.json_eval.json_field = "output"     # JSONL field: generated JSON text
_C.json_eval.ref_field = "reference"   # JSONL field: ground-truth JSON
_C.json_eval.schema_path = ""          # JSON Schema 文件路径

# ── Log-specific configs ─────────────────────────────────────
_C.log_eval = CN()
_C.log_eval.template_field = "EventTemplate"       # 生成的模板
_C.log_eval.ref_template_field = "RefEventTemplate" # ground-truth 模板
_C.log_eval.text_field = "LogText"                  # 生成的日志文本
_C.log_eval.ref_text_field = "RefLogText"           # 参考日志文本
_C.log_eval.variables_field = "Variables"            # 生成的变量列表
_C.log_eval.ref_variables_field = "RefVariables"     # ground-truth 变量列表
_C.log_eval.level_field = "Level"                    # 生成的日志级别
_C.log_eval.ref_level_field = "RefLevel"             # ground-truth 日志级别
_C.log_eval.privacy_ratings_field = "PrivacyRatings" # 人工隐私打分


def get_cfg(config_file_path):
    config = _C.clone()
    if config_file_path:
        config.merge_from_file(config_file_path)
    if config.output_dir and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    return config
