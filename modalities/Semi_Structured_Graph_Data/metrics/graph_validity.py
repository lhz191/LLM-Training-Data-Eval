"""
Graph Validity — 规则验证率  (Survey §5.2.1)

                    1
    Valid_rule = -------- Σ_{G ∈ G_gen} I{ passes_rules(G) }
                |G_gen|

用法:
    from graph_validity import compute_graph_validity, text_to_graph
    from LLM4GraphGen import RULES  # 论文提供的 8 种 rule

    # 方式1: 使用论文的 rule
    result = compute_graph_validity(data, rules=["tree", "cycle"], rule_registry=RULES)

    # 方式2: 传入自定义 rule
    def my_rule(G, spec):
        return G.number_of_nodes() > 5, "节点数检查"
    result = compute_graph_validity(data, rules=[my_rule])
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ════════════════════════════════════════════════════════════
# 解析: 文本 → networkx Graph
# ════════════════════════════════════════════════════════════

def text_to_graph(
    text: str,
    fmt: str = "edge_list",
    directed: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    将文本解析为 networkx 图对象。

    参数:
        text:     图的文本表示
        fmt:      "edge_list" | "adjacency" | "networkx_json"
        directed: 是否为有向图

    返回:
        (Graph对象 or None, 错误信息 or None)
    """
    if not HAS_NX:
        return None, "networkx 未安装"

    GraphCls = nx.DiGraph if directed else nx.Graph

    if fmt == "edge_list":
        # 支持多种常见边列表格式
        patterns = [
            r'\((\w+)\s*,\s*(\w+)(?:\s*,\s*[\d.]+)?\)',  # (A, B) 或 (A, B, weight)
            r'(\w+)\s*->\s*(\w+)',                          # A->B
            r'(\w+)\s*--?\s*(\w+)',                          # A-B 或 A--B
            r'\[(\w+)\s*,\s*(\w+)\]',                        # [A, B]
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            if matches:
                G = GraphCls()
                G.add_edges_from([(m[0], m[1]) for m in matches])
                return G, None
        return None, "无法解析边列表"

    elif fmt == "adjacency":
        try:
            data = json.loads(text)
            if isinstance(data, list) and all(isinstance(row, list) for row in data):
                G = nx.from_numpy_array(np.array(data, dtype=float), create_using=GraphCls)
                return G, None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None, "无法解析邻接矩阵"

    elif fmt == "networkx_json":
        try:
            data = json.loads(text) if isinstance(text, str) else text
            G = nx.node_link_graph(data, directed=directed)
            return G, None
        except Exception as e:
            return None, str(e)

    return None, f"未知格式: {fmt}"


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════

# Rule 的类型：可以是字符串（从 registry 查找）或直接传入函数
RuleType = Union[str, Callable]


def compute_graph_validity(
    data: List[Dict[str, Any]],
    rules: List[RuleType],
    rule_registry: Dict[str, Callable] = None,
    graph_field: str = "graph",
    fmt: str = "edge_list",
    directed: bool = False,
    spec: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    计算 Valid_rule: 生成图中通过所有规则检查的比例。

    参数:
        data:           JSONL 数据列表，每条包含一个图的文本表示
        rules:          规则列表，元素可以是:
                        - str: 从 rule_registry 中查找（如 "tree", "cycle"）
                        - callable: 自定义函数，签名为 (G, spec) -> (bool, str)
        rule_registry:  字符串规则名 -> 函数 的映射表
                        推荐传入 LLM4GraphGen.RULES
        graph_field:    data 中图文本所在的 key（默认 "graph"）
        fmt:            图文本格式: "edge_list" | "adjacency" | "networkx_json"
        directed:       是否按有向图解析
        spec:           全局任务参数（如 {"num_nodes": 15, "k": 3}），
                        单条数据中的同名字段会覆盖此值

    返回:
        {
            "valid_rule_rate": float,     # 核心指标
            "per_rule_rates":  dict,      # 每个规则各自的通过率
            "total":           int,
            "passed":          int,
            "rules_applied":   list[str], # 实际使用的规则名
            "details":         list,      # 前 50 条的逐样本结果
        }
    """
    if rule_registry is None:
        rule_registry = {}
    if spec is None:
        spec = {}

    # 解析 rules: 字符串 → 从 registry 取；callable → 直接用
    resolved_rules: List[Tuple[str, Callable]] = []
    for r in rules:
        if isinstance(r, str):
            fn = rule_registry.get(r)
            if fn is None:
                raise ValueError(f"规则 '{r}' 未在 rule_registry 中注册。"
                                 f"可用: {list(rule_registry.keys())}")
            resolved_rules.append((r, fn))
        elif callable(r):
            resolved_rules.append((getattr(r, '__name__', 'custom'), r))
        else:
            raise TypeError(f"规则类型错误: {type(r)}，需要 str 或 callable")

    rule_names = [name for name, _ in resolved_rules]

    total = 0
    all_pass = 0
    per_rule_pass = {name: 0 for name in rule_names}
    details = []

    for item in data:
        text = item.get(graph_field, '')
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        if not text:
            continue
        total += 1

        # 合并全局 spec 和单条数据中的参数
        merged_spec = dict(spec)
        for key in ['num_nodes', 'num_edges', 'num_components', 'k',
                     'partition_sizes', 'degree_sequence', 'max_degree']:
            if key in item:
                merged_spec[key] = item[key]

        G, parse_err = text_to_graph(str(text), fmt=fmt, directed=directed)

        sample_pass = True
        sample_detail = {'id': item.get('id', item.get('sample_id', total - 1))}

        for name, checker in resolved_rules:
            if G is None:
                ok, msg = False, f"解析失败: {parse_err}"
            else:
                ok, msg = checker(G, merged_spec)

            sample_detail[name] = {'pass': ok, 'detail': msg}
            if ok:
                per_rule_pass[name] += 1
            else:
                sample_pass = False

        sample_detail['all_pass'] = sample_pass
        if sample_pass:
            all_pass += 1
        details.append(sample_detail)

    return {
        'valid_rule_rate': all_pass / total if total > 0 else 0.0,
        'per_rule_rates': {r: per_rule_pass[r] / total if total > 0 else 0.0
                           for r in rule_names},
        'total': total,
        'passed': all_pass,
        'rules_applied': rule_names,
        'details': details[:50],
    }
