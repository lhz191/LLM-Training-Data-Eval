#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Complexity 任务复杂度指标

四个维度评估 API 调用数据的任务复杂度。
维度 A/B/C 为静态分析（无外部依赖），维度 D 需要 LLM API（可选开启）。

A. 工具选择难度（Tool Selection Difficulty）
   - 候选工具池越大，选对工具越难
   - 1-3 个候选 → easy (0.2) | 4-10 → moderate (0.5) | >10 → hard (1.0)

B. 参数填充难度（Param Filling Difficulty）
   - 需要填的参数越多、有嵌套类型越难
   - 0-2 参数 → easy (0.2) | 3-8 → moderate (0.5) | >8 或嵌套 → hard (1.0)

C. 多步规划难度（Planning Difficulty）
   基于 Cognitive Load Theory (Beyond Accuracy, 2025) 的思路：
   - 调用链长度 + 步间数据依赖检测
   - 注意力距离 (Memory Load): 依赖步之间隔了多少步，距离越远记忆负担越大
   - 选择干扰度 (Selection Load): 被依赖的 response 中同类候选值数量，越多越难选对
   - 单步 → easy (0.2) | 多步无依赖 → moderate (0.5) | 多步+依赖 → hard (1.0)

D. Query 清晰度（Query Clarity, LLM Judge, 可选）
   通过 LLM 评估每条 query 作为训练数据的质量：
   - 意图明确性: 用户指令是否清晰表达了任务
   - 参数充分性: 是否提供了 API 调用所需的关键参数
   - 表述质量: 语法规范性和自然度
   不清晰的 query = 低质量训练数据

使用方式:

    from loaders import ToolBenchLoader
    from metrics.task_complexity import compute_task_complexity

    loader = ToolBenchLoader('/path/to/toolbench.json')

    results = compute_task_complexity(
        data_iterator=loader.iterate(),
        dataset_name='ToolBench',
        max_samples=10000,
        output_file='results/toolbench/task_complexity.json',
        enable_query_clarity=True,  # 开启 LLM judge (需要 API)
    )
"""

import re
import json
import time
from collections import Counter
from datetime import datetime
from typing import Iterator, Dict, Any, Optional, List, Tuple

import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import APIAgentSample, APICall  # noqa: E402

# =============================================================================
# LLM 配置（用于 Query Clarity 评估）
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4.1'


# =============================================================================
# 分数映射
# =============================================================================

def tool_selection_to_score(n_candidate: int) -> float:
    """候选工具数 → 选择难度分数"""
    if n_candidate <= 3:
        return 0.2
    elif n_candidate <= 10:
        return 0.5
    else:
        return 1.0


def param_filling_to_score(n_args: int, has_complex: bool) -> float:
    """参数量 + 是否有嵌套类型 → 填充难度分数"""
    if has_complex and n_args > 4:
        return 1.0
    if n_args <= 2:
        return 0.2
    elif n_args <= 8:
        return 0.5
    else:
        return 1.0


def planning_to_score(chain_length: int, dep_depth: int) -> float:
    """调用链长 + 依赖深度 → 规划难度分数"""
    if chain_length <= 1:
        return 0.2
    if dep_depth >= 1:
        return 1.0
    return 0.5


def score_to_label(score: float) -> str:
    if score <= 0.2:
        return "easy"
    elif score <= 0.5:
        return "moderate"
    else:
        return "hard"


# =============================================================================
# 单样本特征提取
# =============================================================================

def _get_real_calls(sample: APIAgentSample) -> List[APICall]:
    """过滤掉 Finish / 空名调用，返回真实 API 调用列表"""
    calls = []
    for c in sample.api_calls:
        name = (c.name or "").strip()
        if not name or name.lower() in ("finish", "finalaction"):
            continue
        calls.append(c)
    return calls


def _extract_tool_selection(sample: APIAgentSample, real_calls: List[APICall]) -> Dict[str, Any]:
    """维度 A: 工具选择"""
    n_candidate = len(sample.tools)
    used_names = set()
    for c in real_calls:
        used_names.add((c.name or "").strip())
    n_distinct_used = len(used_names)
    selection_ratio = n_distinct_used / n_candidate if n_candidate > 0 else 0.0
    score = tool_selection_to_score(n_candidate)
    return {
        "n_candidate_tools": n_candidate,
        "n_distinct_tools_used": n_distinct_used,
        "selection_ratio": selection_ratio,
        "score": score,
        "label": score_to_label(score),
    }


def _extract_param_filling(sample: APIAgentSample, real_calls: List[APICall]) -> Dict[str, Any]:
    """维度 B: 参数填充"""
    n_required_total = 0
    n_args_filled = 0
    has_complex = False

    for call in real_calls:
        tool = sample.get_tool_by_name((call.name or "").strip())
        if tool is not None:
            n_required_total += len(tool.get_required_params())
        args = call.arguments
        if isinstance(args, dict):
            n_args_filled += len(args)
            for v in args.values():
                if isinstance(v, (dict, list)):
                    has_complex = True

    score = param_filling_to_score(n_args_filled, has_complex)
    return {
        "n_required_params_total": n_required_total,
        "n_args_filled_total": n_args_filled,
        "has_complex_args": has_complex,
        "score": score,
        "label": score_to_label(score),
    }


def _parse_response_to_obj(response: str) -> Any:
    """
    将 API response 字符串解析为 Python 对象。

    ToolBench 的 response 格式是 {"error": "", "response": "<python repr string>"}，
    内层用单引号（Python repr），不是合法 JSON。需要多层尝试：
    1. json.loads 直接解析
    2. json.loads 外层 → ast.literal_eval 内层 "response" 字段
    3. ast.literal_eval 整个字符串
    """
    import ast

    if not response:
        return None

    # 策略 1: 直接 JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "response" in data:
            inner = data["response"]
            if isinstance(inner, (dict, list)):
                return inner
            if isinstance(inner, str) and inner.strip():
                # 内层是字符串，尝试再解析（ToolBench 的常见格式）
                try:
                    return json.loads(inner)
                except (json.JSONDecodeError, ValueError):
                    pass
                try:
                    return ast.literal_eval(inner)
                except (ValueError, SyntaxError):
                    pass
            return data
        return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 2: ast.literal_eval 整体
    try:
        return ast.literal_eval(response)
    except (ValueError, SyntaxError):
        pass

    return None


def _count_similar_values_in_response(response: str, target_value: str) -> int:
    """
    统计 response 中与 target_value 同类型的候选实体数量（选择干扰度）。

    策略：
    1. 解析 response 为结构化对象（兼容 JSON 和 Python repr 格式），
       递归查找 target_value 所在的 key，统计同 key 下不同值数量。
       例如 response 含 [{"id": "A"}, {"id": "B"}, {"id": "C"}]，target="A"，
       则 key="id" 有 3 个不同值，干扰项 = 2。
    2. 解析失败时 fallback 到正则启发式。
    """
    if not response or not target_value:
        return 0

    target = str(target_value).strip()
    if len(target) < 2:
        return 0

    # --- 策略 1: 结构感知 ---
    data = _parse_response_to_obj(response)
    if data is not None:
        key_values: Dict[str, set] = {}

        def _collect(obj: Any, depth: int = 0) -> None:
            if depth > 15:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        vs = str(v)
                        if k not in key_values:
                            key_values[k] = set()
                        key_values[k].add(vs)
                    elif isinstance(v, (dict, list)):
                        _collect(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _collect(item, depth + 1)

        try:
            _collect(data)
            for key, vals in key_values.items():
                if target in vals and len(vals) > 1:
                    return len(vals) - 1
            return 0
        except (RecursionError, TypeError):
            pass

    # --- 策略 2: Fallback 正则启发式 ---
    id_pattern = re.match(r'^([a-zA-Z_-]+)[\d_-]+', target)
    if id_pattern:
        prefix = id_pattern.group(1)
        matches = re.findall(re.escape(prefix) + r'[\d_-]+', response)
        return max(len(set(matches)) - 1, 0)

    if target.isdigit() and len(target) >= 4:
        matches = re.findall(r'\b\d{' + str(len(target)) + r'}\b', response)
        return max(len(set(matches)) - 1, 0)

    return 0


def _extract_planning(real_calls: List[APICall]) -> Dict[str, Any]:
    """
    维度 C: 多步规划

    基于 Cognitive Load Theory (Beyond Accuracy, 2025) 的思路，从三个方面评估：

    1. 依赖检测: call[i] 的 argument 值是否出现在 call[j] (j < i) 的 response 中，
       说明存在数据依赖（后续调用用到了前面调用的结果）。
    2. 注意力距离 (Memory Load): 依赖操作之间隔了多少步。
       距离越远，agent 需要在更长的上下文中"记住"并正确引用前面的值，难度越大。
    3. 选择干扰度 (Selection Load): 被依赖的 response 中包含多少个同类型的候选值。
       如果 response 返回了 5 个 order_id 但只需引用其中 1 个，
       agent 需要从干扰项中选出正确的，难度比只有 1 个值高得多。
    """
    chain_length = len(real_calls)
    n_distinct = len(set((c.name or "").strip() for c in real_calls))

    dep_depth = 0
    dep_links = 0
    dep_distances: List[int] = []
    selection_interferences: List[int] = []

    for i in range(1, len(real_calls)):
        current_args = real_calls[i].arguments
        if not isinstance(current_args, dict) or not current_args:
            continue
        arg_values = set()
        for v in current_args.values():
            s = str(v).strip()
            if len(s) >= 3:
                arg_values.add(s)
        if not arg_values:
            continue

        for j in range(i - 1, -1, -1):
            resp = real_calls[j].response
            if not resp:
                continue
            for av in arg_values:
                if av in resp:
                    dep_links += 1
                    dep_depth = max(dep_depth, dep_links)

                    dep_distances.append(i - j)

                    n_similar = _count_similar_values_in_response(resp, av)
                    selection_interferences.append(n_similar)

                    break

    unique_api_ratio = n_distinct / chain_length if chain_length > 0 else 0.0
    avg_dep_distance = float(np.mean(dep_distances)) if dep_distances else 0.0
    max_dep_distance = max(dep_distances) if dep_distances else 0
    avg_selection_interference = float(np.mean(selection_interferences)) if selection_interferences else 0.0
    max_selection_interference = max(selection_interferences) if selection_interferences else 0

    score = planning_to_score(chain_length, dep_depth)
    return {
        "chain_length": chain_length,
        "n_distinct_apis": n_distinct,
        "unique_api_ratio": unique_api_ratio,
        "dependency_depth": dep_depth,
        "dependency_links": dep_links,
        "avg_dependency_distance": avg_dep_distance,
        "max_dependency_distance": max_dep_distance,
        "avg_selection_interference": avg_selection_interference,
        "max_selection_interference": max_selection_interference,
        "score": score,
        "label": score_to_label(score),
    }


# =============================================================================
# 维度 D: Query 清晰度 (LLM Judge)
# =============================================================================

QUERY_CLARITY_PROMPT = """你是一个 API Agent 训练数据质量评估专家。请评估以下 query（用户给 agent 的指令）作为训练数据时的清晰度。

【Query】
{query}

【可用工具列表】
{tool_names}

请从以下三个方面评估：

1. 意图明确性 (Intent Clarity): query 是否清晰表达了用户想要完成的任务？
   - 1 分: 意图清晰、无歧义，一个合格的 agent 能明确知道该做什么
   - 0.5 分: 基本能理解意图，但存在部分模糊或缺少关键细节
   - 0 分: 意图严重不清，无法确定用户到底想做什么

2. 参数充分性 (Parameter Sufficiency): query 中是否提供了调用 API 所需的关键参数信息？
   - 1 分: 所需参数信息齐全或可从上下文合理推断
   - 0.5 分: 缺少部分非核心参数，但核心参数完整
   - 0 分: 缺少关键参数，agent 无法合理补全

3. 表述质量 (Expression Quality): query 的语法和表述是否规范？
   - 1 分: 语句通顺、无语法错误、表述自然
   - 0.5 分: 有小瑕疵但不影响理解
   - 0 分: 表述混乱、严重语法错误、或明显是乱码/模板占位符

请以 JSON 格式输出：
```json
{{
    "intent_clarity": {{
        "score": 0/0.5/1,
        "reason": "简要说明"
    }},
    "param_sufficiency": {{
        "score": 0/0.5/1,
        "reason": "简要说明"
    }},
    "expression_quality": {{
        "score": 0/0.5/1,
        "reason": "简要说明"
    }},
    "is_low_quality": true/false,
    "overall_reason": "一句话总结该 query 的质量问题（如无问题写'质量合格'）"
}}
```

只输出 JSON，不要其他内容。"""


def _call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ OpenAI 库未安装，请运行: pip install openai")
        return None

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️ LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def _parse_clarity_response(response: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 query clarity JSON。"""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        required = ["intent_clarity", "param_sufficiency", "expression_quality"]
        for key in required:
            if key not in data or "score" not in data[key]:
                return None
        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def evaluate_query_clarity(
    sample: APIAgentSample,
) -> Dict[str, Any]:
    """
    对单条样本的 query 做 LLM judge 清晰度评估。

    Returns:
        包含三个子维度分数、is_low_quality 标记和 overall_reason 的字典。
        LLM 调用失败时返回 None 标记的结果。
    """
    query = (sample.query or "").strip()
    if not query:
        return {
            "intent_clarity": 0.0,
            "param_sufficiency": 0.0,
            "expression_quality": 0.0,
            "is_low_quality": True,
            "overall_reason": "query 为空",
            "llm_success": False,
        }

    tool_names = ", ".join(
        t.name for t in (sample.available_tools or []) if t.name
    ) or "(无工具定义)"

    prompt = QUERY_CLARITY_PROMPT.format(query=query, tool_names=tool_names)
    raw = _call_llm(prompt)
    parsed = _parse_clarity_response(raw)

    if parsed is None:
        return {
            "intent_clarity": None,
            "param_sufficiency": None,
            "expression_quality": None,
            "is_low_quality": None,
            "overall_reason": "LLM 解析失败",
            "llm_success": False,
        }

    intent = float(parsed["intent_clarity"]["score"])
    param = float(parsed["param_sufficiency"]["score"])
    expr = float(parsed["expression_quality"]["score"])
    is_low = parsed.get("is_low_quality", (intent + param + expr) / 3.0 < 0.5)

    return {
        "intent_clarity": intent,
        "param_sufficiency": param,
        "expression_quality": expr,
        "is_low_quality": bool(is_low),
        "overall_reason": parsed.get("overall_reason", ""),
        "llm_success": True,
        "detail": parsed,
    }


def compute_query_clarity(
    data_iterator: Iterator[APIAgentSample],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    批量评估 query 清晰度（LLM judge），独立于 compute_task_complexity 调用。

    Returns:
        数据集级别的 query 清晰度统计。
    """
    print("=" * 70)
    print("Query Clarity 查询清晰度评估 (LLM Judge)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    start_time = time.time()
    n_samples = 0
    n_llm_success = 0

    intent_scores: List[float] = []
    param_scores: List[float] = []
    expr_scores: List[float] = []
    n_low_quality = 0
    low_quality_examples: List[Dict[str, Any]] = []

    for sample in tqdm(data_iterator, desc="Evaluating query clarity"):
        if max_samples is not None and n_samples >= max_samples:
            break
        n_samples += 1

        result = evaluate_query_clarity(sample)

        if result["llm_success"]:
            n_llm_success += 1
            intent_scores.append(result["intent_clarity"])
            param_scores.append(result["param_sufficiency"])
            expr_scores.append(result["expression_quality"])
            if result["is_low_quality"]:
                n_low_quality += 1
                low_quality_examples.append({
                    "query": sample.query or "",
                    "reason": result["overall_reason"],
                    "scores": {
                        "intent": result["intent_clarity"],
                        "param": result["param_sufficiency"],
                        "expr": result["expression_quality"],
                    },
                })

    elapsed = time.time() - start_time

    avg_intent = float(np.mean(intent_scores)) if intent_scores else 0.0
    avg_param = float(np.mean(param_scores)) if param_scores else 0.0
    avg_expr = float(np.mean(expr_scores)) if expr_scores else 0.0
    avg_overall = (avg_intent + avg_param + avg_expr) / 3.0

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_samples": n_samples,
        "n_llm_success": n_llm_success,
        "llm_success_rate": n_llm_success / n_samples if n_samples > 0 else 0.0,

        "overall_clarity_score": avg_overall,
        "intent_clarity_score": avg_intent,
        "param_sufficiency_score": avg_param,
        "expression_quality_score": avg_expr,

        "n_low_quality": n_low_quality,
        "low_quality_ratio": n_low_quality / n_llm_success if n_llm_success > 0 else 0.0,

        "score_distributions": {
            "intent_clarity": _summary_stats(intent_scores),
            "param_sufficiency": _summary_stats(param_scores),
            "expression_quality": _summary_stats(expr_scores),
        },

        "low_quality_examples": low_quality_examples,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒 | 样本 {n_samples} | LLM 成功 {n_llm_success}")
    print("=" * 70)
    print(f"  ★ 综合清晰度:   {avg_overall:.3f}")
    print(f"    意图明确性:    {avg_intent:.3f}")
    print(f"    参数充分性:    {avg_param:.3f}")
    print(f"    表述质量:      {avg_expr:.3f}")
    print(f"  低质量样本: {n_low_quality} ({100*n_low_quality/n_llm_success:.1f}%)" if n_llm_success > 0 else "")
    print()
    if low_quality_examples:
        print("【低质量样本示例】")
        for i, ex in enumerate(low_quality_examples, 1):
            print(f"  {i}. [{ex['scores']}] {ex['query']}")
            print(f"     原因: {ex['reason']}")
    print("=" * 70)

    return results


# =============================================================================
# 主函数
# =============================================================================

def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def compute_task_complexity(
    data_iterator: Iterator[APIAgentSample],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    enable_query_clarity: bool = False,
) -> Dict[str, Any]:
    """
    计算数据集级别的任务难度指标。

    静态维度（A/B/C）始终计算；维度 D（Query 清晰度）需要 LLM API，
    通过 enable_query_clarity=True 开启。

    Args:
        data_iterator: APIAgentSample 迭代器
        dataset_name: 数据集名称
        output_file: 若指定，则将结果以 JSON 格式写入该路径
        max_samples: 最多评估的样本数
        enable_query_clarity: 是否开启 LLM judge 评估 query 清晰度

    Returns:
        包含多维度难度评估 + 综合分数的字典
    """
    print("=" * 70)
    print("Task Complexity 任务复杂度评估")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    start_time = time.time()
    n_samples = 0

    # 每个维度的分数列表
    sel_scores: List[float] = []
    param_scores: List[float] = []
    plan_scores: List[float] = []

    # 分布计数
    sel_counter = Counter()
    param_counter = Counter()
    plan_counter = Counter()

    # 原始特征列表（用于统计）
    n_candidate_list: List[int] = []
    n_distinct_used_list: List[int] = []
    n_args_filled_list: List[int] = []
    chain_length_list: List[int] = []
    dep_depth_list: List[int] = []
    dep_distance_list: List[float] = []
    sel_interference_list: List[float] = []

    # 维度 D (可选)
    clarity_intent_scores: List[float] = []
    clarity_param_scores: List[float] = []
    clarity_expr_scores: List[float] = []
    n_clarity_low = 0
    n_clarity_success = 0
    clarity_low_examples: List[Dict[str, Any]] = []

    if enable_query_clarity:
        print("已开启 Query Clarity LLM Judge 评估\n")

    for sample in tqdm(data_iterator, desc="Computing task difficulty"):
        if max_samples is not None and n_samples >= max_samples:
            break
        n_samples += 1

        real_calls = _get_real_calls(sample)

        # --- A. 工具选择 ---
        sel = _extract_tool_selection(sample, real_calls)
        sel_scores.append(sel["score"])
        sel_counter[sel["label"]] += 1
        n_candidate_list.append(sel["n_candidate_tools"])
        n_distinct_used_list.append(sel["n_distinct_tools_used"])

        # --- B. 参数填充 ---
        param = _extract_param_filling(sample, real_calls)
        param_scores.append(param["score"])
        param_counter[param["label"]] += 1
        n_args_filled_list.append(param["n_args_filled_total"])

        # --- C. 多步规划 ---
        plan = _extract_planning(real_calls)
        plan_scores.append(plan["score"])
        plan_counter[plan["label"]] += 1
        chain_length_list.append(plan["chain_length"])
        dep_depth_list.append(plan["dependency_depth"])
        dep_distance_list.append(plan["avg_dependency_distance"])
        sel_interference_list.append(plan["avg_selection_interference"])

        # --- D. Query 清晰度 (可选, LLM Judge) ---
        if enable_query_clarity:
            cla = evaluate_query_clarity(sample)
            if cla["llm_success"]:
                n_clarity_success += 1
                clarity_intent_scores.append(cla["intent_clarity"])
                clarity_param_scores.append(cla["param_sufficiency"])
                clarity_expr_scores.append(cla["expression_quality"])
                if cla["is_low_quality"]:
                    n_clarity_low += 1
                    clarity_low_examples.append({
                        "query": sample.query or "",
                        "reason": cla["overall_reason"],
                    })

    elapsed = time.time() - start_time

    # 各维度平均分
    avg_sel = float(np.mean(sel_scores)) if sel_scores else 0.0
    avg_param = float(np.mean(param_scores)) if param_scores else 0.0
    avg_plan = float(np.mean(plan_scores)) if plan_scores else 0.0

    # 综合难度（三维度等权平均）
    overall_score = (avg_sel + avg_param + avg_plan) / 3.0

    # 构建结果
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_samples": n_samples,

        # ========== 核心分数 ==========
        "overall_difficulty_score": overall_score,
        "tool_selection_score": avg_sel,
        "param_filling_score": avg_param,
        "planning_score": avg_plan,

        # ========== 维度 A: 工具选择难度 ==========
        "tool_selection_difficulty": {
            "score": avg_sel,
            "raw_stats": {
                "n_candidate_tools": _summary_stats(n_candidate_list),
                "n_distinct_tools_used": _summary_stats(n_distinct_used_list),
            },
            "distribution": {
                "easy": sel_counter["easy"],
                "moderate": sel_counter["moderate"],
                "hard": sel_counter["hard"],
            },
            "ratios": {
                "easy": sel_counter["easy"] / n_samples if n_samples > 0 else 0,
                "moderate": sel_counter["moderate"] / n_samples if n_samples > 0 else 0,
                "hard": sel_counter["hard"] / n_samples if n_samples > 0 else 0,
            },
        },

        # ========== 维度 B: 参数填充难度 ==========
        "param_filling_difficulty": {
            "score": avg_param,
            "raw_stats": {
                "n_args_filled": _summary_stats(n_args_filled_list),
            },
            "distribution": {
                "easy": param_counter["easy"],
                "moderate": param_counter["moderate"],
                "hard": param_counter["hard"],
            },
            "ratios": {
                "easy": param_counter["easy"] / n_samples if n_samples > 0 else 0,
                "moderate": param_counter["moderate"] / n_samples if n_samples > 0 else 0,
                "hard": param_counter["hard"] / n_samples if n_samples > 0 else 0,
            },
        },

        # ========== 维度 C: 多步规划难度 ==========
        "planning_difficulty": {
            "score": avg_plan,
            "raw_stats": {
                "chain_length": _summary_stats(chain_length_list),
                "dependency_depth": _summary_stats(dep_depth_list),
                "avg_dependency_distance": _summary_stats(dep_distance_list),
                "avg_selection_interference": _summary_stats(sel_interference_list),
            },
            "distribution": {
                "easy": plan_counter["easy"],
                "moderate": plan_counter["moderate"],
                "hard": plan_counter["hard"],
            },
            "ratios": {
                "easy": plan_counter["easy"] / n_samples if n_samples > 0 else 0,
                "moderate": plan_counter["moderate"] / n_samples if n_samples > 0 else 0,
                "hard": plan_counter["hard"] / n_samples if n_samples > 0 else 0,
            },
        },
    }

    # 维度 D (可选)
    if enable_query_clarity:
        avg_clarity_intent = float(np.mean(clarity_intent_scores)) if clarity_intent_scores else 0.0
        avg_clarity_param = float(np.mean(clarity_param_scores)) if clarity_param_scores else 0.0
        avg_clarity_expr = float(np.mean(clarity_expr_scores)) if clarity_expr_scores else 0.0
        avg_clarity = (avg_clarity_intent + avg_clarity_param + avg_clarity_expr) / 3.0

        results["query_clarity"] = {
            "overall_score": avg_clarity,
            "intent_clarity": avg_clarity_intent,
            "param_sufficiency": avg_clarity_param,
            "expression_quality": avg_clarity_expr,
            "n_llm_success": n_clarity_success,
            "n_low_quality": n_clarity_low,
            "low_quality_ratio": n_clarity_low / n_clarity_success if n_clarity_success > 0 else 0.0,
            "score_distributions": {
                "intent_clarity": _summary_stats(clarity_intent_scores),
                "param_sufficiency": _summary_stats(clarity_param_scores),
                "expression_quality": _summary_stats(clarity_expr_scores),
            },
            "low_quality_examples": clarity_low_examples,
        }
        results["query_clarity_score"] = avg_clarity

    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print(f"  总样本数: {n_samples:,}")
    print()

    print("=" * 70)
    print("【核心分数】")
    print(f"  ★ 综合难度:     {overall_score:.3f}")
    print(f"  ★ 工具选择难度: {avg_sel:.3f}")
    print(f"  ★ 参数填充难度: {avg_param:.3f}")
    print(f"  ★ 多步规划难度: {avg_plan:.3f}")
    print("=" * 70)
    print()

    # 维度 A
    sel_easy_r = sel_counter["easy"] / n_samples if n_samples > 0 else 0
    sel_mod_r = sel_counter["moderate"] / n_samples if n_samples > 0 else 0
    sel_hard_r = sel_counter["hard"] / n_samples if n_samples > 0 else 0
    print("【工具选择难度详情】(基于候选工具池大小)")
    print(f"  平均候选工具数: {np.mean(n_candidate_list):.1f} | "
          f"平均实际使用: {np.mean(n_distinct_used_list):.1f}")
    print()
    print("  分布:")
    print(f"    - easy (1-3 工具, 0.2分):    {sel_counter['easy']:,} ({100*sel_easy_r:.1f}%)")
    print(f"    - moderate (4-10, 0.5分):    {sel_counter['moderate']:,} ({100*sel_mod_r:.1f}%)")
    print(f"    - hard (>10 工具, 1分):      {sel_counter['hard']:,} ({100*sel_hard_r:.1f}%)")
    print()

    # 维度 B
    par_easy_r = param_counter["easy"] / n_samples if n_samples > 0 else 0
    par_mod_r = param_counter["moderate"] / n_samples if n_samples > 0 else 0
    par_hard_r = param_counter["hard"] / n_samples if n_samples > 0 else 0
    print("【参数填充难度详情】(基于参数量 + 嵌套类型)")
    print(f"  平均填充参数数: {np.mean(n_args_filled_list):.1f}")
    print()
    print("  分布:")
    print(f"    - easy (0-2 参数, 0.2分):        {param_counter['easy']:,} ({100*par_easy_r:.1f}%)")
    print(f"    - moderate (3-8 参数, 0.5分):    {param_counter['moderate']:,} ({100*par_mod_r:.1f}%)")
    print(f"    - hard (>8 或嵌套, 1分):         {param_counter['hard']:,} ({100*par_hard_r:.1f}%)")
    print()

    # 维度 C
    pln_easy_r = plan_counter["easy"] / n_samples if n_samples > 0 else 0
    pln_mod_r = plan_counter["moderate"] / n_samples if n_samples > 0 else 0
    pln_hard_r = plan_counter["hard"] / n_samples if n_samples > 0 else 0
    n_has_dep = sum(1 for d in dep_depth_list if d >= 1)
    avg_dep_dist = float(np.mean(dep_distance_list)) if dep_distance_list else 0.0
    avg_sel_interf = float(np.mean(sel_interference_list)) if sel_interference_list else 0.0
    print("【多步规划难度详情】(基于调用链长 + 数据依赖 + 认知负荷)")
    print(f"  平均链长: {np.mean(chain_length_list):.1f} | "
          f"有数据依赖的样本: {n_has_dep:,} ({100*n_has_dep/n_samples:.1f}%)")
    print(f"  平均注意力距离 (Memory Load): {avg_dep_dist:.2f} | "
          f"平均选择干扰度 (Selection Load): {avg_sel_interf:.2f}")
    print()
    print("  分布:")
    print(f"    - easy (单步, 0.2分):            {plan_counter['easy']:,} ({100*pln_easy_r:.1f}%)")
    print(f"    - moderate (多步无依赖, 0.5分):  {plan_counter['moderate']:,} ({100*pln_mod_r:.1f}%)")
    print(f"    - hard (多步+依赖, 1分):         {plan_counter['hard']:,} ({100*pln_hard_r:.1f}%)")
    print()

    # 维度 D (可选)
    if enable_query_clarity and n_clarity_success > 0:
        avg_ci = float(np.mean(clarity_intent_scores))
        avg_cp = float(np.mean(clarity_param_scores))
        avg_ce = float(np.mean(clarity_expr_scores))
        low_r = n_clarity_low / n_clarity_success
        print("【Query 清晰度详情】(LLM Judge)")
        print(f"  LLM 评估成功: {n_clarity_success}/{n_samples}")
        print(f"  意图明确性: {avg_ci:.3f} | 参数充分性: {avg_cp:.3f} | 表述质量: {avg_ce:.3f}")
        print(f"  低质量样本: {n_clarity_low} ({100*low_r:.1f}%)")
        if clarity_low_examples:
            print("  低质量示例:")
            for i, ex in enumerate(clarity_low_examples, 1):
                print(f"    {i}. {ex['query']}")
                print(f"       原因: {ex['reason']}")
        print()

    # 综合解读
    print("【综合解读】")
    dims = [("工具选择", avg_sel), ("参数填充", avg_param), ("多步规划", avg_plan)]
    if enable_query_clarity and n_clarity_success > 0:
        dims.append(("Query清晰度", results.get("query_clarity", {}).get("overall_score", 0.0)))
    for dim_name, dim_score in dims:
        if dim_score < 0.6:
            print(f"  {dim_name}: 偏易，大部分样本较简单")
        elif dim_score > 0.85:
            print(f"  {dim_name}: 较难，有助于训练高阶能力")
        else:
            print(f"  {dim_name}: 难度适中")
    print("=" * 70)

    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    from loaders import ToolBenchLoader, XLAMLoader, ArceeAgentDataLoader  # noqa: E402

    parser = argparse.ArgumentParser(description="Task Complexity 任务复杂度评估")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["toolbench", "xlam", "arcee"],
        help="数据集名称",
    )
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多评估的样本数")
    parser.add_argument("--enable-query-clarity", action="store_true",
                        help="开启 LLM Judge 评估 Query 清晰度 (维度 D)")

    args = parser.parse_args()

    if args.dataset == "toolbench":
        default_path = "/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json"
        data_path = args.data_path or default_path
        loader = ToolBenchLoader(data_path)
        iterator = loader.iterate()
        dataset_name = "ToolBench"
    elif args.dataset == "xlam":
        default_path = "/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl"
        data_path = args.data_path or default_path
        loader = XLAMLoader(data_path)
        iterator = loader.iterate()
        dataset_name = "xLAM-60k"
    else:
        default_path = "/mnt/petrelfs/liuhaoze/datasets/Agent_Data/agent-data/arcee_agent_data_api_only.jsonl"
        data_path = args.data_path or default_path
        loader = ArceeAgentDataLoader(data_path)
        iterator = loader.iterate()
        dataset_name = "Arcee-AgentData"

    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/task_complexity_results.json"

    compute_task_complexity(
        data_iterator=iterator,
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
        enable_query_clarity=args.enable_query_clarity,
    )
