#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Complexity 任务复杂度指标

三个维度评估 API 调用数据的任务复杂度，每个维度输出一个 0.2/0.5/1.0 的难度分数：

1. 工具选择难度（Tool Selection Difficulty）
   - 候选工具池越大，选对工具越难
   - 1-3 个候选 → easy (0.2)
   - 4-10 个候选 → moderate (0.5)
   - >10 个候选 → hard (1.0)

2. 参数填充难度（Param Filling Difficulty）
   - 需要填的参数越多、有嵌套类型越难
   - 0-2 个参数 → easy (0.2)
   - 3-8 个参数 → moderate (0.5)
   - >8 个参数或含嵌套类型 → hard (1.0)

3. 多步规划难度（Planning Difficulty）
   - 基于调用链长度 + 步间数据依赖检测
   - 单步调用 → easy (0.2)
   - 多步但无数据依赖 → moderate (0.5)
   - 多步且有数据依赖 → hard (1.0)

使用方式:

    from loaders import ToolBenchLoader
    from metrics.task_complexity import compute_task_complexity

    loader = ToolBenchLoader('/path/to/toolbench.json')

    results = compute_task_complexity(
        data_iterator=loader.iterate(),
        dataset_name='ToolBench',
        max_samples=10000,
        output_file='results/toolbench/task_complexity.json',
    )
"""

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


def _extract_planning(real_calls: List[APICall]) -> Dict[str, Any]:
    """
    维度 C: 多步规划

    依赖检测: 如果 call[i] 的某个 argument 值出现在 call[j] (j < i) 的 response 中，
    说明存在数据依赖（后续调用用到了前面调用的结果）。
    """
    chain_length = len(real_calls)
    n_distinct = len(set((c.name or "").strip() for c in real_calls))

    dep_depth = 0
    dep_links = 0
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
                    break

    unique_api_ratio = n_distinct / chain_length if chain_length > 0 else 0.0
    score = planning_to_score(chain_length, dep_depth)
    return {
        "chain_length": chain_length,
        "n_distinct_apis": n_distinct,
        "unique_api_ratio": unique_api_ratio,
        "dependency_depth": dep_depth,
        "dependency_links": dep_links,
        "score": score,
        "label": score_to_label(score),
    }


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
) -> Dict[str, Any]:
    """
    计算数据集级别的任务难度指标（三个维度 + 综合分数）。

    Args:
        data_iterator: APIAgentSample 迭代器
        dataset_name: 数据集名称
        output_file: 若指定，则将结果以 JSON 格式写入该路径
        max_samples: 最多评估的样本数

    Returns:
        包含三维度难度评估 + 综合分数的字典
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
    print("【多步规划难度详情】(基于调用链长 + 数据依赖)")
    print(f"  平均链长: {np.mean(chain_length_list):.1f} | "
          f"有数据依赖的样本: {n_has_dep:,} ({100*n_has_dep/n_samples:.1f}%)")
    print()
    print("  分布:")
    print(f"    - easy (单步, 0.2分):            {plan_counter['easy']:,} ({100*pln_easy_r:.1f}%)")
    print(f"    - moderate (多步无依赖, 0.5分):  {plan_counter['moderate']:,} ({100*pln_mod_r:.1f}%)")
    print(f"    - hard (多步+依赖, 1分):         {plan_counter['hard']:,} ({100*pln_hard_r:.1f}%)")
    print()

    # 综合解读
    print("【综合解读】")
    for dim_name, dim_score in [("工具选择", avg_sel), ("参数填充", avg_param), ("多步规划", avg_plan)]:
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
    )
