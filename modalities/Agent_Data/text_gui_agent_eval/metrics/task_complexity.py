#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Complexity 指标 - 任务复杂度评估

四个维度评估 GUI Agent 训练数据的任务复杂度：

A. 目标定位难度 (Localization Difficulty)
   基于 PATHWAYS 论文的发现，通过目标元素在 DOM 树中的深度评估：
   - 0-3 → surface (0.5) | 4-9 → moderate (0.75) | >9 → deep (1.0)

B. 完成难度 (Completion Difficulty)
   基于动作步数：
   - 1-5 步 → easy (0.5) | 6-15 步 → moderate (0.75) | 16+ 步 → hard (1.0)

C. 动作类型多样性 (Action Type Diversity)
   任务需要混合多少种操作类型 (click/type/scroll/select/hover...)：
   - 1-3 种 → simple (0.2) | 3-6 种 → moderate (0.5) | >6 种 → complex (1.0)
   混合操作任务对 agent 的能力要求更全面。

D. 指令清晰度 (Instruction Clarity, LLM Judge, 可选)
   通过 LLM 评估 instruction 作为训练数据的质量：
   - 意图明确性 / 操作可推断性 / 表述质量
   不清晰的指令 = 低质量训练数据。

使用方式:
    from metrics.task_complexity import compute_task_complexity
    from loaders import Mind2WebLoader
    from executor.mind2web import Mind2WebLocator

    loader = Mind2WebLoader('/path/to/data')
    locator = Mind2WebLocator()

    results = compute_task_complexity(
        data_iterator=loader.iterate(),
        locator=locator,
        dataset_name='Mind2Web',
        output_file='task_complexity_results.json',
        max_samples=100,
        enable_instruction_clarity=True,  # 开启 LLM judge (需要 API)
    )
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from collections import Counter

import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import Record
from text_gui_executor import HTMLLocator

# =============================================================================
# LLM 配置（用于 Instruction Clarity 评估）
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4.1'


# =============================================================================
# 深度到分数的映射
# =============================================================================

def depth_to_score(depth: int) -> float:
    """
    将 DOM 深度映射到分数
    
    阈值定义：
    - 0-3: 低复杂度（表面可见）
    - 4-9: 中等复杂度
    - >9: 高复杂度（深层隐藏）
    
    Args:
        depth: DOM 深度（-1 表示未找到）
        
    Returns:
        分数：
        - 0: 找不到
        - 0.5: 表面（深度 0-3）
        - 0.75: 中等（深度 4-9）
        - 1: 深层（深度 > 9）
    """
    if depth < 0:
        return 0.0
    elif depth <= 3:
        return 0.5
    elif depth <= 9:
        return 0.75
    else:
        return 1.0


def score_to_label(score: float) -> str:
    """将分数转换为标签"""
    if score == 0:
        return "not_found"
    elif score == 0.5:
        return "surface"
    elif score == 0.75:
        return "moderate"
    else:
        return "deep"


def steps_to_score(steps: int) -> float:
    """
    将动作步数映射到完成难度分数
    
    Args:
        steps: 动作步数
        
    Returns:
        分数：
        - 0.5: 简单 (1-5 步)
        - 0.75: 中等 (6-15 步)
        - 1.0: 困难 (16+ 步)
    """
    if steps <= 5:
        return 0.5
    elif steps <= 15:
        return 0.75
    else:
        return 1.0


def steps_score_to_label(score: float) -> str:
    """将步数分数转换为标签"""
    if score == 0.5:
        return "easy"
    elif score == 0.75:
        return "moderate"
    else:
        return "hard"


# =============================================================================
# 维度 C: 动作类型多样性
# =============================================================================

def action_diversity_to_score(n_types: int) -> float:
    """
    将动作类型种类数映射到复杂度分数。

    1-3 种 → 0.2 (简单)
    3-6 种 → 0.5 (中等)
    >6 种 → 1.0 (复杂)
    """
    if n_types <= 3:
        return 0.2
    elif n_types <= 6:
        return 0.5
    else:
        return 1.0


def action_diversity_label(score: float) -> str:
    if score <= 0.2:
        return "simple"
    elif score <= 0.5:
        return "moderate"
    else:
        return "complex"


def _extract_action_diversity(record: Record) -> Dict[str, Any]:
    """
    分析一个 record 的动作类型多样性。

    返回种类数、类型分布、以及 entropy 作为连续量化指标。
    """
    type_counts = Counter(a.action_type for a in record.actions)
    n_types = len(type_counts)
    total = sum(type_counts.values())

    entropy = 0.0
    if total > 0 and n_types > 1:
        for cnt in type_counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)

    score = action_diversity_to_score(n_types)
    return {
        "n_action_types": n_types,
        "action_type_distribution": dict(type_counts),
        "action_type_entropy": entropy,
        "score": score,
        "label": action_diversity_label(score),
    }


# =============================================================================
# 维度 D: 指令清晰度 (LLM Judge, 可选)
# =============================================================================

INSTRUCTION_CLARITY_PROMPT = """你是一个 GUI Agent 训练数据质量评估专家。请评估以下指令（instruction）作为 Agent 训练数据时的清晰度。

【指令】
{instruction}

【目标网站】
{website}

【动作序列摘要】
共 {n_actions} 步，动作类型: {action_types}

请从以下三个方面评估：

1. 意图明确性 (Intent Clarity): 指令是否清晰表达了用户想在网页上完成的任务？
   - 1 分: 意图清晰，一个合格的 agent 能明确知道该做什么
   - 0.5 分: 基本能理解，但存在部分模糊
   - 0 分: 意图严重不清，无法确定用户到底想做什么

2. 操作可推断性 (Action Inferability): 从指令出发，agent 是否能合理推断出需要执行的操作？
   - 1 分: 操作路径可从指令自然推导
   - 0.5 分: 大部分操作可推断，但部分步骤需要猜测
   - 0 分: 指令与实际操作之间存在严重断层

3. 表述质量 (Expression Quality): 指令的语法和表述是否规范？
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
    "action_inferability": {{
        "score": 0/0.5/1,
        "reason": "简要说明"
    }},
    "expression_quality": {{
        "score": 0/0.5/1,
        "reason": "简要说明"
    }},
    "is_low_quality": true/false,
    "overall_reason": "一句话总结（如无问题写'质量合格'）"
}}
```

只输出 JSON，不要其他内容。"""


def _call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ OpenAI 库未安装，请运行: pip install openai")
        return None

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
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
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        for key in ["intent_clarity", "action_inferability", "expression_quality"]:
            if key not in data or "score" not in data[key]:
                return None
        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def evaluate_instruction_clarity(record: Record) -> Dict[str, Any]:
    """对单条 record 的 instruction 做 LLM judge 清晰度评估。"""
    instruction = (record.instruction or "").strip()
    if not instruction:
        return {
            "intent_clarity": 0.0,
            "action_inferability": 0.0,
            "expression_quality": 0.0,
            "is_low_quality": True,
            "overall_reason": "instruction 为空",
            "llm_success": False,
        }

    website = record.website or "(未知)"
    n_actions = len(record.actions)
    action_types = ", ".join(sorted(set(a.action_type for a in record.actions)))

    prompt = INSTRUCTION_CLARITY_PROMPT.format(
        instruction=instruction,
        website=website,
        n_actions=n_actions,
        action_types=action_types,
    )
    raw = _call_llm(prompt)
    parsed = _parse_clarity_response(raw)

    if parsed is None:
        return {
            "intent_clarity": None,
            "action_inferability": None,
            "expression_quality": None,
            "is_low_quality": None,
            "overall_reason": "LLM 解析失败",
            "llm_success": False,
        }

    intent = float(parsed["intent_clarity"]["score"])
    infer = float(parsed["action_inferability"]["score"])
    expr = float(parsed["expression_quality"]["score"])
    is_low = parsed.get("is_low_quality", (intent + infer + expr) / 3.0 < 0.5)

    return {
        "intent_clarity": intent,
        "action_inferability": infer,
        "expression_quality": expr,
        "is_low_quality": bool(is_low),
        "overall_reason": parsed.get("overall_reason", ""),
        "llm_success": True,
        "detail": parsed,
    }


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


# =============================================================================
# 主函数
# =============================================================================

def compute_task_complexity(
    data_iterator: Iterator[Record],
    locator: HTMLLocator,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
    use_cleaned_html: bool = True,
    enable_instruction_clarity: bool = False,
) -> Dict[str, Any]:
    """
    计算数据集级别的任务复杂度（四个维度）。

    维度 A/B/C 为静态分析，维度 D 需要 LLM API（可选开启）。

    Args:
        data_iterator: Record 迭代器
        locator: HTMLLocator 实例（需实现 locate_with_depth 方法）
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
        use_cleaned_html: 是否使用 cleaned_html（默认 True），否则使用 raw_html
        enable_instruction_clarity: 是否开启 LLM judge 评估指令清晰度
    """
    print("=" * 70)
    print("Task Complexity Evaluation (任务复杂度)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"HTML 类型: {'cleaned_html' if use_cleaned_html else 'raw_html'}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    if enable_instruction_clarity:
        print("已开启 Instruction Clarity LLM Judge 评估")
    print()

    start_time = time.time()

    # 统计
    total_records = 0
    total_actions = 0
    total_depth = 0
    total_score = 0.0

    depth_counter = Counter()  # 深度分布
    score_counter = Counter()  # 分数分布
    reason_counter = Counter()  # 定位结果分布
    step_counter = Counter()   # 每个 record 的动作步数分布
    all_step_counts = []       # 所有 record 的步数列表

    # 维度 C: 动作类型多样性
    diversity_scores: List[float] = []
    diversity_counter = Counter()
    n_types_list: List[int] = []
    entropy_list: List[float] = []
    global_action_type_counter = Counter()

    # 维度 D: 指令清晰度 (可选)
    clarity_intent_scores: List[float] = []
    clarity_infer_scores: List[float] = []
    clarity_expr_scores: List[float] = []
    n_clarity_success = 0
    n_clarity_low = 0
    clarity_low_examples: List[Dict[str, Any]] = []

    action_results = []  # 每个 action 的详细结果
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 记录当前 record 的动作步数
        record_step_count = len(record.actions)
        all_step_counts.append(record_step_count)
        step_counter[record_step_count] += 1
        
        for action in record.actions:
            # 获取 HTML
            html = action.cleaned_html if use_cleaned_html else action.raw_html
            
            # 定位并获取深度
            success, depth, reason = locator.locate_with_depth(action, html)
            
            # 跳过不需要定位的操作（say, scroll, load 等）
            # 这些操作没有目标元素，不应计入任务复杂度
            if reason == "no_uid_required":
                continue
            
            total_actions += 1
            
            # 计算分数
            score = depth_to_score(depth)
            label = score_to_label(score)
            
            # 统计
            if success and depth >= 0:
                total_depth += depth
                depth_counter[depth] += 1
            
            total_score += score
            score_counter[label] += 1
            reason_counter[reason] += 1
            
            action_results.append({
                'sample_id': record.sample_id,
                'action_type': action.action_type,
                'success': success,
                'depth': depth,
                'score': score,
                'label': label,
                'reason': reason,
            })
        
        # --- C. 动作类型多样性 ---
        div = _extract_action_diversity(record)
        diversity_scores.append(div["score"])
        diversity_counter[div["label"]] += 1
        n_types_list.append(div["n_action_types"])
        entropy_list.append(div["action_type_entropy"])
        for atype, cnt in div["action_type_distribution"].items():
            global_action_type_counter[atype] += cnt

        # --- D. 指令清晰度 (可选, LLM Judge) ---
        if enable_instruction_clarity:
            cla = evaluate_instruction_clarity(record)
            if cla["llm_success"]:
                n_clarity_success += 1
                clarity_intent_scores.append(cla["intent_clarity"])
                clarity_infer_scores.append(cla["action_inferability"])
                clarity_expr_scores.append(cla["expression_quality"])
                if cla["is_low_quality"]:
                    n_clarity_low += 1
                    clarity_low_examples.append({
                        "instruction": record.instruction or "",
                        "reason": cla["overall_reason"],
                    })

        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            avg_score = total_score / total_actions if total_actions > 0 else 0
            print(f"  [{total_records:,} records, {total_actions:,} actions] "
                  f"{rate:.1f} rec/s | avg_score: {avg_score:.3f}")

    elapsed = time.time() - start_time
    
    # 计算统计值
    found_actions = sum(score_counter[k] for k in ['surface', 'moderate', 'deep'])
    avg_depth = total_depth / found_actions if found_actions > 0 else 0
    avg_score = total_score / total_actions if total_actions > 0 else 0
    
    # 各类别比例
    surface_ratio = score_counter['surface'] / total_actions if total_actions > 0 else 0
    moderate_ratio = score_counter['moderate'] / total_actions if total_actions > 0 else 0
    deep_ratio = score_counter['deep'] / total_actions if total_actions > 0 else 0
    not_found_ratio = score_counter['not_found'] / total_actions if total_actions > 0 else 0
    
    # 动作步数统计
    avg_steps = sum(all_step_counts) / len(all_step_counts) if all_step_counts else 0
    min_steps = min(all_step_counts) if all_step_counts else 0
    max_steps = max(all_step_counts) if all_step_counts else 0
    median_steps = sorted(all_step_counts)[len(all_step_counts) // 2] if all_step_counts else 0
    
    # 步数区间分布 (按难度分)
    steps_easy = sum(1 for s in all_step_counts if s <= 5)       # 简单: 1-5 步
    steps_moderate = sum(1 for s in all_step_counts if 6 <= s <= 15)  # 中等: 6-15 步
    steps_hard = sum(1 for s in all_step_counts if s > 15)       # 困难: 16+ 步
    
    # 计算完成难度分数 (每个 record 的步数分数的平均值)
    completion_scores = [steps_to_score(s) for s in all_step_counts]
    avg_completion_score = sum(completion_scores) / len(completion_scores) if completion_scores else 0
    
    # 维度 C 统计
    avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
    avg_n_types = float(np.mean(n_types_list)) if n_types_list else 0.0
    avg_entropy = float(np.mean(entropy_list)) if entropy_list else 0.0

    # 构建结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'html_type': 'cleaned_html' if use_cleaned_html else 'raw_html',

        # 基本统计
        'total_records': total_records,
        'total_actions': total_actions,
        'found_actions': found_actions,

        # ========== 核心分数 ==========
        'localization_difficulty_score': avg_score,
        'completion_difficulty_score': avg_completion_score,
        'action_diversity_score': avg_diversity,

        # ========== 维度 A: 目标定位难度 ==========
        'localization_difficulty': {
            'score': avg_score,
            'avg_depth': avg_depth,
            'distribution': {
                'surface': score_counter['surface'],
                'moderate': score_counter['moderate'],
                'deep': score_counter['deep'],
                'not_found': score_counter['not_found'],
            },
            'ratios': {
                'surface': surface_ratio,
                'moderate': moderate_ratio,
                'deep': deep_ratio,
                'not_found': not_found_ratio,
            },
            'depth_distribution': dict(sorted(depth_counter.items())),
        },

        # ========== 维度 B: 完成难度 ==========
        'completion_difficulty': {
            'score': avg_completion_score,
            'avg_steps': avg_steps,
            'min_steps': min_steps,
            'max_steps': max_steps,
            'median_steps': median_steps,
            'distribution': {
                'easy': steps_easy,
                'moderate': steps_moderate,
                'hard': steps_hard,
            },
            'ratios': {
                'easy': steps_easy / total_records if total_records > 0 else 0,
                'moderate': steps_moderate / total_records if total_records > 0 else 0,
                'hard': steps_hard / total_records if total_records > 0 else 0,
            },
            'detailed_distribution': dict(sorted(step_counter.items())),
        },

        # ========== 维度 C: 动作类型多样性 ==========
        'action_type_diversity': {
            'score': avg_diversity,
            'avg_n_types': avg_n_types,
            'avg_entropy': avg_entropy,
            'raw_stats': {
                'n_action_types': _summary_stats(n_types_list),
                'action_type_entropy': _summary_stats(entropy_list),
            },
            'distribution': {
                'simple': diversity_counter['simple'],
                'moderate': diversity_counter['moderate'],
                'complex': diversity_counter['complex'],
            },
            'ratios': {
                'simple': diversity_counter['simple'] / total_records if total_records > 0 else 0,
                'moderate': diversity_counter['moderate'] / total_records if total_records > 0 else 0,
                'complex': diversity_counter['complex'] / total_records if total_records > 0 else 0,
            },
            'global_action_type_counts': dict(global_action_type_counter.most_common()),
        },

        # 其他详细分布
        'reason_distribution': dict(reason_counter),

        'sample_results': action_results,
    }

    # 维度 D (可选)
    if enable_instruction_clarity and n_clarity_success > 0:
        avg_ci = float(np.mean(clarity_intent_scores))
        avg_ca = float(np.mean(clarity_infer_scores))
        avg_ce = float(np.mean(clarity_expr_scores))
        avg_clarity = (avg_ci + avg_ca + avg_ce) / 3.0

        results["instruction_clarity"] = {
            "overall_score": avg_clarity,
            "intent_clarity": avg_ci,
            "action_inferability": avg_ca,
            "expression_quality": avg_ce,
            "n_llm_success": n_clarity_success,
            "n_low_quality": n_clarity_low,
            "low_quality_ratio": n_clarity_low / n_clarity_success if n_clarity_success > 0 else 0.0,
            "score_distributions": {
                "intent_clarity": _summary_stats(clarity_intent_scores),
                "action_inferability": _summary_stats(clarity_infer_scores),
                "expression_quality": _summary_stats(clarity_expr_scores),
            },
            "low_quality_examples": clarity_low_examples,
        }
        results["instruction_clarity_score"] = avg_clarity
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print(f"  总记录数: {total_records:,}")
    print(f"  总动作数: {total_actions:,}")
    print(f"  成功定位: {found_actions:,} ({100*found_actions/total_actions:.1f}%)" if total_actions > 0 else "")
    print()

    # 核心分数
    print("=" * 70)
    print("【核心分数】")
    print(f"  ★ 目标定位难度:     {avg_score:.3f}")
    print(f"  ★ 完成难度:         {avg_completion_score:.3f}")
    print(f"  ★ 动作类型多样性:   {avg_diversity:.3f}")
    if enable_instruction_clarity and n_clarity_success > 0:
        print(f"  ★ 指令清晰度:       {results['instruction_clarity']['overall_score']:.3f}")
    print("=" * 70)
    print()

    # 维度 A: 目标定位难度
    print("【A. 目标定位难度详情】(基于 DOM 深度)")
    print(f"  平均 DOM 深度: {avg_depth:.2f}")
    print()
    print("  分布:")
    print(f"    - 表面 (depth 0-3, 0.5分):  {score_counter['surface']:,} ({100*surface_ratio:.1f}%)")
    print(f"    - 中等 (depth 4-9, 0.75分): {score_counter['moderate']:,} ({100*moderate_ratio:.1f}%)")
    print(f"    - 深层 (depth >9, 1分):     {score_counter['deep']:,} ({100*deep_ratio:.1f}%)")
    print(f"    - 未找到 (0分):             {score_counter['not_found']:,} ({100*not_found_ratio:.1f}%)")
    print()

    # 维度 B: 完成难度
    easy_ratio = steps_easy / total_records if total_records > 0 else 0
    moderate_step_ratio = steps_moderate / total_records if total_records > 0 else 0
    hard_ratio = steps_hard / total_records if total_records > 0 else 0

    print("【B. 完成难度详情】(基于动作步数)")
    print(f"  平均步数: {avg_steps:.1f} | 中位数: {median_steps} | 范围: {min_steps}-{max_steps}")
    print()
    print("  分布:")
    print(f"    - 简单 (1-5 步, 0.5分):   {steps_easy:,} ({100*easy_ratio:.1f}%)")
    print(f"    - 中等 (6-15 步, 0.75分): {steps_moderate:,} ({100*moderate_step_ratio:.1f}%)")
    print(f"    - 困难 (16+ 步, 1分):     {steps_hard:,} ({100*hard_ratio:.1f}%)")
    print()

    # 维度 C: 动作类型多样性
    div_simple_r = diversity_counter['simple'] / total_records if total_records > 0 else 0
    div_mod_r = diversity_counter['moderate'] / total_records if total_records > 0 else 0
    div_complex_r = diversity_counter['complex'] / total_records if total_records > 0 else 0

    print("【C. 动作类型多样性详情】")
    print(f"  平均操作种类: {avg_n_types:.2f} | 平均 entropy: {avg_entropy:.3f}")
    print(f"  全局动作类型分布: {dict(global_action_type_counter.most_common())}")
    print()
    print("  分布:")
    print(f"    - simple (1-3 种, 0.2分):   {diversity_counter['simple']:,} ({100*div_simple_r:.1f}%)")
    print(f"    - moderate (3-6 种, 0.5分): {diversity_counter['moderate']:,} ({100*div_mod_r:.1f}%)")
    print(f"    - complex (>6 种, 1分):     {diversity_counter['complex']:,} ({100*div_complex_r:.1f}%)")
    print()

    # 维度 D (可选)
    if enable_instruction_clarity and n_clarity_success > 0:
        ic = results["instruction_clarity"]
        low_r = ic["low_quality_ratio"]
        print("【D. 指令清晰度详情】(LLM Judge)")
        print(f"  LLM 评估成功: {n_clarity_success}/{total_records}")
        print(f"  意图明确性: {ic['intent_clarity']:.3f} | "
              f"操作可推断性: {ic['action_inferability']:.3f} | "
              f"表述质量: {ic['expression_quality']:.3f}")
        print(f"  低质量样本: {n_clarity_low} ({100*low_r:.1f}%)")
        if clarity_low_examples:
            print("  低质量示例:")
            for i, ex in enumerate(clarity_low_examples, 1):
                print(f"    {i}. {ex['instruction']}")
                print(f"       原因: {ex['reason']}")
        print()

    # 综合解读
    print("【综合解读】")
    dims = [("目标定位", avg_score), ("完成难度", avg_completion_score), ("动作多样性", avg_diversity)]
    if enable_instruction_clarity and n_clarity_success > 0:
        dims.append(("指令清晰度", results.get("instruction_clarity", {}).get("overall_score", 0.0)))
    for dim_name, dim_score in dims:
        if dim_score < 0.5:
            print(f"  {dim_name}: 偏低")
        elif dim_score > 0.8:
            print(f"  {dim_name}: 较高")
        else:
            print(f"  {dim_name}: 适中")
    print("=" * 70)
    
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="任务复杂度评估（基于 Target Depth）")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mind2web", "webshop", "weblinx"],
                        help="数据集名称")
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--use-raw-html", action="store_true",
                        help="使用 raw_html 而非 cleaned_html")
    parser.add_argument("--enable-instruction-clarity", action="store_true",
                        help="开启 LLM Judge 评估指令清晰度 (维度 D)")

    args = parser.parse_args()
    
    # 加载数据和 Locator
    if args.dataset == "mind2web":
        from loaders import Mind2WebLoader
        from executor.mind2web import Mind2WebLocator
        
        data_path = args.data_path or "/home/liuhaoze/Desktop/mind2web/train_0.json"
        loader = Mind2WebLoader(data_path)
        locator = Mind2WebLocator()
        dataset_name = "Mind2Web"
        
    elif args.dataset == "weblinx":
        from loaders import WebLINXLoader
        from executor.weblinx import WebLINXLocator
        
        data_path = args.data_path or "/home/liuhaoze/Desktop/mind2web/weblinx"
        loader = WebLINXLoader(data_path, 'train')
        locator = WebLINXLocator()
        dataset_name = "WebLINX"
        
    else:
        print(f"数据集 {args.dataset} 暂不支持")
        sys.exit(1)
    
    # 输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/task_complexity_results.json"
    
    # 运行评估
    results = compute_task_complexity(
        data_iterator=loader.iterate(),
        locator=locator,
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
        use_cleaned_html=not args.use_raw_html,
        enable_instruction_clarity=args.enable_instruction_clarity,
    )
