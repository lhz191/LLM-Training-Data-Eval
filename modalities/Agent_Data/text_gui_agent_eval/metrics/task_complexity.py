#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Complexity 指标 - 任务复杂度评估

通过目标元素在 DOM 树中的深度（Target Depth）来反映任务复杂度：
- 深度浅 = 目标元素在"表面"，任务简单，Agent 不需要深入探索
- 深度深 = 目标元素"藏得较深"，任务复杂，需要 Agent 主动探索

基于 PATHWAYS 论文的发现：
- 现有训练数据大都假设信息充足、信息就放在表面
- 导致 Agent 缺乏探索能力，只会利用表面信息
- 当信息藏得较深或表面有误导性时，Agent 容易犯错

评分标准（基于 Target Depth）：
- 0: 找不到目标元素
- 0.5: 表面可见（DOM 深度 0-3）
- 0.75: 中等深度（DOM 深度 4-9）
- 1: 深层隐藏（DOM 深度 > 9）

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
    )
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any
from collections import Counter

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import Record
from text_gui_executor import HTMLLocator


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
) -> Dict[str, Any]:
    """
    计算信息接地性指标
    
    遍历每个 record 的每个 action，使用 locator 定位目标元素并计算 DOM 深度。
    
    Args:
        data_iterator: Record 迭代器
        locator: HTMLLocator 实例（需实现 locate_with_depth 方法）
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
        use_cleaned_html: 是否使用 cleaned_html（默认 True），否则使用 raw_html
    
    Returns:
        结果字典，包含：
        - total_records: 总记录数
        - total_actions: 总动作数
        - avg_depth: 平均 DOM 深度
        - avg_score: 平均分数
        - depth_distribution: 深度分布
        - score_distribution: 分数分布（surface/moderate/deep/not_found）
    """
    print("=" * 70)
    print("Task Complexity Evaluation (任务复杂度 - 基于 Target Depth)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"HTML 类型: {'cleaned_html' if use_cleaned_html else 'raw_html'}")
    if max_samples:
        print(f"样本限制: {max_samples}")
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
    
    action_results = []  # 每个 action 的详细结果
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
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
            
            # 记录详细结果（可选，用于调试）
            if len(action_results) < 1000:  # 只保存前 1000 个
                action_results.append({
                    'sample_id': record.sample_id,
                    'action_type': action.action_type,
                    'success': success,
                    'depth': depth,
                    'score': score,
                    'label': label,
                    'reason': reason,
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
        
        # 核心指标
        'avg_depth': avg_depth,
        'avg_score': avg_score,
        
        # 分数分布
        'surface_ratio': surface_ratio,      # 表面可见
        'moderate_ratio': moderate_ratio,    # 中等深度
        'deep_ratio': deep_ratio,            # 深层隐藏
        'not_found_ratio': not_found_ratio,  # 找不到
        
        # 详细分布
        'score_distribution': dict(score_counter),
        'depth_distribution': dict(sorted(depth_counter.items())),
        'reason_distribution': dict(reason_counter),
        
        # 样本结果（用于调试）
        'sample_results': action_results[:100],  # 只保存前 100 个
    }
    
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
    print(f"  成功定位: {found_actions:,} ({100*found_actions/total_actions:.1f}%)")
    print()
    print(f"  📊 平均 DOM 深度: {avg_depth:.2f}")
    print(f"  📊 平均分数: {avg_score:.3f}")
    print()
    print("  分数分布:")
    print(f"    - 表面 (depth 0-3):  {score_counter['surface']:,} ({100*surface_ratio:.1f}%)")
    print(f"    - 中等 (depth 4-9):  {score_counter['moderate']:,} ({100*moderate_ratio:.1f}%)")
    print(f"    - 深层 (depth > 9):  {score_counter['deep']:,} ({100*deep_ratio:.1f}%)")
    print(f"    - 未找到:            {score_counter['not_found']:,} ({100*not_found_ratio:.1f}%)")
    print()
    
    # 解读
    if avg_score < 0.6:
        print("  ⚠️ 数据偏向'表面信息'，Agent 可能缺乏深度探索能力")
    elif avg_score > 0.8:
        print("  ✅ 数据包含较多'深层信息'，有助于训练探索能力")
    else:
        print("  📈 数据深度分布较均衡")
    
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
    )
