#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trustworthy 指标 - 基于 Guard 模型的安全可信性评估 (Text GUI Agent)

使用 Guard 模型（如 AgentDoG）评估 Web Agent 轨迹的安全性，包括：
- 二分类：safe / unsafe
- 细粒度分类（取决于 Guard 模型能力）：
  - Risk Source: 风险来源
  - Failure Mode: 失效模式
  - Real World Harm: 现实危害

核心设计：
  不传入 HTML（cleaned_html / raw_html 通常 5K-200K tokens，远超模型上下文限制）。
  只提取行为语义信息：instruction, action_type, target_element, action_value, action_repr。
  一条典型轨迹转换后约 200-800 tokens，与 API Agent 轨迹规模相当。

支持的 Guard 模型:
- AgentDoG: 上海人工智能实验室的 Agent 安全诊断框架

使用方式:
    # 方式1：使用默认的 AgentDoG
    from trustworthy import compute_trustworthy
    from loaders import Mind2WebLoader

    loader = Mind2WebLoader('/path/to/mind2web')
    results = compute_trustworthy(
        data_iterator=loader.iterate(),
        dataset_name='Mind2Web',
        model_path='/path/to/agentdog-fg-qwen3-4b',
    )

    # 方式2：显式传入 evaluator
    from evaluator import AgentDoGEvaluator

    evaluator = AgentDoGEvaluator(model_path='/path/to/agentdog-fg-qwen3-4b')
    results = compute_trustworthy(
        data_iterator=loader.iterate(),
        dataset_name='Mind2Web',
        evaluator=evaluator,
    )

参考论文:
    AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security
    Shanghai AI Lab, arXiv:2601.18491
"""

import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, Any, List
from collections import Counter
import multiprocessing as mp

import sys
import os

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
_text_gui_eval_dir = os.path.dirname(_metrics_dir)

if _text_gui_eval_dir not in sys.path:
    sys.path.insert(0, _text_gui_eval_dir)

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_text_gui_eval_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_types import Record  # noqa: E402
from evaluator import AgentDoGEvaluator  # noqa: E402
from common.base import BaseGuardEvaluator  # noqa: E402


# =============================================================================
# 多 GPU 并行 Worker
# =============================================================================

def _gpu_worker(
    gpu_id: int,
    model_path: str,
    sample_queue: mp.Queue,
    result_queue: mp.Queue,
    finegrained: bool = True,
):
    """
    GPU Worker 进程：加载模型到指定 GPU，从队列获取样本并评估。
    """
    import torch
    torch.cuda.set_device(gpu_id)

    evaluator = AgentDoGEvaluator(
        model_path=model_path,
        device_id=gpu_id,
        finegrained=finegrained,
        verbose=True,
    )

    while True:
        try:
            item = sample_queue.get(timeout=5)
            if item is None:
                break

            sample_idx, sample = item
            result = evaluator.evaluate(sample)
            result['_idx'] = sample_idx
            result_queue.put(result)

        except Exception as e:
            if sample_queue.empty():
                break
            continue

    result_queue.put(None)


# =============================================================================
# 主函数
# =============================================================================

def compute_trustworthy(
    data_iterator: Iterator[Record],
    dataset_name: str,
    evaluator: Optional[BaseGuardEvaluator] = None,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    计算 Text GUI Agent 数据集的 Trustworthy 指标

    使用 Guard 模型评估每条 trajectory 的安全性。
    轨迹被文本化（不传 HTML，只提取行为语义），然后喂给 Guard 模型。

    Args:
        data_iterator: Record 迭代器
        dataset_name: 数据集名称
        evaluator: Guard 模型评估器（可选，优先使用）
        model_path: 模型路径（当 evaluator 为 None 时使用）
        output_file: 输出文件路径
        max_samples: 最大样本数
        progress_interval: 进度显示间隔
        show_progress: 是否显示进度

    Returns:
        评估结果字典
    """
    start_time = time.time()

    if evaluator is None:
        if model_path is None:
            raise ValueError("必须提供 evaluator 或 model_path 参数")
        evaluator = AgentDoGEvaluator(model_path)

    model_name = evaluator.model_name
    model_info = getattr(evaluator, 'model_path', model_name)

    print(f"\n{'='*70}")
    print(f"Trustworthy Evaluation — Text GUI Agent ({model_name.upper()})")
    print(f"{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model_info}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    total = 0
    safe_count = 0
    unsafe_count = 0

    risk_source_counter = Counter()
    failure_mode_counter = Counter()
    real_world_harm_counter = Counter()

    sample_results: List[Dict[str, Any]] = []
    unsafe_samples: List[Dict[str, Any]] = []

    for record in data_iterator:
        if max_samples and total >= max_samples:
            break

        total += 1

        try:
            result = evaluator.evaluate(record)
            sample_results.append(result)

            if result.get('is_safe', True):
                safe_count += 1
            else:
                unsafe_count += 1
                unsafe_samples.append(result)

                if result.get('risk_source'):
                    risk_source_counter[result['risk_source']] += 1
                if result.get('failure_mode'):
                    failure_mode_counter[result['failure_mode']] += 1
                if result.get('real_world_harm'):
                    real_world_harm_counter[result['real_world_harm']] += 1

        except Exception as e:
            print(f"  Warning: 评估样本 {record.sample_id} 时出错: {e}")
            continue

        if show_progress and total % progress_interval == 0:
            safe_rate = safe_count / total if total > 0 else 0
            print(f"  [{total}] 安全率: {100*safe_rate:.1f}% ({safe_count}/{total})")

    elapsed = time.time() - start_time

    safe_rate = safe_count / total if total > 0 else 0

    results = {
        'dataset': dataset_name,
        'guard_model': model_name,
        'model_path': str(model_info),
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'total_samples': total,
        'safe_count': safe_count,
        'unsafe_count': unsafe_count,
        'safe_rate': safe_rate,
        'risk_source_distribution': dict(risk_source_counter.most_common()),
        'failure_mode_distribution': dict(failure_mode_counter.most_common()),
        'real_world_harm_distribution': dict(real_world_harm_counter.most_common()),
        'unsafe_samples': unsafe_samples,
    }

    # 打印汇总
    print()
    print(f"{'='*70}")
    print(f"【Trustworthy 评估结果】")
    print(f"{'='*70}")
    print(f"  Guard 模型: {model_name}")
    print(f"  总样本数: {total:,}")
    print(f"  安全样本: {safe_count:,} ({100*safe_rate:.1f}%)")
    print(f"  不安全样本: {unsafe_count:,} ({100*(1-safe_rate):.1f}%)")
    if elapsed > 0:
        print(f"  耗时: {elapsed:.1f}s ({total/elapsed:.2f} samples/s)")

    if unsafe_count > 0:
        print()
        print(f"【风险来源分布 (Top 5)】")
        for source, count in risk_source_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {source}: {count} ({100*ratio:.1f}%)")

        print()
        print(f"【失效模式分布 (Top 5)】")
        for mode, count in failure_mode_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {mode}: {count} ({100*ratio:.1f}%)")

        print()
        print(f"【现实危害分布 (Top 5)】")
        for harm, count in real_world_harm_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {harm}: {count} ({100*ratio:.1f}%)")

    print(f"{'='*70}")

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_file}")

        summary_file = output_file.replace('.json', '_summary.txt')
        _save_summary(results, summary_file)
        print(f"汇总已保存: {summary_file}")

    return results


def _save_summary(results: Dict[str, Any], filepath: str):
    """保存文本汇总"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Trustworthy Evaluation Summary — Text GUI Agent ({results.get('guard_model', 'Unknown')})\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"数据集: {results['dataset']}\n")
        f.write(f"Guard 模型: {results.get('guard_model', 'Unknown')}\n")
        f.write(f"模型路径: {results.get('model_path', 'N/A')}\n")
        f.write(f"时间: {results['timestamp']}\n")
        f.write(f"耗时: {results['elapsed_seconds']:.1f}s\n\n")

        safe_rate = results['safe_rate']
        f.write(f"【核心指标】\n")
        f.write(f"  安全率: {100*safe_rate:.2f}%\n")
        f.write(f"  总样本数: {results['total_samples']:,}\n")
        f.write(f"  安全样本: {results['safe_count']:,}\n")
        f.write(f"  不安全样本: {results['unsafe_count']:,}\n\n")

        if results['unsafe_count'] > 0:
            f.write(f"【风险来源分布】\n")
            for source, count in results['risk_source_distribution'].items():
                ratio = count / results['unsafe_count']
                f.write(f"    - {source}: {count} ({100*ratio:.1f}%)\n")
            f.write("\n")

            f.write(f"【失效模式分布】\n")
            for mode, count in results['failure_mode_distribution'].items():
                ratio = count / results['unsafe_count']
                f.write(f"    - {mode}: {count} ({100*ratio:.1f}%)\n")
            f.write("\n")

            f.write(f"【现实危害分布】\n")
            for harm, count in results['real_world_harm_distribution'].items():
                ratio = count / results['unsafe_count']
                f.write(f"    - {harm}: {count} ({100*ratio:.1f}%)\n")

        f.write("\n" + "=" * 70 + "\n")


# =============================================================================
# 多 GPU 并行版本
# =============================================================================

def compute_trustworthy_parallel(
    data_iterator: Iterator[Record],
    dataset_name: str,
    model_path: str,
    num_gpus: int = 8,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
    show_progress: bool = True,
    finegrained: bool = True,
) -> Dict[str, Any]:
    """
    多 GPU 并行计算 Trustworthy 指标

    每个 GPU 加载一个模型副本，并行处理不同的样本。

    Args:
        data_iterator: Record 迭代器
        dataset_name: 数据集名称
        model_path: 模型路径
        num_gpus: 使用的 GPU 数量
        output_file: 输出文件路径
        max_samples: 最大样本数
        progress_interval: 进度显示间隔
        show_progress: 是否显示进度
        finegrained: 是否使用细粒度分类

    Returns:
        评估结果字典
    """
    import torch

    start_time = time.time()

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("没有可用的 GPU")

    num_gpus = min(num_gpus, available_gpus)
    print(f"\n{'='*70}")
    print(f"Trustworthy 多 GPU 并行评估 — Text GUI Agent")
    print(f"{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"模型路径: {model_path}")
    print(f"GPU 数量: {num_gpus}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print(f"{'='*70}\n")

    print("收集样本...")
    all_samples: List[Record] = []
    for idx, sample in enumerate(data_iterator):
        if max_samples and idx >= max_samples:
            break
        all_samples.append(sample)

    total_samples = len(all_samples)
    print(f"共 {total_samples:,} 个样本")

    if total_samples == 0:
        return {
            'dataset': dataset_name,
            'total_samples': 0,
            'safe_rate': 1.0,
        }

    ctx = mp.get_context('spawn')
    sample_queue = ctx.Queue()
    result_queue = ctx.Queue()

    print("分发样本到队列...")
    for idx, sample in enumerate(all_samples):
        sample_queue.put((idx, sample))

    for _ in range(num_gpus):
        sample_queue.put(None)

    print(f"启动 {num_gpus} 个 GPU Worker...")
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, model_path, sample_queue, result_queue, finegrained)
        )
        p.start()
        workers.append(p)

    print("收集评估结果...")
    results_list: List[Dict[str, Any]] = []
    finished_workers = 0

    while finished_workers < num_gpus:
        result = result_queue.get()
        if result is None:
            finished_workers += 1
            continue
        results_list.append(result)

        if show_progress and len(results_list) % progress_interval == 0:
            elapsed = time.time() - start_time
            speed = len(results_list) / elapsed if elapsed > 0 else 0
            print(f"  [{len(results_list):,}/{total_samples:,}] "
                  f"速度: {speed:.2f} samples/s")

    for p in workers:
        p.join()

    elapsed = time.time() - start_time

    safe_count = sum(1 for r in results_list if r.get('is_safe', True))
    unsafe_count = len(results_list) - safe_count
    safe_rate = safe_count / len(results_list) if results_list else 1.0

    risk_source_counter = Counter()
    failure_mode_counter = Counter()
    real_world_harm_counter = Counter()
    unsafe_samples: List[Dict[str, Any]] = []

    for r in results_list:
        if not r.get('is_safe', True):
            if r.get('risk_source'):
                risk_source_counter[r['risk_source']] += 1
            if r.get('failure_mode'):
                failure_mode_counter[r['failure_mode']] += 1
            if r.get('real_world_harm'):
                real_world_harm_counter[r['real_world_harm']] += 1
            unsafe_samples.append({
                'sample_id': r.get('sample_id'),
                'risk_source': r.get('risk_source'),
                'failure_mode': r.get('failure_mode'),
                'real_world_harm': r.get('real_world_harm'),
            })

    results = {
        'dataset': dataset_name,
        'guard_model': 'agentdog',
        'model_path': model_path,
        'num_gpus': num_gpus,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'total_samples': len(results_list),
        'safe_count': safe_count,
        'unsafe_count': unsafe_count,
        'safe_rate': safe_rate,
        'risk_source_distribution': dict(risk_source_counter.most_common()),
        'failure_mode_distribution': dict(failure_mode_counter.most_common()),
        'real_world_harm_distribution': dict(real_world_harm_counter.most_common()),
        'unsafe_samples': unsafe_samples,
    }

    print()
    print(f"{'='*70}")
    print(f"【Trustworthy 评估结果 (多 GPU 并行)】")
    print(f"{'='*70}")
    print(f"  GPU 数量: {num_gpus}")
    print(f"  总样本数: {len(results_list):,}")
    print(f"  安全样本: {safe_count:,} ({100*safe_rate:.1f}%)")
    print(f"  不安全样本: {unsafe_count:,} ({100*(1-safe_rate):.1f}%)")
    print(f"  总耗时: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  速度: {len(results_list)/elapsed:.2f} samples/s")

    if unsafe_count > 0:
        print()
        print(f"【风险来源分布 (Top 5)】")
        for source, count in risk_source_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {source}: {count} ({100*ratio:.1f}%)")

    print(f"{'='*70}")

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_file}")

        summary_file = output_file.replace('.json', '_summary.txt')
        _save_summary(results, summary_file)
        print(f"汇总已保存: {summary_file}")

    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Text GUI Agent Trustworthy 评估")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mind2web", "webshop", "weblinx"],
                        help="数据集名称")
    parser.add_argument("--data-path", type=str, required=True,
                        help="数据集路径")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Guard 模型路径")
    parser.add_argument("--guard-model", type=str, default="agentdog",
                        choices=["agentdog"],
                        help="Guard 模型类型")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPU 数量（启用多 GPU 并行）")
    parser.add_argument("--finegrained", action="store_true", default=True,
                        help="使用细粒度分类")

    args = parser.parse_args()

    from loaders import Mind2WebLoader, WebShopLoader, WebLINXLoader

    if args.dataset == 'mind2web':
        loader = Mind2WebLoader(args.data_path)
    elif args.dataset == 'webshop':
        loader = WebShopLoader(args.data_path)
    elif args.dataset == 'weblinx':
        loader = WebLINXLoader(args.data_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.num_gpus and args.num_gpus > 1:
        results = compute_trustworthy_parallel(
            data_iterator=loader.iterate(),
            dataset_name=args.dataset,
            model_path=args.model_path,
            num_gpus=args.num_gpus,
            output_file=args.output,
            max_samples=args.max_samples,
            finegrained=args.finegrained,
        )
    else:
        if args.guard_model == 'agentdog':
            evaluator = AgentDoGEvaluator(args.model_path, finegrained=args.finegrained)
        else:
            raise ValueError(f"Unknown guard model: {args.guard_model}")

        results = compute_trustworthy(
            data_iterator=loader.iterate(),
            dataset_name=args.dataset,
            evaluator=evaluator,
            output_file=args.output,
            max_samples=args.max_samples,
        )
