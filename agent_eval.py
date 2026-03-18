#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - Agent Workflow 入口

用户传入数据集路径，自动完成：模态检测 -> 管道生成 -> Layer 1 审计

Usage:
    python agent_eval.py /path/to/dataset.jsonl
    python agent_eval.py /path/to/dataset.jsonl --model gpt-4.1 --samples 20
"""

import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='LLM Training Data Evaluation - Agent Workflow',
    )
    parser.add_argument('dataset_path', type=str, help='数据集路径')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='LLM 模型 (默认 gpt-4.1)')
    parser.add_argument('--samples', type=int, default=20, help='采样条数 (默认 20)')
    parser.add_argument('--round', type=int, default=None, help='只运行指定轮次 (1/2/3)')
    args = parser.parse_args()

    path = Path(args.dataset_path)
    if not path.exists():
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)

    print("=" * 60)
    print("  LLM Training Data Evaluation - Agent Workflow")
    print("=" * 60)

    from agent_workflow.round1_detect import run as round1_run

    # Round 1: 模态检测
    r1_result = round1_run(
        dataset_path=str(path),
        model=args.model,
        num_samples=args.samples,
    )

    if args.round == 1:
        print("\n[完成] 仅运行 Round 1。")
        return

    # Round 2: 生成 data_types + loader (TODO)
    print("\n[TODO] Round 2: 管道生成")

    # Round 3: 完善 metric + 运行审计 (TODO)
    print("\n[TODO] Round 3: 审计运行")


if __name__ == '__main__':
    main()
