#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Validity 指标 - GUI Agent 轨迹有效性检查

使用 LLM Judge 评估：
1. 一致性 (Consistency): 动作序列中的所有动作是否都是出于完成用户指令的目的而执行的
2. 完整性 (Completeness): 给定最后一个动作执行前的页面状态和最后一个动作，
   判断执行该动作后能否完成任务目标

注意：中间过程允许错误尝试，只要轨迹中自己能修正就可以。

使用方式:
    from metrics.trajectory_validity import compute_trajectory_validity
    from loaders import Mind2WebLoader
    
    loader = Mind2WebLoader('/path/to/data')
    
    results = compute_trajectory_validity(
        data_iterator=loader.iterate(),
        dataset_name='Mind2Web',
        output_file='trajectory_validity_results.json',
        max_samples=100,
    )
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import Record

# =============================================================================
# LLM 配置
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4.1'

# =============================================================================
# Prompt 模板
# =============================================================================

TRAJECTORY_VALIDITY_PROMPT = """你是一个 GUI Agent 数据质量评估专家。请评估以下 Agent 执行轨迹的有效性。

【用户指令】
{instruction}

【执行的动作序列】
{action_reprs}

【最后一个动作执行前的页面状态】
{last_page}

【最后一个动作】
{last_action}

请评估：
1. 一致性 (Consistency): 动作序列中的所有动作是否都是出于完成用户指令的目的而执行的？
   - 判断这些动作是否都是为了实现用户指令而做的合理操作。
   - 注意：中间过程允许错误尝试和探索，只要最终是朝着目标前进，且轨迹中能自己修正即可。
   - 例如：用户要"搜索并购买红色笔记本电脑"，动作序列包含搜索、筛选、查看商品、添加购物车，
     这些都是为了完成购买目标的合理操作，应判定为一致。

2. 完整性 (Completeness): 在当前页面状态下执行最后一个动作后，任务能否完成？
   - 你需要根据【最后一个动作执行前的页面状态】和【最后一个动作】，
     判断执行该动作后是否能实现用户指令的目标。
   - 例如：如果用户要"添加商品到购物车"，页面显示商品详情，最后一个动作是
     点击"Add to Cart"按钮，则执行后任务完成。
   - 例如：如果用户要"搜索航班"，页面是搜索表单，最后一个动作是点击"搜索"
     按钮，则执行后任务完成。

评分标准（只能是 0, 0.5, 1 三个值）：
- 1: 完全一致 / 执行最后动作后任务完成
- 0.5: 部分一致 / 执行最后动作后接近完成但可能还需几步
- 0: 不一致 / 执行最后动作后明显无法完成任务

请以 JSON 格式输出：
```json
{{
    "consistency": {{
        "score": 0/0.5/1,
        "reason": "简要说明一致性判断理由"
    }},
    "completeness": {{
        "score": 0/0.5/1,
        "reason": "简要说明完整性判断理由（说明执行最后动作后的预期结果）"
    }}
}}
```

只输出 JSON，不要其他内容。"""

# =============================================================================
# LLM 调用
# =============================================================================

def _call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    调用 LLM API
    
    Args:
        prompt: 提示词
        max_retries: 最大重试次数
        
    Returns:
        LLM 响应文本，失败返回 None
    """
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
                time.sleep(2 ** attempt)  # 指数退避
    
    return None


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """
    解析 LLM 响应
    
    Args:
        response: LLM 响应文本
        
    Returns:
        解析后的字典
    """
    if not response:
        return {
            'consistency': {'score': 0.0, 'reason': 'LLM 调用失败'},
            'completeness': {'score': 0.0, 'reason': 'LLM 调用失败'},
            'parse_error': True,
        }
    
    try:
        # 提取 JSON 部分
        response = response.strip()
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        result = json.loads(response.strip())
        return result
    except Exception as e:
        return {
            'consistency': {'score': 0.0, 'reason': f'解析失败: {e}'},
            'completeness': {'score': 0.0, 'reason': f'解析失败: {e}'},
            'parse_error': True,
            'raw_response': response,
        }


# =============================================================================
# 主函数
# =============================================================================

def compute_trajectory_validity(
    data_iterator: Iterator[Record],
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 10,
) -> Dict[str, Any]:
    """
    计算轨迹有效性指标
    
    Args:
        data_iterator: Record 迭代器
        dataset_name: 数据集名称
        output_file: 结果输出文件
        max_samples: 最大样本数（用于测试）
        progress_interval: 进度显示间隔
    
    Returns:
        结果字典，包含：
        - total_records: 总记录数
        - avg_consistency: 平均一致性分数
        - avg_completeness: 平均完整性分数
        - record_results: 每条记录的详细结果
    """
    print("=" * 70)
    print("Trajectory Validity Evaluation (LLM Judge)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()
    
    start_time = time.time()
    
    # 统计
    total_records = 0
    total_consistency = 0.0
    total_completeness = 0.0
    llm_failures = 0
    
    record_results = []
    
    for record in data_iterator:
        if max_samples and total_records >= max_samples:
            break
        
        total_records += 1
        
        # 准备 LLM 输入
        instruction = record.instruction or "(无指令)"
        
        # 动作序列
        action_reprs = []
        for i, action in enumerate(record.actions):
            repr_str = action.action_repr or f"{action.action_type}"
            action_reprs.append(f"Step {i+1}: {repr_str}")
        action_reprs_str = "\n".join(action_reprs)
        
        # 最后一个动作和页面
        if record.actions:
            last_action = record.actions[-1]
            last_action_str = last_action.action_repr or f"{last_action.action_type}"
            last_page = last_action.cleaned_html or "(无页面内容)"
        else:
            last_action_str = "(无动作)"
            last_page = "(无页面)"
        
        # 构建 prompt
        prompt = TRAJECTORY_VALIDITY_PROMPT.format(
            instruction=instruction,
            action_reprs=action_reprs_str,
            last_page=last_page,
            last_action=last_action_str,
        )
        
        # 调用 LLM
        response = _call_llm(prompt)
        result = _parse_llm_response(response)
        
        # 提取分数
        consistency_score = result.get('consistency', {}).get('score', 0.0)
        completeness_score = result.get('completeness', {}).get('score', 0.0)
        
        if result.get('parse_error'):
            llm_failures += 1
        
        total_consistency += consistency_score
        total_completeness += completeness_score
        
        # 记录结果
        record_result = {
            'sample_id': record.sample_id,
            'instruction': instruction,
            'n_actions': len(record.actions),
            'last_action': last_action_str,
            'consistency': result.get('consistency', {}),
            'completeness': result.get('completeness', {}),
        }
        
        if result.get('parse_error'):
            record_result['llm_error'] = True
        
        record_results.append(record_result)
        
        # 进度
        if progress_interval and total_records % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_records / elapsed if elapsed > 0 else 0
            avg_cons = total_consistency / total_records
            avg_comp = total_completeness / total_records
            print(f"  [{total_records:,}] {rate:.2f} 条/秒 | 一致性: {avg_cons:.2f} | 完整性: {avg_comp:.2f}")
    
    elapsed = time.time() - start_time
    
    # 计算平均值
    avg_consistency = total_consistency / total_records if total_records > 0 else 0.0
    avg_completeness = total_completeness / total_records if total_records > 0 else 0.0
    
    # 构建结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'total_records': total_records,
        'llm_failures': llm_failures,
        'avg_consistency': avg_consistency,
        'avg_completeness': avg_completeness,
        'record_results': record_results,
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
    print(f"  LLM 调用失败: {llm_failures}")
    print()
    print(f"  📊 平均一致性分数: {avg_consistency:.3f}")
    print(f"  📊 平均完整性分数: {avg_completeness:.3f}")
    print()
    
    # 分数分布
    if record_results:
        high_consistency = sum(1 for r in record_results if r.get('consistency', {}).get('score', 0) >= 0.8)
        high_completeness = sum(1 for r in record_results if r.get('completeness', {}).get('score', 0) >= 0.8)
        print(f"  一致性 >= 0.8: {high_consistency}/{total_records} ({100*high_consistency/total_records:.1f}%)")
        print(f"  完整性 >= 0.8: {high_completeness}/{total_records} ({100*high_completeness/total_records:.1f}%)")
    
    print("=" * 70)
    
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹有效性评估 (LLM Judge)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mind2web", "webshop", "weblinx"],
                        help="数据集名称")
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.dataset == "mind2web":
        from loaders import Mind2WebLoader
        data_path = args.data_path or "/home/liuhaoze/Desktop/mind2web/train_0.json"
        loader = Mind2WebLoader(data_path)
        dataset_name = "Mind2Web"
        
    elif args.dataset == "webshop":
        from loaders import WebShopLoader
        data_path = args.data_path or os.path.join(parent_dir, 'webshop/baseline_models/data/il_trajs_finalized_images.jsonl')
        loader = WebShopLoader(data_path)
        dataset_name = "WebShop"
        
    elif args.dataset == "weblinx":
        from loaders import WebLINXLoader
        data_path = args.data_path or "/home/liuhaoze/Downloads/raw_data"
        loader = WebLINXLoader(data_path, 'train')
        dataset_name = "WebLINX"
    
    # 输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/trajectory_validity_results.json"
    
    # 运行评估
    results = compute_trajectory_validity(
        data_iterator=loader.iterate(),
        dataset_name=dataset_name,
        output_file=output_file,
        max_samples=args.max_samples,
    )
