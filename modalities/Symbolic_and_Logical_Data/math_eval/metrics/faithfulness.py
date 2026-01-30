#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithfulness 指标 - 使用 LLM-as-Judge 评估推理步骤的忠实性

评估维度：
1. 逻辑连贯性：推理步骤之间是否有逻辑联系
2. 步骤有效性：每一步是否是有效的数学变换
3. 无跳跃性：是否存在未解释的跳跃
4. 与问题相关性：推理是否围绕问题展开
"""

import json
import time
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import MathSample


# =============================================================================
# 配置
# =============================================================================

# API 配置
OPENAI_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
OPENAI_BASE_URL = 'http://35.220.164.252:3888/v1/'
OPENAI_MODEL = 'deepseek-v3'


# =============================================================================
# Prompt 模板
# =============================================================================

FAITHFULNESS_PROMPT = """你是一个数学推理评估专家。请评估以下数学问题的解答过程是否忠实、合理。

【数学问题】
{question}

【解答过程】
{solution}

请从以下维度评估解答过程（每个维度 1-5 分）：

1. **逻辑连贯性 (Coherence)**：推理步骤之间是否有清晰的逻辑联系？
   - 5分：每一步都自然地从前一步推导出来
   - 3分：大部分步骤连贯，但有少量跳跃
   - 1分：步骤之间缺乏逻辑联系

2. **步骤有效性 (Validity)**：每一步是否是有效的数学变换？
   - 5分：所有步骤都是正确的数学操作
   - 3分：大部分步骤正确，有少量错误
   - 1分：存在明显的数学错误

3. **完整性 (Completeness)**：推理是否完整，没有未解释的跳跃？
   - 5分：每一步都有充分解释
   - 3分：有些步骤可以更详细
   - 1分：存在重大跳跃，关键步骤缺失

4. **相关性 (Relevance)**：推理是否围绕问题展开，没有无关内容？
   - 5分：所有内容都与解题相关
   - 3分：有少量无关内容
   - 1分：包含大量无关或冗余内容

请以 JSON 格式输出评估结果：
```json
{{
    "coherence": <1-5>,
    "validity": <1-5>,
    "completeness": <1-5>,
    "relevance": <1-5>,
    "overall": <1-5>,
    "explanation": "<简要解释评分理由>"
}}
```

只输出 JSON，不要其他内容。"""


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class FaithfulnessScore:
    """单条数据的忠实性评分"""
    coherence: float        # 逻辑连贯性
    validity: float         # 步骤有效性
    completeness: float     # 完整性
    relevance: float        # 相关性
    overall: float          # 整体评分
    explanation: str        # 评分理由
    
    @property
    def average(self) -> float:
        """四个维度的平均分"""
        return (self.coherence + self.validity + self.completeness + self.relevance) / 4


# =============================================================================
# LLM 调用
# =============================================================================

class LLMJudge:
    """LLM 评判器"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, 
                 base_url: str = OPENAI_BASE_URL,
                 model: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def evaluate_faithfulness(self, question: str, solution: str, 
                              max_retries: int = 3) -> Optional[FaithfulnessScore]:
        """
        评估解答的忠实性
        
        Args:
            question: 数学问题
            solution: 解答过程
            max_retries: 最大重试次数
            
        Returns:
            FaithfulnessScore 或 None（如果评估失败）
        """
        # 处理 solution 是列表的情况
        if isinstance(solution, list):
            solution = "\n\n".join(f"【程序 {i+1}】\n{s}" for i, s in enumerate(solution))
        
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            solution=solution
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                return FaithfulnessScore(
                    coherence=float(result.get('coherence', 3)),
                    validity=float(result.get('validity', 3)),
                    completeness=float(result.get('completeness', 3)),
                    relevance=float(result.get('relevance', 3)),
                    overall=float(result.get('overall', 3)),
                    explanation=result.get('explanation', '')
                )
                
            except json.JSONDecodeError as e:
                print(f"  JSON 解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"  API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None


# =============================================================================
# 主评估函数
# =============================================================================

def compute_faithfulness(
    data_iterator: Iterator[MathSample],
    output_file: Optional[str] = None,
    progress_interval: int = 10,
    dataset_name: str = "Unknown",
    max_samples: Optional[int] = None,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    计算数据集的 Faithfulness 指标
    
    Args:
        data_iterator: 数据迭代器
        output_file: 结果保存路径
        progress_interval: 进度打印间隔
        dataset_name: 数据集名称
        max_samples: 最大评估样本数（None 表示全部）
        api_key: OpenAI API Key
        base_url: API Base URL
        model: 模型名称
        
    Returns:
        包含统计结果的字典
    """
    print("=" * 70)
    print("Faithfulness Evaluation (LLM-as-Judge)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 初始化
    judge = LLMJudge(api_key=api_key, base_url=base_url, model=model)
    start_time = time.time()
    
    # 统计变量
    total = 0
    success = 0
    failed = 0
    
    # 累计分数
    sum_coherence = 0.0
    sum_validity = 0.0
    sum_completeness = 0.0
    sum_relevance = 0.0
    sum_overall = 0.0
    
    # 详细结果
    detailed_results = []
    failed_samples = []
    
    for sample in data_iterator:
        if max_samples and total >= max_samples:
            break
            
        total += 1
        
        # 评估
        score = judge.evaluate_faithfulness(sample.question, sample.solution)
        
        if score:
            success += 1
            sum_coherence += score.coherence
            sum_validity += score.validity
            sum_completeness += score.completeness
            sum_relevance += score.relevance
            sum_overall += score.overall
            
            detailed_results.append({
                'sample_id': sample.sample_id,
                'coherence': score.coherence,
                'validity': score.validity,
                'completeness': score.completeness,
                'relevance': score.relevance,
                'overall': score.overall,
                'explanation': score.explanation,
            })
        else:
            failed += 1
            failed_samples.append({
                'sample_id': sample.sample_id,
                'question': sample.question[:200],
            })
        
        # 进度
        if progress_interval and total % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            avg_overall = sum_overall / success if success > 0 else 0
            print(f"  已处理 {total} 条... ({rate:.1f} 条/秒, 平均分: {avg_overall:.2f})")
    
    elapsed = time.time() - start_time
    
    # 计算平均分
    avg_coherence = sum_coherence / success if success > 0 else 0
    avg_validity = sum_validity / success if success > 0 else 0
    avg_completeness = sum_completeness / success if success > 0 else 0
    avg_relevance = sum_relevance / success if success > 0 else 0
    avg_overall = sum_overall / success if success > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # 统计
        'total': total,
        'success': success,
        'failed': failed,
        
        # 平均分 (1-5)
        'avg_coherence': avg_coherence,
        'avg_validity': avg_validity,
        'avg_completeness': avg_completeness,
        'avg_relevance': avg_relevance,
        'avg_overall': avg_overall,
        
        # 归一化分数 (0-1)
        'norm_coherence': (avg_coherence - 1) / 4,
        'norm_validity': (avg_validity - 1) / 4,
        'norm_completeness': (avg_completeness - 1) / 4,
        'norm_relevance': (avg_relevance - 1) / 4,
        'norm_overall': (avg_overall - 1) / 4,
        
        # 详细结果
        'detailed_results': detailed_results,
        'failed_samples': failed_samples,
    }
    
    # 打印结果
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【统计】")
    print(f"  总样本数:     {total}")
    print(f"  成功评估:     {success}")
    print(f"  评估失败:     {failed}")
    print()
    print(f"【平均分数】(1-5 分)")
    print(f"  逻辑连贯性 (Coherence):    {avg_coherence:.2f}")
    print(f"  步骤有效性 (Validity):     {avg_validity:.2f}")
    print(f"  完整性 (Completeness):     {avg_completeness:.2f}")
    print(f"  相关性 (Relevance):        {avg_relevance:.2f}")
    print(f"  整体评分 (Overall):        {avg_overall:.2f}")
    print()
    print(f"【归一化分数】(0-1)")
    print(f"  Faithfulness Score:        {results['norm_overall']:.4f}")
    print()
    print("=" * 70)
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    
    return results


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 简单测试
    print("测试 LLM Judge...")
    
    judge = LLMJudge()
    
    test_question = "计算 2 + 3 × 4"
    test_solution = """
首先，根据运算优先级，我们需要先计算乘法。
3 × 4 = 12
然后，计算加法。
2 + 12 = 14
因此，答案是 14。
"""
    
    score = judge.evaluate_faithfulness(test_question, test_solution)
    
    if score:
        print(f"评估成功！")
        print(f"  Coherence: {score.coherence}")
        print(f"  Validity: {score.validity}")
        print(f"  Completeness: {score.completeness}")
        print(f"  Relevance: {score.relevance}")
        print(f"  Overall: {score.overall}")
        print(f"  Explanation: {score.explanation}")
    else:
        print("评估失败！")
