#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Validity 指标 - 验证推理过程的有效性

与 Faithfulness（LLM 打分）不同，这个指标通过 LLM 验证推理过程的有效性。

评估两个维度：
1. 推理过程有效性 (is_valid)：推理过程是否正确、能否解决 query
2. 结果一致性 (result_match)：LLM 按推理过程推导出的结果是否与 GT 一致

最终判定 (overall_valid)：
- 必须同时满足：is_valid=True AND result_match=True
- 即：推理过程正确 + 推导结果与 GT 一致

使用方式:
    from reasoning_validity import compute_reasoning_validity
    from code_executor import OpenMathCodeExtractor, BoxedAnswerExtractor
    from loaders import OpenMathInstructLoader
    
    loader = OpenMathInstructLoader('/path/to/OpenMathInstruct-1')
    
    results = compute_reasoning_validity(
        data_iterator=loader.iterate(),
        code_extractor=OpenMathCodeExtractor(),
        answer_extractor=BoxedAnswerExtractor(),
        output_file='reasoning_validity_results.json'
    )
"""

import re
import json
import time
import asyncio
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI

from data_types import MathSample
from code_executor import CodeExtractor, AnswerExtractor, ResultComparator, compare_math_answers
from openmath_executor import BoxedAnswerExtractor, DirectAnswerExtractor


# =============================================================================
# 配置
# =============================================================================

OPENAI_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
OPENAI_BASE_URL = 'http://35.220.164.252:3888/v1/'
OPENAI_MODEL = 'deepseek-v3'


# =============================================================================
# Prompt 模板
# =============================================================================

# 代码 solution 的 prompt
CODE_REASONING_PROMPT = """你是一个代码分析专家。请分析以下数学问题和对应的 Python 代码解答。

【数学问题】
{question}

【Python 代码】
```python
{code}
```

请判断代码的逻辑是否正确，能否解决上述数学问题。

**特别注意**：如果上面包含多段代码，可能有两种情况：
1. **独立解法**（用"【解答 N】"标记）：多个独立的解法程序，每个都应该能正确解决问题
- 如果所有代码的逻辑都正确且解决同一个问题，则 is_valid=true
- 如果任意一段代码有逻辑错误或会报错，则 is_valid=false
2. **前后依赖**（用"# --- 代码块分隔 ---"标记）：多个代码块是同一个推理过程的不同步骤，后面的代码可能依赖前面定义的变量
   - 需要整体逻辑正确才能判定 is_valid=true

请以 JSON 格式输出：
```json
{{
    "is_valid": true/false,
    "reason": "简要说明代码逻辑是否正确"
}}
```

注意：
- is_valid 表示代码逻辑是否正确，能否解决问题
- 不需要预测执行结果，只需判断逻辑正确性

只输出 JSON，不要其他内容。"""


# 自然语言 solution 的 prompt
NL_REASONING_PROMPT = """你是一个数学推理专家。请根据以下数学问题和推理过程，给出最终答案。

【数学问题】
{question}

【推理过程】
{solution}

请完成以下任务：
1. 判断这个推理过程是否正确，能否解决上述数学问题
2. 根据推理过程，给出最终答案

请以 JSON 格式输出：
```json
{{
    "is_valid": true/false,
    "reason": "简要说明推理过程是否正确",
    "predicted_result": "根据推理过程得出的最终答案（只写数值或表达式）"
}}
```

注意：
- is_valid 表示推理过程是否正确、逻辑是否自洽
- predicted_result 只写最终答案，如 "29"、"3.14"、"x=5" 等

只输出 JSON，不要其他内容。"""


# =============================================================================
# Solution 处理函数
# =============================================================================

def mask_answer(solution: str, answer_extractor: AnswerExtractor) -> str:
    """
    使用 answer_extractor 提取答案，然后在 solution 中 mask 掉
    
    策略：
    1. 用 answer_extractor 提取答案
    2. 在 solution 中找到该答案的位置，替换为 [MASKED]
    
    这样 mask 逻辑与数据集格式无关，由 answer_extractor 负责定位答案
    """
    extracted_answer = answer_extractor.extract(solution)
    
    if not extracted_answer:
        # 没有提取到答案，返回原文
        return solution
    
    # 在 solution 中找到答案并替换（从后往前找，替换最后一个出现的位置）
    idx = solution.rfind(extracted_answer)
    if idx >= 0:
        return solution[:idx] + '[MASKED]' + solution[idx + len(extracted_answer):]
    
    # 如果找不到精确匹配，返回原文
    return solution


# =============================================================================
# LLM 调用
# =============================================================================

def call_llm(
    prompt: str,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
    max_retries: int = 3,
    sample_id: str = None
) -> Optional[Dict]:
    """
    调用 LLM 并解析 JSON 响应
    
    Args:
        sample_id: 样本 ID（用于错误日志）
    
    Returns:
        解析后的 dict，或 None（如果失败）
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
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
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            id_info = f" [{sample_id}]" if sample_id else ""
            print(f"  JSON 解析失败{id_info} (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            id_info = f" [{sample_id}]" if sample_id else ""
            print(f"  API 调用失败{id_info} (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None


# =============================================================================
# 主评估函数
# =============================================================================

def compute_reasoning_validity(
    data_iterator: Iterator[MathSample],
    code_extractor: CodeExtractor,
    answer_extractor: AnswerExtractor,
    output_file: Optional[str] = None,
    progress_interval: int = 10,
    dataset_name: str = "Unknown",
    max_samples: Optional[int] = None,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    计算数据集的 Reasoning Validity 指标
    
    Args:
        data_iterator: 数据迭代器
        code_extractor: 代码提取器（用于判断是否有代码、提取代码）
        answer_extractor: 答案提取器（用于从 solution 中提取答案进行 mask）
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
    print("Reasoning Validity Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # 统计变量
    total = 0
    success = 0
    failed = 0
    
    # 分类统计
    code_total = 0
    code_valid = 0           # LLM 认为代码逻辑正确
    code_overall_valid = 0   # 代码 solution：逻辑正确即为有效
    
    nl_total = 0
    nl_valid = 0             # LLM 认为推理正确
    nl_match = 0             # 预测结果与 GT 匹配
    nl_overall_valid = 0     # NL solution：逻辑正确 + 结果匹配
    
    # 详细结果
    detailed_results = []
    failed_samples = []
    invalid_samples = []
    
    # 当前批次的 invalid 样本（用于进度输出）
    batch_invalid_ids = []
    
    for sample in data_iterator:
        if max_samples and total >= max_samples:
            break
        
        total += 1
        
        # 处理 solution（可能是列表）
        solution = sample.solution
        if isinstance(solution, list):
            solution = "\n\n".join(f"【解答 {i+1}】\n{s}" for i, s in enumerate(solution))
        
        # 使用 code_extractor 提取所有代码（extract_all_code 会提取所有代码块）
        if hasattr(code_extractor, 'extract_all_code'):
            code = code_extractor.extract_all_code(solution)
        else:
            code = code_extractor.extract(solution)
        
        if code:
            # 代码 solution
            solution_type = "code"
            prompt = CODE_REASONING_PROMPT.format(question=sample.question, code=code)
        else:
            # NL solution：用 answer_extractor mask 答案
            solution_type = "nl"
            masked_solution = mask_answer(solution, answer_extractor)
            prompt = NL_REASONING_PROMPT.format(question=sample.question, solution=masked_solution)
        
        # 调用 LLM
        llm_result = call_llm(prompt, api_key, base_url, model, sample_id=sample.sample_id)
        
        if llm_result:
            success += 1
            
            is_valid = llm_result.get('is_valid', False)
            reason = llm_result.get('reason', '')
            
            # 分类统计和结果处理
            if solution_type == "code":
                # 代码 solution：只判断逻辑正确性，不需要预测结果
                code_total += 1
                if is_valid:
                    code_valid += 1
                    code_overall_valid += 1
                
                predicted_result = None
                result_match = None  # 代码 solution 不评估结果匹配
                overall_valid = is_valid
            else:
                # NL solution：需要预测结果并与 GT 比较
                predicted_result = str(llm_result.get('predicted_result', '')).strip()
                
                result_match = False
                if predicted_result and predicted_result not in ('ERROR', 'INVALID', ''):
                    gt = sample.ground_truth
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            result_match = compare_math_answers(predicted_result, gt[0])
                        else:
                            gt_str = str(gt)
                            result_match = compare_math_answers(predicted_result, gt_str)
                    else:
                        result_match = compare_math_answers(predicted_result, gt)
                
                overall_valid = is_valid and result_match
                
                nl_total += 1
                if is_valid:
                    nl_valid += 1
                if result_match:
                    nl_match += 1
                if overall_valid:
                    nl_overall_valid += 1
            
            # 记录详细结果
            detailed_results.append({
                'sample_id': sample.sample_id,
                'solution_type': solution_type,
                'is_valid': is_valid,
                'reason': reason,
                'predicted_result': predicted_result,
                'ground_truth': sample.ground_truth,
                'result_match': result_match,
                'overall_valid': overall_valid,
            })
            
            # 记录无效样本
            if not overall_valid:
                invalid_samples.append({
                    'sample_id': sample.sample_id,
                    'solution_type': solution_type,
                    'is_valid': is_valid,
                    'result_match': result_match,
                    'reason': reason,
                    'predicted': predicted_result,
                    'ground_truth': sample.ground_truth,
                })
                batch_invalid_ids.append(sample.sample_id)
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
            overall_valid_count = code_overall_valid + nl_overall_valid
            overall_rate = overall_valid_count / success if success > 0 else 0
            print(f"  [{total}/{max_samples or '?'}] {rate:.1f} 条/秒, 有效率: {overall_rate:.2%}")
            # 显示当前批次的 invalid 样本
            if batch_invalid_ids:
                print(f"    无效样本: {batch_invalid_ids}")
                batch_invalid_ids = []  # 清空批次
    
    elapsed = time.time() - start_time
    
    # 计算比率
    def safe_div(a, b):
        return a / b if b > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # 总体统计
        'total': total,
        'success': success,
        'failed': failed,
        
        # 代码 solution 统计（只评估逻辑正确性）
        'code_total': code_total,
        'code_valid_rate': safe_div(code_valid, code_total),
        # code_match_rate 不再适用于代码 solution
        'code_overall_valid_rate': safe_div(code_overall_valid, code_total),
        
        # NL solution 统计
        'nl_total': nl_total,
        'nl_valid_rate': safe_div(nl_valid, nl_total),
        'nl_match_rate': safe_div(nl_match, nl_total),
        'nl_overall_valid_rate': safe_div(nl_overall_valid, nl_total),
        
        # 总体有效率
        'overall_valid_rate': safe_div(code_overall_valid + nl_overall_valid, success),
        
        # 详细结果
        'detailed_results': detailed_results,
        'failed_samples': failed_samples,
        'invalid_samples': invalid_samples,
    }
    
    # 打印结果
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【总体统计】")
    print(f"  总样本数:       {total}")
    print(f"  成功评估:       {success}")
    print(f"  评估失败:       {failed}")
    print()
    
    if code_total > 0:
        print(f"【代码 Solution】({code_total} 条)")
        print(f"  逻辑正确率:     {results['code_valid_rate']:.2%}")
        print(f"  最终有效率:     {results['code_overall_valid_rate']:.2%}")
        print()
    
    if nl_total > 0:
        print(f"【NL Solution】({nl_total} 条)")
        print(f"  推理正确率:     {results['nl_valid_rate']:.2%}")
        print(f"  结果匹配率:     {results['nl_match_rate']:.2%}")
        print(f"  最终有效率:     {results['nl_overall_valid_rate']:.2%}")
        print()
    
    print(f"【最终指标】")
    print(f"  Reasoning Validity: {results['overall_valid_rate']:.4f}")
    print()
    print("=" * 70)
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    
    return results


# =============================================================================
# 并行版本
# =============================================================================

def _process_single_sample_for_reasoning(
    sample: MathSample,
    code_extractor: CodeExtractor,
    answer_extractor: AnswerExtractor,
    comparator_class: type,
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    """
    处理单个样本（用于并行）
    
    Args:
        comparator_class: 结果比较器类（用于 code solution 的比较）
    """
    # 处理 solution（可能是列表）
    solution = sample.solution
    if isinstance(solution, list):
        solution = "\n\n".join(f"【解答 {i+1}】\n{s}" for i, s in enumerate(solution))
    
    # 使用 code_extractor 判断是否有代码
    code = code_extractor.extract(solution)
    
    if code:
        solution_type = "code"
        prompt = CODE_REASONING_PROMPT.format(question=sample.question, code=code)
    else:
        solution_type = "nl"
        masked_solution = mask_answer(solution, answer_extractor)
        prompt = NL_REASONING_PROMPT.format(question=sample.question, solution=masked_solution)
    
    # 调用 LLM
    llm_result = call_llm(prompt, api_key, base_url, model, sample_id=sample.sample_id)
    
    if llm_result:
        is_valid = llm_result.get('is_valid', False)
        reason = llm_result.get('reason', '')
        
        if solution_type == "code":
            # 代码 solution：只判断逻辑正确性，不需要预测结果
            # 结果匹配由 Validity 指标负责
            return {
                'success': True,
                'sample_id': sample.sample_id,
                'solution_type': solution_type,
                'is_valid': is_valid,
                'reason': reason,
                'predicted_result': None,
                'ground_truth': sample.ground_truth,
                'result_match': None,  # 代码 solution 不评估结果匹配
                'overall_valid': is_valid,  # 代码 solution 的有效性只看逻辑
            }
        else:
            # NL solution：需要预测结果并与 GT 比较
            predicted_result = str(llm_result.get('predicted_result', '')).strip()
            
            result_match = False
            if predicted_result and predicted_result not in ('ERROR', 'INVALID', ''):
                gt = sample.ground_truth
                if isinstance(gt, list):
                    if len(gt) == 1:
                        result_match = compare_math_answers(predicted_result, gt[0])
                    else:
                        gt_str = str(gt)
                        result_match = compare_math_answers(predicted_result, gt_str)
                else:
                    result_match = compare_math_answers(predicted_result, gt)
            
            overall_valid = is_valid and result_match
            
            return {
                'success': True,
                'sample_id': sample.sample_id,
                'solution_type': solution_type,
                'is_valid': is_valid,
                'reason': reason,
                'predicted_result': predicted_result,
                'ground_truth': sample.ground_truth,
                'result_match': result_match,
                'overall_valid': overall_valid,
            }
    else:
        return {
            'success': False,
            'sample_id': sample.sample_id,
            'question': sample.question[:200],
        }


def compute_reasoning_validity_parallel(
    data_iterator: Iterator[MathSample],
    code_extractor: CodeExtractor,
    answer_extractor: AnswerExtractor,
    comparator_class: type = ResultComparator,
    output_file: Optional[str] = None,
    progress_interval: int = 100,
    dataset_name: str = "Unknown",
    max_samples: Optional[int] = None,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
    max_workers: int = 32,
) -> Dict[str, Any]:
    """
    并行版本的 Reasoning Validity 评估
    
    使用 ProcessPoolExecutor 并发调用 LLM API
    
    Args:
        comparator_class: 结果比较器类（用于 code solution 比较，默认 ResultComparator）
        max_workers: 并行进程数（默认 32）
        其他参数同 compute_reasoning_validity
    """
    print("=" * 70)
    print("Reasoning Validity Evaluation (并行模式)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model}")
    print(f"并行线程: {max_workers}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # 收集所有样本
    samples = []
    for sample in data_iterator:
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(sample)
    
    total = len(samples)
    print(f"待处理样本数: {total}")
    print()
    
    # 统计变量
    success = 0
    failed = 0
    code_total = 0
    code_valid = 0
    code_overall_valid = 0
    nl_total = 0
    nl_valid = 0
    nl_match = 0
    nl_overall_valid = 0
    
    detailed_results = []
    failed_samples = []
    invalid_samples = []
    
    # 当前批次的 invalid 样本（用于进度输出）
    batch_invalid_ids = []
    
    # 并行处理（使用进程池，避免线程不安全问题）
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_sample_for_reasoning,
                sample, code_extractor, answer_extractor, comparator_class,
                api_key, base_url, model
            ): sample.sample_id
            for sample in samples
        }
        
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            if result['success']:
                success += 1
                solution_type = result['solution_type']
                is_valid = result['is_valid']
                result_match = result['result_match']
                overall_valid = result['overall_valid']
                
                if solution_type == "code":
                    # 代码 solution：只统计逻辑正确性
                    code_total += 1
                    if is_valid:
                        code_valid += 1
                    if overall_valid:
                        code_overall_valid += 1
                else:
                    # NL solution：统计逻辑正确性和结果匹配
                    nl_total += 1
                    if is_valid:
                        nl_valid += 1
                    if result_match:
                        nl_match += 1
                    if overall_valid:
                        nl_overall_valid += 1
                
                detailed_results.append(result)
                
                if not overall_valid:
                    invalid_samples.append(result)
                    batch_invalid_ids.append(result.get('sample_id', 'unknown'))
            else:
                failed += 1
                failed_samples.append(result)
            
            # 进度
            if progress_interval and completed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                overall_valid_count = code_overall_valid + nl_overall_valid
                overall_rate = overall_valid_count / success if success > 0 else 0
                print(f"  [{completed}/{total}] {rate:.1f} 条/秒, 有效率: {overall_rate:.2%}")
                # 显示当前批次的 invalid 样本（排序后输出）
                if batch_invalid_ids:
                    # 按 sample_id 中的数字排序
                    try:
                        sorted_ids = sorted(batch_invalid_ids, key=lambda x: int(x.split('_')[-1]))
                    except:
                        sorted_ids = sorted(batch_invalid_ids)
                    print(f"    无效样本: {sorted_ids}")
                    batch_invalid_ids = []  # 清空批次
    
    elapsed = time.time() - start_time
    
    # 计算比率
    def safe_div(a, b):
        return a / b if b > 0 else 0
    
    # 结果
    results = {
        'dataset': dataset_name,
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'max_workers': max_workers,
        
        'total': total,
        'success': success,
        'failed': failed,
        
        # 代码 solution 统计（只评估逻辑正确性）
        'code_total': code_total,
        'code_valid_rate': safe_div(code_valid, code_total),
        'code_overall_valid_rate': safe_div(code_overall_valid, code_total),
        
        # NL solution 统计
        'nl_total': nl_total,
        'nl_valid_rate': safe_div(nl_valid, nl_total),
        'nl_match_rate': safe_div(nl_match, nl_total),
        'nl_overall_valid_rate': safe_div(nl_overall_valid, nl_total),
        
        'overall_valid_rate': safe_div(code_overall_valid + nl_overall_valid, success),
        
        'detailed_results': detailed_results,
        'failed_samples': failed_samples,
        'invalid_samples': invalid_samples,
    }
    
    # 打印结果
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒 ({total/elapsed:.1f} 条/秒)")
    print("=" * 70)
    print()
    print(f"【总体统计】")
    print(f"  总样本数:       {total}")
    print(f"  成功评估:       {success}")
    print(f"  评估失败:       {failed}")
    print()
    
    if code_total > 0:
        print(f"【代码 Solution】({code_total} 条)")
        print(f"  逻辑正确率:     {results['code_valid_rate']:.2%}")
        print(f"  最终有效率:     {results['code_overall_valid_rate']:.2%}")
        print()
    
    if nl_total > 0:
        print(f"【NL Solution】({nl_total} 条)")
        print(f"  推理正确率:     {results['nl_valid_rate']:.2%}")
        print(f"  结果匹配率:     {results['nl_match_rate']:.2%}")
        print(f"  最终有效率:     {results['nl_overall_valid_rate']:.2%}")
        print()
    
    print(f"【最终指标】")
    print(f"  Reasoning Validity: {results['overall_valid_rate']:.4f}")
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
    pass  # BoxedAnswerExtractor and DirectAnswerExtractor already imported at module level
    
    # 测试 mask 功能
    print("=" * 70)
    print("测试 mask_answer 函数（使用 BoxedAnswerExtractor）")
    print("=" * 70)
    print()
    
    boxed_extractor = BoxedAnswerExtractor()
    
    # 测试用例：(输入, 提取的答案, 期望输出)
    test_cases = [
        # 简单 boxed
        (
            "答案是 \\boxed{29}。",
            "29",
            "答案是 \\boxed{[MASKED]}。"
        ),
        # LaTeX 分数
        (
            "答案是 \\boxed{\\frac{1}{2}}",
            "\\frac{1}{2}",
            "答案是 \\boxed{[MASKED]}"
        ),
        # 完整的 solution 示例
        (
            "The answer is $2009$ divided by the number of sides (= $100$) to get $\\boxed{20.09}$.",
            "20.09",
            "The answer is $2009$ divided by the number of sides (= $100$) to get $\\boxed{[MASKED]}$."
        ),
        # 嵌套 boxed（提取最内层）
        (
            "因此 $a + b = \\boxed{-3+26=\\boxed{23}}$",
            "23",  # BoxedAnswerExtractor 提取最内层
            "因此 $a + b = \\boxed{-3+26=\\boxed{[MASKED]}}$"
        ),
    ]
    
    all_passed = True
    for i, (input_text, expected_extracted, expected_output) in enumerate(test_cases, 1):
        extracted = boxed_extractor.extract(input_text)
        result = mask_answer(input_text, boxed_extractor)
        
        extract_ok = extracted == expected_extracted
        mask_ok = result == expected_output
        passed = extract_ok and mask_ok
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"测试 {i}: {status}")
        print(f"  输入:     {input_text}")
        print(f"  提取答案: {extracted} (期望: {expected_extracted}) {'✅' if extract_ok else '❌'}")
        print(f"  Mask后:   {result}")
        print(f"  期望:     {expected_output} {'✅' if mask_ok else '❌'}")
        print()
    
    print("=" * 70)
    print(f"测试结果: {'全部通过 ✅' if all_passed else '有失败 ❌'}")
    print("=" * 70)
