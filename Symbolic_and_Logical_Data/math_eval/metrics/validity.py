#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validity 指标 - 数学推理数据验证

支持两种验证方式：
1. 代码执行验证（Code/TIR 类型）：执行代码 → 与预期输出比较
2. 答案验证（NL/CoT 类型）：提取 \boxed{} 答案 → 与 ground_truth 比较

指标定义 (来自论文):
    f_check(q, c, y) = 1 if 验证通过, else 0
    
    Acc_verify = (1/N) * Σ f_check(q_i, c_i, y_i)

多代码块处理说明：
    当前实现只处理"多个 solution 独立"的情况：
    
    1. OpenMath 数据集：
       - solution 是单个字符串，可能包含多个 <llm-code> 块
       - 这些代码块是前后依赖的（后续代码块可能使用前面定义的变量）
       - 当前只提取并验证第一个代码块与第一个 <llm-code-output>
       - 这种前后依赖的情况，我们认为第一个代码块是主要逻辑，因此validity指标主要验证第一个代码块的执行结果
    
    2. LILA 数据集：
       - solution 是列表，每个元素是独立的解法程序
       - 多个 solution 是对同一个 answer 的多种解法
       - 每个程序独立执行，结果与 ground_truth 比较
    
    如需支持 OpenMath 的多代码块依赖执行，可以：
    - 添加 MultiCodeComparator 定义不同数据集的多代码块比较方式
    - 使用 IPython TerminalInteractiveShell 保持会话状态
    - 比较每一个代码块的llm ouput

使用方式:
    from validity import compute_validity
    from code_executor import OpenMathCodeExtractor, OpenMathExecutor
    from loaders import OpenMathInstructLoader
    
    loader = OpenMathInstructLoader('/path/to/OpenMathInstruct-1')
    
    results = compute_validity(
        data_iterator=loader.iterate(),
        extractor=OpenMathCodeExtractor(),
        executor=OpenMathExecutor(),
        output_file='validity_results.json'
    )
"""

import sys
sys.set_int_max_str_digits(0)

import re
import json
import time
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Any, Iterator, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import MathSample
from code_executor import (
    CodeExtractor, CodeExecutor, AnswerExtractor,
    compare_results, compare_math_answers,
    ResultComparator, get_comparator
)
from openmath_executor import BoxedAnswerExtractor

warnings.filterwarnings('ignore')


# =============================================================================
# 多进程辅助函数
# =============================================================================

def _process_single_sample(
    sample_data: Tuple[int, str, str, str, str, str, str, Dict],
    code_extractor_class: type,
    executor_class: type,
    answer_extractor_class: type,
    comparator_class: type,
) -> Dict[str, Any]:
    """
    处理单个样本（在子进程中执行）
    
    Args:
        sample_data: (idx, sample_id, question, solution, ground_truth, source_dataset, question_type, metadata)
        code_extractor_class: 代码提取器类
        executor_class: 执行器类
        answer_extractor_class: 答案提取器类
        comparator_class: 结果比较器类
    
    Returns:
        处理结果字典
    """
    # 在子进程中忽略所有警告
    import warnings
    import sys
    warnings.filterwarnings('ignore')
    
    idx, sample_id, question, solution, ground_truth, source_dataset, question_type, metadata = sample_data
    
    # DEBUG: 每1万条输出一次进度
    debug_this = (idx % 10000 == 0)
    if debug_this:
        print(f"[WORKER] idx={idx}", file=sys.stderr, flush=True)
    
    # 在子进程中创建实例
    code_extractor = code_extractor_class()
    executor = executor_class()
    answer_extractor = answer_extractor_class()
    comparator = comparator_class()
    
    result = {
        'idx': idx,
        'sample_id': sample_id,
        'has_code': False,
        'code_match': False,
        'code_error': None,
        'code_mismatch': False,
        'nl_match': False,
        'nl_mismatch': False,
        'nl_no_answer': False,
        'nl_no_gt': False,
        'error_detail': None,
        'mismatch_detail': None,
    }
    
    # 判断 solution 是单个还是多个
    solutions = solution if isinstance(solution, list) else [solution]
    
    # 提取代码
    codes = []
    for sol in solutions:
        code = code_extractor.extract(sol)
        if code is not None:
            codes.append(code)
    
    if codes:
        # === 有代码：执行验证 ===
        result['has_code'] = True
        
        any_error = False
        error_msg = None
        exec_results = []
        
        for code in codes:
            exec_result, error = executor.execute(code)
            if error:
                any_error = True
                error_msg = error
                break
            exec_results.append(exec_result)
        
        if any_error:
            result['code_error'] = error_msg
            result['error_detail'] = {
                'sample_id': sample_id,
                'question': question,
                'ground_truth': ground_truth,
                'error': error_msg,
                'code': codes[0] if codes else None,
            }
        else:
            # 比较结果
            expected = code_extractor.extract_output(solutions[0]) if len(solutions) == 1 else None
            if expected is None:
                expected = ground_truth
            
            if expected is None:
                result['code_match'] = True  # 没有预期输出，视为匹配
            else:
                # 每个程序的 result 都要与整个 expected（可能是列表）匹配
                all_match = True
                mismatch_result = None
                
                for i, exec_result in enumerate(exec_results):
                    try:
                        if not comparator.compare(exec_result, expected):
                            all_match = False
                            mismatch_result = exec_result
                            break
                    except Exception as e:
                        # SymPy 对象可能无法 str/repr，所以只输出类型
                        print(f"[ERROR] comparator.compare failed at idx={idx}, sample_id={sample_id}", file=sys.stderr, flush=True)
                        print(f"[ERROR] exec_result type={type(exec_result).__name__}", file=sys.stderr, flush=True)
                        print(f"[ERROR] expected type={type(expected).__name__}", file=sys.stderr, flush=True)
                        print(f"[ERROR] exception: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                        # 不 raise，视为 mismatch
                        all_match = False
                        mismatch_result = f"<compare error: {type(e).__name__}>"
                        break
                
                if all_match:
                    result['code_match'] = True
                else:
                    result['code_mismatch'] = True
                    # 安全地转换为字符串，避免 SymPy 对象 str() 报错
                    try:
                        expected_str = repr(expected) if hasattr(expected, 'is_number') else str(expected)
                    except:
                        expected_str = f"<{type(expected).__name__}>"
                    try:
                        actual_str = repr(mismatch_result) if hasattr(mismatch_result, 'is_number') else str(mismatch_result)
                    except:
                        actual_str = f"<{type(mismatch_result).__name__}>" if mismatch_result is not None else None
                    
                    # 检查是否是"无解"情况（空列表/空字典）
                    is_empty_result = actual_str in ('[]', '{}', 'None', '')
                    is_inf_result = actual_str in ('inf', '-inf', 'nan')
                    
                    # 检测数据质量问题的标记
                    question_lower = str(question).lower() if question else ''
                    code_lower = str(codes[0]).lower() if codes else ''
                    
                    data_quality_tag = ''
                    if is_empty_result:
                        data_quality_tag = '(无解)'
                    elif is_inf_result:
                        data_quality_tag = '(inf)'
                    elif 'orthogonalize' in question_lower or 'normalize' in question_lower:
                        # 正交化/归一化问题：向量方向可以取反，数学等价但符号不同
                        data_quality_tag = '(共线)'
                    elif 'eigenvalue' in question_lower or 'eigenvector' in question_lower or '.eig' in code_lower:
                        # 特征值/特征向量问题：特征向量可以乘以任意常数
                        data_quality_tag = '(共线)'
                    elif 'null space' in question_lower or 'nullspace' in question_lower or '.nullspace' in code_lower:
                        # 零空间问题：零空间向量可以缩放
                        data_quality_tag = '(共线)'
                    else:
                        # 对于没有特殊标签的样本，显示 output 和 gt 的简短信息
                        out_short = actual_str[:20] + '...' if actual_str and len(actual_str) > 20 else actual_str
                        gt_short = expected_str[:20] + '...' if expected_str and len(expected_str) > 20 else expected_str
                        data_quality_tag = f'(out:{out_short}|gt:{gt_short})'
                    
                    result['mismatch_detail'] = {
                        'sample_id': sample_id,
                        'question': question,
                        'ground_truth': ground_truth,
                        'code': codes[0] if codes else None,
                        'expected': expected_str,
                        'actual': actual_str,
                        'is_empty_result': is_empty_result,
                        'data_quality_tag': data_quality_tag,
                    }
    else:
        # === 无代码：答案验证 ===
        # 提取的答案 vs ground_truth
        sol_str = solutions[0] if solutions else ''
        extracted_answer = answer_extractor.extract(sol_str)
        
        # 获取 ground_truth
        gt = ground_truth
        if isinstance(gt, list):
            gt = str(gt)
        
        if extracted_answer is None or extracted_answer == '':
            result['nl_no_answer'] = True
        elif gt is None or gt == '':
            result['nl_no_gt'] = True
        else:
            # 使用 compare_math_answers 进行数学等价性比较
            match = compare_math_answers(str(extracted_answer), str(gt))
            if match:
                result['nl_match'] = True
            else:
                result['nl_mismatch'] = True
                result['mismatch_detail'] = {
                    'sample_id': sample_id,
                    'question': question,
                    'solution': solution,
                    'extracted_answer': extracted_answer,
                    'ground_truth': ground_truth,
                }
    
    return result


def compute_validity_parallel(
    data_iterator: Iterator[MathSample],
    code_extractor_class: type,
    executor_class: type,
    answer_extractor_class: type = BoxedAnswerExtractor,
    comparator_class: type = ResultComparator,
    output_file: Optional[str] = None,
    progress_interval: int = 50000,
    total_count: Optional[int] = None,
    dataset_name: str = 'unknown',
    num_workers: Optional[int] = None,
    chunk_size: int = 100,
) -> Dict[str, Any]:
    """
    计算 Validity 指标（多进程并行版本）
    
    注意：需要传入类而不是实例，因为实例无法跨进程序列化
    
    Args:
        data_iterator: MathSample 迭代器
        code_extractor_class: 代码提取器类（如 OpenMathCodeExtractor）
        executor_class: 代码执行器类（如 OpenMathExecutor）
        answer_extractor_class: 答案提取器类（默认 BoxedAnswerExtractor）
        comparator_class: 结果比较器类（默认 ResultComparator）
        output_file: 结果输出文件
        progress_interval: 进度显示间隔
        total_count: 总数（用于显示进度百分比）
        dataset_name: 数据集名称
        num_workers: 并行进程数（默认为 CPU 核心数）
        chunk_size: 每批处理的样本数
        
    Returns:
        结果字典
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # 最多使用 32 个进程
    
    print("=" * 70)
    print(f"Validity 评估 - {dataset_name} (并行模式, {num_workers} workers)")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(flush=True)
    
    start_time = time.time()
    
    # 将迭代器转换为可序列化的数据列表
    print(">>> 加载数据到内存...", flush=True)
    samples_data = []
    for idx, sample in enumerate(data_iterator):
        samples_data.append((
            idx,
            sample.sample_id,
            sample.question,
            sample.solution,
            sample.ground_truth,
            sample.source_dataset,
            sample.question_type,
            sample.metadata or {},
        ))
    
    total = len(samples_data)
    print(f">>> 共 {total:,} 条数据，开始并行处理...", flush=True)
    
    # 统计变量
    with_code = 0
    no_code = 0
    code_matches = 0
    code_exec_errors = 0
    code_mismatches = 0
    nl_matches = 0
    nl_mismatches = 0
    nl_no_answer = 0
    nl_no_gt = 0
    
    code_error_samples = []
    code_mismatch_samples = []
    nl_mismatch_samples = []
    
    # 记录最近一批（进度间隔内）的不匹配样本 idx
    recent_mismatch_ids = []  # 格式: "sample_id" 或 "sample_id(无解)"
    recent_error_ids = []
    
    # 创建处理函数
    process_func = partial(
        _process_single_sample,
        code_extractor_class=code_extractor_class,
        executor_class=executor_class,
        answer_extractor_class=answer_extractor_class,
        comparator_class=comparator_class,
    )
    
    # 使用进程池并行处理
    # maxtasksperchild: 每个 worker 处理 1000 个任务后重启，释放内存
    processed = 0
    with Pool(processes=num_workers, maxtasksperchild=1000) as pool:
        for result in pool.imap(process_func, samples_data, chunksize=chunk_size):
            processed += 1
            
            if result['has_code']:
                with_code += 1
                if result['code_error']:
                    code_exec_errors += 1
                    recent_error_ids.append(result['sample_id'])
                    if result['error_detail']:
                        code_error_samples.append(result['error_detail'])
                elif result['code_match']:
                    code_matches += 1
                elif result['code_mismatch']:
                    code_mismatches += 1
                    if result['mismatch_detail']:
                        code_mismatch_samples.append(result['mismatch_detail'])
                        # 使用 worker 中计算的 data_quality_tag
                        tag = result['mismatch_detail'].get('data_quality_tag', '')
                        recent_mismatch_ids.append(f"{result['sample_id']}{tag}")
            else:
                no_code += 1
                if result['nl_no_answer']:
                    nl_no_answer += 1
                elif result['nl_no_gt']:
                    nl_no_gt += 1
                elif result['nl_match']:
                    nl_matches += 1
                elif result['nl_mismatch']:
                    nl_mismatches += 1
                    recent_mismatch_ids.append(result['sample_id'])
                    if result['mismatch_detail']:
                        nl_mismatch_samples.append(result['mismatch_detail'])
            
            # 进度显示
            if progress_interval and processed % progress_interval == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed
                code_rate = code_matches / with_code if with_code > 0 else 0
                pct = processed / total * 100
                print(f"  [{processed:,}/{total:,}] ({pct:.1f}%) {speed:.0f} 条/秒, 代码匹配率: {code_rate:.2%}", flush=True)
                
                # 输出本批次不匹配的样本 idx（如果有）
                if recent_mismatch_ids:
                    print(f"    不匹配样本: {recent_mismatch_ids}", flush=True)
                if recent_error_ids:
                    print(f"    执行错误样本: {recent_error_ids}", flush=True)
                
                # 清空本批次记录
                recent_mismatch_ids = []
                recent_error_ids = []
    
    elapsed = time.time() - start_time
    
    # 计算指标
    code_total = with_code
    code_correct = code_matches
    code_acc = code_correct / code_total if code_total > 0 else 0.0
    
    nl_total = nl_matches + nl_mismatches
    nl_correct = nl_matches
    nl_acc = nl_correct / nl_total if nl_total > 0 else 0.0
    
    overall_correct = code_correct + nl_correct
    overall_total = code_total + nl_total
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    
    # 构建结果
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'num_workers': num_workers,
        
        'total': total,
        'with_code': with_code,
        'no_code': no_code,
        
        'code_matches': code_matches,
        'code_exec_errors': code_exec_errors,
        'code_mismatches': code_mismatches,
        'code_no_expected': 0,
        'code_acc': code_acc,
        
        'nl_matches': nl_matches,
        'nl_mismatches': nl_mismatches,
        'nl_no_answer': nl_no_answer,
        'nl_no_gt': nl_no_gt,
        'nl_acc': nl_acc,
        
        'overall_acc': overall_acc,
        
        'code_error_samples': code_error_samples,
        'code_mismatch_samples': code_mismatch_samples,
        'nl_mismatch_samples': nl_mismatch_samples,
    }
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成 - 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"处理速度: {total/elapsed:.0f} 条/秒")
    print("=" * 70)
    print()
    
    print(f"【基础统计】")
    print(f"  总样本数:     {total:,}")
    print(f"  有代码:       {with_code:,} ({with_code/total*100:.1f}%)")
    print(f"  无代码:       {no_code:,} ({no_code/total*100:.1f}%)")
    print()
    
    print(f"【代码执行验证】")
    print(f"  ✅ 匹配:       {code_matches:,}")
    print(f"  ❌ 执行错误:   {code_exec_errors:,}")
    print(f"  ❌ 结果不匹配: {code_mismatches:,}")
    print(f"  📊 Code Acc:   {code_acc:.4f} ({code_acc:.2%})")
    print()
    
    print(f"【答案验证】")
    print(f"  ✅ 匹配:       {nl_matches:,}")
    print(f"  ❌ 不匹配:     {nl_mismatches:,}")
    print(f"  📊 NL Acc:     {nl_acc:.4f} ({nl_acc:.2%})")
    print()
    
    print(f"【整体指标】")
    print(f"  📊 Overall Acc: {overall_acc:.4f} ({overall_acc:.2%})")
    print()
    print("=" * 70)
    
    return results


def compute_validity(
    data_iterator: Iterator[MathSample],
    code_extractor: CodeExtractor,
    executor: CodeExecutor,
    answer_extractor: Optional[AnswerExtractor] = None,
    comparator: Optional[ResultComparator] = None,
    output_file: Optional[str] = None,
    progress_interval: int = 50000,
    total_count: Optional[int] = None,
    dataset_name: str = 'unknown'
) -> Dict[str, Any]:
    """
    计算 Validity 指标
    
    对于有代码的样本：执行代码，与预期输出比较
    对于无代码的样本：提取答案，与 ground_truth 比较
    
    Args:
        data_iterator: MathSample 迭代器
        code_extractor: 代码提取器
        executor: 代码执行器
        answer_extractor: 答案提取器（可选，默认使用 BoxedAnswerExtractor）
        comparator: 结果比较器（可选，默认使用 ResultComparator）
        output_file: 结果输出文件
        progress_interval: 进度显示间隔
        total_count: 总数
        dataset_name: 数据集名称
        
    Returns:
        结果字典
    """
    # 默认使用 BoxedAnswerExtractor
    if answer_extractor is None:
        answer_extractor = BoxedAnswerExtractor()
    # 默认使用 ResultComparator
    if comparator is None:
        comparator = ResultComparator()
    start_time = time.time()
    
    # === 统计 ===
    total = 0
    
    # 有代码样本统计
    with_code = 0
    code_matches = 0
    code_exec_errors = 0
    code_mismatches = 0
    code_no_expected = 0
    
    # 无代码样本统计
    no_code = 0
    nl_matches = 0
    nl_mismatches = 0
    nl_no_answer = 0  # 无法提取答案
    nl_no_gt = 0      # 无 ground_truth
    
    # 详细样本
    code_error_samples: List[Dict] = []
    code_mismatch_samples: List[Dict] = []
    nl_mismatch_samples: List[Dict] = []
    
    # 记录最近一批（进度间隔内）的不匹配样本 idx
    recent_mismatch_ids = []
    recent_error_ids = []
    
    print("=" * 70, flush=True)
    print("Validity Verification", flush=True)
    print("=" * 70, flush=True)
    print(f"数据集: {dataset_name}", flush=True)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)
    
    # 开始迭代前输出一条确认信息
    print(">>> 开始迭代数据...", flush=True)
    
    # DEBUG 输出间隔（每1万条输出一次）
    DEBUG_INTERVAL = 10000
    
    for sample in data_iterator:
        total += 1
        
        # DEBUG: 每1万条数据输出一次
        if total % DEBUG_INTERVAL == 1 or total == 1:
            print(f"[DEBUG] 处理样本 #{total} ({sample.sample_id})...", flush=True)
        
        # 判断 solution 是单个还是多个
        solutions = sample.solution if isinstance(sample.solution, list) else [sample.solution]
        
        # 提取代码（从第一个 solution 提取，或者对于 LILA 直接就是代码）
        codes = []
        for sol in solutions:
            code = code_extractor.extract(sol)
            if code is not None:
                codes.append(code)
        
        if codes:
            # === 有代码：执行验证 ===
            with_code += 1
            
            # 对于多个程序，全部执行，结果都要和 GT 一致
            all_passed = True
            any_error = False
            error_msg = None
            results = []
            
            for code in codes:
                result, error = executor.execute(code)
                if error:
                    any_error = True
                    error_msg = error
                    break
                results.append(result)
            
            if any_error:
                code_exec_errors += 1
                recent_error_ids.append(sample.sample_id)
                code_error_samples.append({
                    'sample_id': sample.sample_id,
                    'question': sample.question,
                    'ground_truth': sample.ground_truth,
                    'error': error_msg,
                    'code': codes[0] if codes else None,
                })
            else:
                # 比较结果
                expected = code_extractor.extract_output(solutions[0]) if len(solutions) == 1 else None
                
                # 如果没有预期输出，用 ground_truth 比较
                if expected is None:
                    expected = sample.ground_truth
                
                if expected is None:
                    code_no_expected += 1
                    code_matches += 1
                else:
                    # 检查所有程序的结果是否都和 GT 一致
                    # 每个程序的 result 都要与整个 expected（可能是列表）匹配
                    all_match = True
                    mismatch_result = None
                    
                    for idx, result in enumerate(results):
                        if not comparator.compare(result, expected):
                            all_match = False
                            mismatch_result = result
                            break
                    
                    if all_match:
                        code_matches += 1
                    else:
                        code_mismatches += 1
                        actual_str = str(mismatch_result) if mismatch_result is not None else None
                        is_empty = actual_str in ('[]', '{}', 'None', '')
                        is_inf = actual_str in ('inf', '-inf', 'nan')
                        
                        # 检测数据质量问题的标记
                        question_str = str(sample.question).lower() if sample.question else ''
                        code_str = str(codes[0]).lower() if codes else ''
                        
                        # 在 ID 后面加标记
                        sample_tag = ''
                        expected_str = str(expected)
                        if is_empty:
                            sample_tag = '(无解)'
                        elif is_inf:
                            sample_tag = '(inf)'
                        elif 'orthogonalize' in question_str or 'normalize' in question_str:
                            # 正交化/归一化问题：向量方向可以取反，数学等价但符号不同
                            sample_tag = '(共线)'
                        elif 'eigenvalue' in question_str or 'eigenvector' in question_str or '.eig' in code_str:
                            # 特征值/特征向量问题：特征向量可以乘以任意常数
                            sample_tag = '(共线)'
                        elif 'null space' in question_str or 'nullspace' in question_str or '.nullspace' in code_str:
                            # 零空间问题：零空间向量可以缩放
                            sample_tag = '(共线)'
                        else:
                            # 对于没有特殊标签的样本，显示 output 和 gt 的简短信息
                            out_short = actual_str[:20] + '...' if actual_str and len(actual_str) > 20 else actual_str
                            gt_short = expected_str[:20] + '...' if expected_str and len(expected_str) > 20 else expected_str
                            sample_tag = f'(out:{out_short}|gt:{gt_short})'
                        
                        recent_mismatch_ids.append(f"{sample.sample_id}{sample_tag}")
                        code_mismatch_samples.append({
                            'sample_id': sample.sample_id,
                            'question': sample.question,
                            'ground_truth': sample.ground_truth,
                            'code': codes[0] if codes else None,
                            'expected': str(expected),
                            'actual': actual_str,
                            'is_empty_result': is_empty,
                            'data_quality_tag': sample_tag if sample_tag else None,
                        })
        else:
            # === 无代码：答案验证 ===
            # 提取的答案 vs ground_truth
            no_code += 1
            
            # 提取答案（使用传入的 answer_extractor）
            sol_str = solutions[0] if solutions else ''
            extracted_answer = answer_extractor.extract(sol_str)
            
            # 获取 ground_truth
            gt = sample.ground_truth
            if isinstance(gt, list):
                gt = str(gt)
            
            if extracted_answer is None or extracted_answer == '':
                nl_no_answer += 1
            elif gt is None or gt == '':
                nl_no_gt += 1
            else:
                # 使用 compare_math_answers 进行数学等价性比较
                match_result = compare_math_answers(str(extracted_answer), str(gt))
                
                if match_result:
                    nl_matches += 1
                else:
                    nl_mismatches += 1
                    recent_mismatch_ids.append(sample.sample_id)
                    nl_mismatch_samples.append({
                        'sample_id': sample.sample_id,
                        'question': sample.question,
                        'solution': sample.solution,
                        'extracted_answer': extracted_answer,
                        'ground_truth': sample.ground_truth,
                    })
        
        # 进度
        if progress_interval and total % progress_interval == 0:
            elapsed = time.time() - start_time
            speed = total / elapsed
            code_rate = code_matches / with_code if with_code > 0 else 0
            if total_count:
                pct = total / total_count * 100
                print(f"  [{total:,}/{total_count:,}] ({pct:.1f}%) {speed:.0f} 条/秒, 代码匹配率: {code_rate:.2%}", flush=True)
            else:
                print(f"  [{total:,}] {speed:.0f} 条/秒, 代码匹配率: {code_rate:.2%}", flush=True)
            
            # 输出本批次不匹配的样本 idx（如果有）
            if recent_mismatch_ids:
                print(f"    不匹配样本: {recent_mismatch_ids}", flush=True)
            if recent_error_ids:
                print(f"    执行错误样本: {recent_error_ids}", flush=True)
            
            # 清空本批次记录
            recent_mismatch_ids = []
            recent_error_ids = []
        
        # 首条数据确认
        if total == 1:
            print(f">>> 首条数据处理完成 (sample_id: {sample.sample_id})", flush=True)
    
    elapsed = time.time() - start_time
    
    # === 计算指标 ===
    # 代码验证指标
    code_acc = code_matches / with_code if with_code > 0 else 0.0
    code_exec_success = code_matches + code_mismatches + code_no_expected
    code_exec_rate = code_exec_success / with_code if with_code > 0 else 0.0
    
    # 答案验证指标
    nl_valid = nl_matches + nl_mismatches
    nl_acc = nl_matches / nl_valid if nl_valid > 0 else 0.0
    
    # 整体指标
    total_matches = code_matches + nl_matches
    total_valid = with_code + nl_valid  # 可验证的样本数
    overall_acc = total_matches / total_valid if total_valid > 0 else 0.0
    
    # 构建结果
    results = {
        # 元信息
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        
        # 总体统计
        'total': total,
        'with_code': with_code,
        'no_code': no_code,
        
        # 代码验证统计
        'code_matches': code_matches,
        'code_exec_errors': code_exec_errors,
        'code_mismatches': code_mismatches,
        'code_no_expected': code_no_expected,
        
        # 答案验证统计
        'nl_matches': nl_matches,
        'nl_mismatches': nl_mismatches,
        'nl_no_answer': nl_no_answer,
        'nl_no_gt': nl_no_gt,
        
        # 指标
        'code_acc': code_acc,
        'code_exec_rate': code_exec_rate,
        'nl_acc': nl_acc,
        'overall_acc': overall_acc,
        
        # 详细样本
        'code_error_samples': code_error_samples,
        'code_mismatch_samples': code_mismatch_samples,
        'nl_mismatch_samples': nl_mismatch_samples,
    }
    
    # 保存
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print()
    print("=" * 70)
    print(f"验证完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【基础统计】")
    print(f"  总样本数:     {total:,}")
    print(f"  有代码:       {with_code:,} ({with_code/total*100:.1f}%)" if total > 0 else "")
    print(f"  无代码:       {no_code:,} ({no_code/total*100:.1f}%)" if total > 0 else "")
    print()
    print(f"【代码执行验证】(有代码样本: {with_code:,})")
    print(f"  ✅ 匹配:       {code_matches:,}")
    print(f"  ❌ 执行错误:   {code_exec_errors:,}")
    print(f"  ❌ 结果不匹配: {code_mismatches:,}")
    print(f"  ⚠️  无预期输出: {code_no_expected:,}")
    print(f"  📊 Code Acc:   {code_acc:.4f} ({code_acc:.2%})")
    print()
    print(f"【答案验证】(无代码样本: {no_code:,})")
    print(f"  ✅ 匹配:       {nl_matches:,}")
    print(f"  ❌ 不匹配:     {nl_mismatches:,}")
    print(f"  ⚠️  无法提取答案: {nl_no_answer:,}")
    print(f"  ⚠️  无 GT:      {nl_no_gt:,}")
    print(f"  📊 NL Acc:     {nl_acc:.4f} ({nl_acc:.2%})")
    print()
    print(f"【整体指标】")
    print(f"  总匹配:        {total_matches:,}")
    print(f"  可验证样本:    {total_valid:,}")
    print(f"  📊 Overall Acc: {overall_acc:.4f} ({overall_acc:.2%})")
    print()
    print("=" * 70)
    
    return results


# 保持向后兼容
compute_code_validity = compute_validity


def load_results(result_file: str) -> Dict[str, Any]:
    """加载结果文件"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(results: Dict[str, Any]):
    """打印结果摘要"""
    print("=" * 70)
    print(f"Validity Results: {results.get('dataset', 'unknown')}")
    print("=" * 70)
    print()
    print(f"验证时间: {results.get('timestamp', 'unknown')}")
    print(f"耗时: {results.get('elapsed_seconds', 0):.1f} 秒")
    print()
    
    total = results['total']
    with_code = results['with_code']
    no_code = results['no_code']
    
    print(f"【基础统计】")
    print(f"  总样本数:     {total:,}")
    print(f"  有代码:       {with_code:,} ({with_code/total*100:.1f}%)")
    print(f"  无代码:       {no_code:,} ({no_code/total*100:.1f}%)")
    print()
    
    # 代码验证
    print(f"【代码执行验证】")
    print(f"  ✅ 匹配:       {results['code_matches']:,}")
    print(f"  ❌ 执行错误:   {results['code_exec_errors']:,}")
    print(f"  ❌ 结果不匹配: {results['code_mismatches']:,}")
    print(f"  📊 Code Acc:   {results['code_acc']:.4f} ({results['code_acc']:.2%})")
    print()
    
    # 答案验证
    print(f"【答案验证】")
    print(f"  ✅ 匹配:       {results['nl_matches']:,}")
    print(f"  ❌ 不匹配:     {results['nl_mismatches']:,}")
    print(f"  📊 NL Acc:     {results['nl_acc']:.4f} ({results['nl_acc']:.2%})")
    print()
    
    # 整体
    print(f"【整体指标】")
    print(f"  📊 Overall Acc: {results['overall_acc']:.4f} ({results['overall_acc']:.2%})")
    print()
    print("=" * 70)
