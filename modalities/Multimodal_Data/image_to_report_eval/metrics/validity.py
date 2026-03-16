#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validity 指标 - Image-to-Report 数据有效性检查

两个层面的有效性验证：

1. 图片有效性 (Image Validity, 静态检查):
   - 图片路径是否存在
   - 图片文件是否可读（能否被 PIL 正常打开）
   - 图片基本属性（尺寸、格式、通道数）

2. 报告有效性 (Report Validity, LLM Multimodal Judge):
   - 报告忠实性 (Faithfulness): 报告描述的内容在图片中是否真实存在 (report → image)
   - 指令合规性 (Compliance): 报告是否按照指令的要求来写 (report → instruction)
   - 指令相关性 (Relevance): 指令对于该图片是否合理/恰当 (instruction ↔ image)

所有数据集共用同一套流程，不需要数据集特有的 checker。

使用方式:
    from loaders import IUXRayLoader
    from metrics.validity import compute_validity

    loader = IUXRayLoader('/path/to/IU-Xray', split='train')

    results = compute_validity(
        data_iterator=loader.iterate(),
        image_base_dir='/path/to/images',
        dataset_name='IU X-Ray (train)',
        output_file='results/validity_iu_xray.json',
        max_samples=100,   # LLM judge 较慢，建议先小批量测试
    )
"""

import os
import sys
import json
import time
import base64
from datetime import datetime
from typing import Optional, Iterator, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample


# =============================================================================
# LLM 配置
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4o'


# =============================================================================
# Prompt 模板
# =============================================================================

REPORT_VALIDITY_PROMPT = """你是一个多模态数据质量评估专家。请根据提供的图片、指令和报告，评估数据的有效性。

【指令 (Instruction)】
{instruction}

【报告 (Report)】
{report}

请逐句审查报告内容，并评估以下三个维度（每个维度只能打 0 / 0.5 / 1 分）：

1. **报告忠实性 (Faithfulness)**：报告中描述的内容，在图片中是否真实存在？
   - 请逐一检查报告中的每个描述/发现，判断图片是否支持该描述
   - 1: 报告的所有描述均可在图片中得到验证，没有凭空捏造的内容
   - 0.5: 报告大部分描述与图片一致，但包含少量图片无法验证的描述
   - 0: 报告包含大量与图片不符或图片中不存在的描述

2. **指令合规性 (Compliance)**：报告是否按照指令的要求来写？
   - 关注指令对格式、覆盖范围、关注点的具体要求
   - 1: 报告完全按照指令要求的格式和内容来组织
   - 0.5: 报告大体符合指令，但在格式或覆盖范围上有偏差
   - 0: 报告完全没有遵循指令的要求

3. **指令相关性 (Relevance)**：指令对于这张图片来说是否合理/恰当？
   - 1: 指令与图片高度匹配，是对该图片合理的任务要求
   - 0.5: 指令与图片有一定关联，但部分要求对该图片不太适用
   - 0: 指令与图片内容明显不相关，或对该图片完全不适用

请以 JSON 格式输出：
```json
{{
    "faithfulness": {{
        "score": 0/0.5/1,
        "reason": "逐一列出报告中的关键描述，并说明图片是否支持"
    }},
    "compliance": {{
        "score": 0/0.5/1,
        "reason": "简要说明指令合规性判断理由"
    }},
    "relevance": {{
        "score": 0/0.5/1,
        "reason": "简要说明指令与图片的匹配程度"
    }}
}}
```

只输出 JSON，不要其他内容。"""


# =============================================================================
# 图片有效性检查（静态）
# =============================================================================

def _check_image_validity(
    image_paths: List[str],
    image_base_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    检查图片的有效性

    Args:
        image_paths: 图片路径列表（来自 sample.images）
        image_base_dir: 图片根目录，用于拼接相对路径

    Returns:
        (image_infos, errors)
        - image_infos: 每张图片的详细信息
        - errors: 错误列表
    """
    image_infos = []
    errors = []

    for i, path in enumerate(image_paths):
        info = {"index": i, "raw_path": path}

        # 拼接完整路径
        if image_base_dir and not os.path.isabs(path):
            full_path = os.path.join(image_base_dir, path)
        else:
            full_path = path
        info["full_path"] = full_path

        # 1. 文件存在性
        if not os.path.exists(full_path):
            info["exists"] = False
            errors.append(f"images[{i}] 文件不存在: {full_path}")
            image_infos.append(info)
            continue
        info["exists"] = True

        # 2. 文件大小
        file_size = os.path.getsize(full_path)
        info["file_size_bytes"] = file_size
        if file_size == 0:
            errors.append(f"images[{i}] 文件大小为 0: {full_path}")
            image_infos.append(info)
            continue

        # 3. 图片可读性（PIL）
        try:
            from PIL import Image
            with Image.open(full_path) as img:
                img.verify()
            # verify() 后需要重新打开才能读取属性
            with Image.open(full_path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["format"] = img.format
                info["mode"] = img.mode
                info["readable"] = True
        except Exception as e:
            info["readable"] = False
            errors.append(f"images[{i}] 无法读取: {e}")

        image_infos.append(info)

    return image_infos, errors


# =============================================================================
# 图片编码（供 LLM 多模态调用）
# =============================================================================

def _encode_image_base64(image_path: str) -> Optional[str]:
    """将图片编码为 base64 字符串"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _get_mime_type(image_path: str) -> str:
    """根据扩展名推断 MIME 类型"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


# =============================================================================
# LLM 调用
# =============================================================================

def _call_multimodal_llm(
    prompt: str,
    image_paths: List[str],
    image_base_dir: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    调用多模态 LLM API（OpenAI Vision 格式）

    Args:
        prompt: 文本提示词
        image_paths: 图片路径列表
        image_base_dir: 图片根目录
        max_retries: 最大重试次数

    Returns:
        LLM 响应文本，失败返回 None
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI 库未安装，请运行: pip install openai")
        return None

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # 构建多模态 content
    content = []

    for path in image_paths:
        if image_base_dir and not os.path.isabs(path):
            full_path = os.path.join(image_base_dir, path)
        else:
            full_path = path

        b64 = _encode_image_base64(full_path)
        if b64 is None:
            continue

        mime = _get_mime_type(full_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    content.append({"type": "text", "text": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """解析 LLM Judge 响应"""
    _default = {
        "faithfulness": {"score": 0.0, "reason": "LLM 调用失败"},
        "compliance": {"score": 0.0, "reason": "LLM 调用失败"},
        "relevance": {"score": 0.0, "reason": "LLM 调用失败"},
        "parse_error": True,
    }
    if not response:
        return _default

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        _default["faithfulness"]["reason"] = f"解析失败: {e}"
        _default["compliance"]["reason"] = f"解析失败: {e}"
        _default["relevance"]["reason"] = f"解析失败: {e}"
        _default["raw_response"] = response
        return _default


# =============================================================================
# 单条样本处理
# =============================================================================

def _process_single_sample(
    sample: ImageToReportSample,
    image_base_dir: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    处理单条样本的有效性检查

    Args:
        sample: 样本
        image_base_dir: 图片根目录
        skip_llm: 是否跳过 LLM Judge（只做图片静态检查）

    Returns:
        检查结果字典
    """
    result = {
        "sample_id": sample.sample_id,
        "image_count": len(sample.images),
    }

    # =========== 1. 图片有效性 ===========
    image_infos, image_errors = _check_image_validity(
        sample.images, image_base_dir
    )
    result["image_validity"] = {
        "all_valid": len(image_errors) == 0,
        "errors": image_errors,
        "images": image_infos,
    }

    # =========== 2. 报告有效性 (LLM Judge) ===========
    if skip_llm:
        result["report_validity"] = {"skipped": True}
        return result

    # 只有所有图片都可用时才调用 LLM
    all_images_readable = all(
        info.get("readable", False) or info.get("exists", False)
        for info in image_infos
    )

    if not all_images_readable:
        result["report_validity"] = {
            "skipped": True,
            "reason": "图片不可用，跳过 LLM Judge",
        }
        return result

    # 构建 prompt（去掉 instruction 中的 <image> token，避免干扰 LLM）
    clean_instruction = sample.instruction.replace("<image>", "").strip()
    prompt = REPORT_VALIDITY_PROMPT.format(
        instruction=clean_instruction,
        report=sample.report,
    )

    response = _call_multimodal_llm(
        prompt=prompt,
        image_paths=sample.images,
        image_base_dir=image_base_dir,
    )

    parsed = _parse_llm_response(response)

    faithfulness_score = parsed.get("faithfulness", {}).get("score", 0.0)
    compliance_score = parsed.get("compliance", {}).get("score", 0.0)
    relevance_score = parsed.get("relevance", {}).get("score", 0.0)

    result["report_validity"] = {
        "faithfulness": parsed.get("faithfulness", {}),
        "compliance": parsed.get("compliance", {}),
        "relevance": parsed.get("relevance", {}),
        "faithfulness_score": float(faithfulness_score),
        "compliance_score": float(compliance_score),
        "relevance_score": float(relevance_score),
        "llm_error": parsed.get("parse_error", False),
    }

    return result


# =============================================================================
# 主评估函数
# =============================================================================

def compute_validity(
    data_iterator: Iterator[ImageToReportSample],
    image_base_dir: Optional[str] = None,
    dataset_name: str = "unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
    skip_llm: bool = False,
    parallel: bool = False,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    计算有效性指标

    Args:
        data_iterator: ImageToReportSample 迭代器
        image_base_dir: 图片根目录（用于拼接相对路径）
        dataset_name: 数据集名称
        output_file: 结果输出 JSON 路径
        max_samples: 最大样本数（LLM judge 较慢，建议先小批量测试）
        progress_interval: 进度显示间隔
        skip_llm: 是否跳过 LLM Judge（只做图片静态检查）
        parallel: 是否并行调用 LLM（多线程）
        max_workers: 并行线程数

    Returns:
        结果字典
    """
    mode_str = "仅图片检查" if skip_llm else ("并行" if parallel else "串行")
    print("=" * 70)
    print(f"Validity Evaluation - {mode_str}")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"图片根目录: {image_base_dir or '(使用原始路径)'}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    if not skip_llm:
        print(f"LLM: {LLM_MODEL}")
    print()

    start_time = time.time()

    # 收集样本
    all_samples = []
    for sample in data_iterator:
        if max_samples and len(all_samples) >= max_samples:
            break
        all_samples.append(sample)

    total_to_process = len(all_samples)
    print(f"共 {total_to_process:,} 条样本待处理")
    print()

    # 统计
    total = 0
    image_valid_count = 0
    image_error_count = 0
    total_faithfulness = 0.0
    total_compliance = 0.0
    total_relevance = 0.0
    llm_evaluated = 0
    llm_failures = 0

    sample_results = []

    # ==================== 处理 ====================
    if parallel and not skip_llm:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_sample, sample, image_base_dir, skip_llm
                ): sample.sample_id
                for sample in all_samples
            }

            for future in as_completed(futures):
                total += 1
                result = future.result()
                sample_results.append(result)

                # 统计图片有效性
                if result["image_validity"]["all_valid"]:
                    image_valid_count += 1
                else:
                    image_error_count += 1

                # 统计报告有效性
                rv = result.get("report_validity", {})
                if not rv.get("skipped", False):
                    llm_evaluated += 1
                    if rv.get("llm_error", False):
                        llm_failures += 1
                    else:
                        total_faithfulness += rv.get("faithfulness_score", 0.0)
                        total_compliance += rv.get("compliance_score", 0.0)
                        total_relevance += rv.get("relevance_score", 0.0)

                if progress_interval and total % progress_interval == 0:
                    _print_progress(total, total_to_process, start_time,
                                    image_valid_count, llm_evaluated,
                                    total_faithfulness, total_compliance,
                                    total_relevance)
    else:
        for sample in all_samples:
            total += 1
            result = _process_single_sample(sample, image_base_dir, skip_llm)
            sample_results.append(result)

            if result["image_validity"]["all_valid"]:
                image_valid_count += 1
            else:
                image_error_count += 1

            rv = result.get("report_validity", {})
            if not rv.get("skipped", False):
                llm_evaluated += 1
                if rv.get("llm_error", False):
                    llm_failures += 1
                else:
                    total_faithfulness += rv.get("faithfulness_score", 0.0)
                    total_compliance += rv.get("compliance_score", 0.0)
                    total_relevance += rv.get("relevance_score", 0.0)

            if progress_interval and total % progress_interval == 0:
                _print_progress(total, total_to_process, start_time,
                                image_valid_count, llm_evaluated,
                                total_faithfulness, total_compliance,
                                total_relevance)

    elapsed = time.time() - start_time

    # 计算指标
    image_valid_rate = image_valid_count / total if total > 0 else 0.0
    llm_success = llm_evaluated - llm_failures
    avg_faithfulness = total_faithfulness / llm_success if llm_success > 0 else 0.0
    avg_compliance = total_compliance / llm_success if llm_success > 0 else 0.0
    avg_relevance = total_relevance / llm_success if llm_success > 0 else 0.0

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "image_base_dir": image_base_dir,
        "llm_model": LLM_MODEL if not skip_llm else None,

        # 总体
        "total": total,

        # 图片有效性
        "image_valid_count": image_valid_count,
        "image_error_count": image_error_count,
        "image_valid_rate": image_valid_rate,

        # 报告有效性
        "llm_evaluated": llm_evaluated,
        "llm_failures": llm_failures,
        "avg_faithfulness": avg_faithfulness,
        "avg_compliance": avg_compliance,
        "avg_relevance": avg_relevance,

        # 详细结果
        "sample_results": sample_results,
    }

    # 保存
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！耗时 {elapsed:.1f} 秒")
    print("=" * 70)
    print()
    print(f"【图片有效性】(静态检查)")
    print(f"  总样本数:     {total:,}")
    print(f"  图片有效:     {image_valid_count:,} ({image_valid_rate:.2%})")
    print(f"  图片异常:     {image_error_count:,}")
    print()

    if not skip_llm:
        print(f"【报告有效性】(LLM Judge: {LLM_MODEL})")
        print(f"  LLM 评估数:   {llm_evaluated:,}")
        print(f"  LLM 失败:     {llm_failures}")
        print(f"  报告忠实性:   {avg_faithfulness:.3f}")
        print(f"  指令合规性:   {avg_compliance:.3f}")
        print(f"  指令相关性:   {avg_relevance:.3f}")
        print()

        # 分数分布
        if sample_results:
            _print_score_distribution(sample_results)

    # 图片错误类型统计
    if image_error_count > 0:
        _print_image_error_summary(sample_results)

    print("=" * 70)

    return results


# =============================================================================
# 辅助打印函数
# =============================================================================

def _print_progress(total, total_to_process, start_time,
                    image_valid_count, llm_evaluated,
                    total_faithfulness, total_compliance,
                    total_relevance):
    """打印进度"""
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    img_rate = image_valid_count / total if total > 0 else 0
    llm_success = llm_evaluated
    avg_faith = total_faithfulness / llm_success if llm_success > 0 else 0
    avg_comp = total_compliance / llm_success if llm_success > 0 else 0
    avg_rel = total_relevance / llm_success if llm_success > 0 else 0
    print(
        f"  [{total:,}/{total_to_process:,}] {rate:.1f} 条/秒 | "
        f"图片有效: {img_rate:.2%} | "
        f"忠实: {avg_faith:.2f} | 合规: {avg_comp:.2f} | 相关: {avg_rel:.2f}"
    )


def _print_score_distribution(sample_results: List[Dict]):
    """打印 LLM Judge 分数分布"""
    faith_scores = []
    comp_scores = []
    rel_scores = []

    for r in sample_results:
        rv = r.get("report_validity", {})
        if rv.get("skipped", False) or rv.get("llm_error", False):
            continue
        faith_scores.append(rv.get("faithfulness_score", 0.0))
        comp_scores.append(rv.get("compliance_score", 0.0))
        rel_scores.append(rv.get("relevance_score", 0.0))

    if not faith_scores:
        return

    n = len(faith_scores)
    for label, scores in [("报告忠实性", faith_scores),
                          ("指令合规性", comp_scores),
                          ("指令相关性", rel_scores)]:
        print(f"  {label}分布 (N={n}):")
        for score_val in [1.0, 0.5, 0.0]:
            count = sum(1 for s in scores if s == score_val)
            print(f"    {score_val}: {count}/{n} ({100 * count / n:.1f}%)")

    print()


def _print_image_error_summary(sample_results: List[Dict]):
    """打印图片错误类型汇总"""
    error_types: Dict[str, int] = {}
    for r in sample_results:
        for err in r.get("image_validity", {}).get("errors", []):
            # 取错误信息的前半部分作为类型
            if "文件不存在" in err:
                key = "文件不存在"
            elif "文件大小为 0" in err:
                key = "文件大小为 0"
            elif "无法读取" in err:
                key = "无法读取"
            else:
                key = err
            error_types[key] = error_types.get(key, 0) + 1

    if error_types:
        print(f"【图片错误类型统计】")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count:,}")
        print()
