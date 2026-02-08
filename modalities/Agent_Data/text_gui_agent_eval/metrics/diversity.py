#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity Metric - GUI Agent 数据集多样性评估

适用于自然数据和合成数据的通用多样性指标，从五个维度评估数据集的多样性。

评估维度:
    1. 页面结构多样性 - APTED 树编辑距离 + Vendi Score
    2. 动作模式多样性 - 动作类型分布熵、覆盖率
    3. 域名/网站多样性 - 域名覆盖广度和分布均匀度
    4. 任务语义多样性 - 基于 Sentence Embedding 的语义距离
    5. 表达模式多样性 - Self-BLEU (检测 Teacher Bias / 模板化表述)

论文依据:
    [1] Corazza et al. "Web Application Testing: Using Tree Kernels to
        Detect Near-duplicate States" (ESEM 2021)
        → DOM 树相似度方法论 ("only body, no scripts" 表示策略)
    [2] Pawlik & Augsten "Tree edit distance: Robust and memory-efficient"
        (Information Systems, 2016)
        → APTED 树编辑距离算法, Python 库 `apted` 是官方实现
    [3] Friedman & Dieng "The Vendi Score: A Diversity Metric for Machine
        Learning" (ICML 2023)
        → 基于核矩阵的通用多样性度量, Python 库 `vendi-score` 是官方实现
    [4] Zhu et al. "Texygen: A Benchmarking Platform for Text Generation
        Models" (SIGIR 2018)
        → Self-BLEU 多样性度量

依赖:
    pip install lxml apted vendi-score sentence-transformers nltk numpy

使用方式:
    from metrics.diversity import compute_diversity
    from loaders import SomeLoader

    loader = SomeLoader('/path/to/data')
    results = compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name='Mind2Web',
        output_file='diversity_results.json'
    )
"""

# =============================================================================
# 标准库
# =============================================================================
import os
import json
import time
import math
import statistics
from datetime import datetime
from typing import List, Dict, Any, Iterator, Optional, Tuple
from collections import Counter
from urllib.parse import urlparse

# =============================================================================
# 第三方库
# =============================================================================
import numpy as np
import lxml.html                                            # DOM 树解析 [1]
from apted import APTED                                     # 树编辑距离 [2]
from apted.helpers import Tree as AptedTree                 # 树编辑距离 [2]
from vendi_score.vendi import score_K as vendi_score_K      # Vendi Score [3]
from sentence_transformers import SentenceTransformer       # 语义嵌入
from nltk.translate.bleu_score import sentence_bleu         # Self-BLEU [4]
from nltk.translate.bleu_score import SmoothingFunction     # Self-BLEU [4]

# =============================================================================
# 项目内部
# =============================================================================
try:
    from ..data_types import Record, Action
except (ImportError, ValueError):
    from data_types import Record, Action


# =============================================================================
# 动作类型定义 (基于 Playwright 官方 API)
#
# 来源: https://playwright.dev/python/docs/api/class-locator
#       https://playwright.dev/python/docs/api/class-page
#
# 标准动作集作为 coverage 计算的分母，各数据集的动作通过映射表归一化
# =============================================================================

# Playwright Locator 方法 (元素操作)
PLAYWRIGHT_LOCATOR_ACTIONS = {
    'click',           # locator.click() - 单击
    'dblclick',        # locator.dblclick() - 双击
    'hover',           # locator.hover() - 悬停
    'tap',             # locator.tap() - 触摸点击
    'fill',            # locator.fill() - 填充输入框
    'type',            # locator.type() - 逐字符输入 (已弃用)
    'press',           # locator.press() - 按键
    'clear',           # locator.clear() - 清空输入框
    'check',           # locator.check() - 勾选复选框
    'uncheck',         # locator.uncheck() - 取消勾选
    'select_option',   # locator.select_option() - 下拉框选择
    'drag_to',         # locator.drag_to() - 拖拽
    'focus',           # locator.focus() - 获取焦点
    'blur',            # locator.blur() - 失去焦点
    'scroll_into_view',  # locator.scroll_into_view_if_needed()
    'set_input_files', # locator.set_input_files() - 上传文件
}

# Playwright Page 方法 (页面导航)
PLAYWRIGHT_PAGE_ACTIONS = {
    'goto',            # page.goto() - 跳转 URL
    'go_back',         # page.go_back() - 后退
    'go_forward',      # page.go_forward() - 前进
    'reload',          # page.reload() - 刷新
}

# 完整标准动作集 (Playwright 官方 API)
PLAYWRIGHT_STANDARD_ACTIONS = PLAYWRIGHT_LOCATOR_ACTIONS | PLAYWRIGHT_PAGE_ACTIONS


# =============================================================================
# 1. 页面结构多样性 (APTED + Vendi Score)
# =============================================================================

def _html_to_apted_tree(html_str: str) -> Optional[AptedTree]:
    """
    将 cleaned_html 解析为 apted 库的 Tree 对象 (仅保留标签结构)。
    """
    if not html_str or not html_str.strip():
        return None

    try:
        doc = lxml.html.fromstring(html_str)
    except Exception:
        return None

    body = doc.find('.//body')
    root = body if body is not None else doc

    def _to_apted(elem):
        tag = elem.tag if isinstance(elem.tag, str) else 'unknown'
        children = []
        for child in elem:
            if not isinstance(child.tag, str):
                continue
            child_tree = _to_apted(child)
            if child_tree is not None:
                children.append(child_tree)
        return AptedTree(tag, *children)

    return _to_apted(root)


def compute_dom_similarity_matrix(
    html_list: List[str],
    show_progress: bool = True,
) -> np.ndarray:
    """计算 N×N DOM 树相似度矩阵。"""
    n = len(html_list)
    if n == 0:
        return np.array([]).reshape(0, 0)

    if show_progress:
        print(f"  📄 解析 {n} 个 HTML → APTED 树...")

    trees = []
    tree_sizes = []
    parse_failures = 0

    for i, html_str in enumerate(html_list):
        tree = _html_to_apted_tree(html_str)
        trees.append(tree)
        if tree is None:
            tree_sizes.append(0)
            parse_failures += 1
        else:
            tree_sizes.append(str(tree).count('{'))
        if show_progress and (i + 1) % 500 == 0:
            print(f"    [{i+1}/{n}] 已解析")

    if show_progress:
        print(f"  ✅ 解析完成 (失败: {parse_failures}/{n})")

    K = np.eye(n, dtype=np.float64)
    total_pairs = n * (n - 1) // 2
    computed = 0

    if show_progress:
        print(f"  🔄 计算 {total_pairs:,} 对 APTED 编辑距离...")

    for i in range(n):
        if trees[i] is None:
            continue
        for j in range(i + 1, n):
            if trees[j] is None:
                continue
            dist = APTED(trees[i], trees[j]).compute_edit_distance()
            max_size = max(tree_sizes[i], tree_sizes[j], 1)
            sim = math.exp(-dist / max_size)
            K[i, j] = sim
            K[j, i] = sim
            computed += 1
            if show_progress and computed % 10000 == 0:
                print(f"    [{computed:,}/{total_pairs:,}] "
                      f"({100 * computed / total_pairs:.1f}%)")

    if show_progress:
        upper_tri = K[np.triu_indices(n, k=1)]
        print(f"  ✅ 相似度矩阵完成 ({computed:,} 对)")
        if len(upper_tri) > 0:
            print(f"     均值={np.mean(upper_tri):.4f}  "
                  f"中位数={np.median(upper_tri):.4f}  "
                  f"范围=[{np.min(upper_tri):.4f}, {np.max(upper_tri):.4f}]")

    return K


def compute_page_diversity(html_list: List[str]) -> Dict[str, Any]:
    """
    计算页面结构多样性。

    输出核心指标: Vendi Score = 等效独立页面模板数 [3]。
    """
    n = len(html_list)
    if n < 2:
        return {
            'vendi_score': float(n),
            'vendi_score_normalized': 1.0 if n == 1 else 0.0,
            'n_pages': n,
            'similarity_mean': 1.0 if n == 1 else 0.0,
            'similarity_std': 0.0,
        }

    K = compute_dom_similarity_matrix(html_list)
    vs = float(vendi_score_K(K, q=1))
    upper_tri = K[np.triu_indices(n, k=1)]

    return {
        'vendi_score': vs,
        'vendi_score_normalized': vs / n,
        'n_pages': n,
        'similarity_mean': float(np.mean(upper_tri)),
        'similarity_std': float(np.std(upper_tri)),
    }


# =============================================================================
# 2. 动作模式多样性
# =============================================================================

def compute_action_diversity(
    action_types: Counter,
    action_mapping: Optional[Dict[str, str]] = None,
    skip_actions: Optional[set] = None,
) -> Dict[str, Any]:
    """
    计算动作类型多样性。

    Args:
        action_types: 原始动作类型计数器
        action_mapping: 数据集动作 → Playwright 标准动作的映射表
        skip_actions: 跳过的动作集合

    Returns:
        action_type_entropy: 归一化熵 [0, 1]
        action_type_coverage: 覆盖率
        action_distribution: 动作分布
    """
    skip_actions = skip_actions or set()
    
    mapped_counts = Counter()
    for action, count in action_types.items():
        action_lower = action.lower()
        
        if action_lower in skip_actions or action in skip_actions:
            continue
        
        if action_mapping:
            if action_lower in action_mapping:
                mapped_counts[action_mapping[action_lower]] += count
            elif action in action_mapping:
                mapped_counts[action_mapping[action]] += count
            else:
                mapped_counts[action] += count
        else:
            mapped_counts[action] += count

    norm_entropy, _, n_types = _compute_entropy(mapped_counts)
    covered = set(mapped_counts.keys()) & PLAYWRIGHT_STANDARD_ACTIONS
    coverage = len(covered) / len(PLAYWRIGHT_STANDARD_ACTIONS)

    return {
        'action_type_entropy': norm_entropy,
        'action_type_coverage': coverage,
        'n_action_types': n_types,
        'action_distribution': dict(mapped_counts.most_common()),
    }


# =============================================================================
# 3. 域名/网站多样性
# =============================================================================

def compute_domain_diversity(domains: Counter) -> Dict[str, Any]:
    """计算域名/网站多样性。"""
    norm_entropy, _, n_domains = _compute_entropy(domains)
    n_samples = sum(domains.values())
    unique_ratio = n_domains / n_samples if n_samples > 0 else 0
    gini = _compute_gini(list(domains.values()))

    return {
        'domain_entropy': norm_entropy,
        'unique_domain_ratio': unique_ratio,
        'domain_gini': gini,
        'n_unique_domains': n_domains,
        'n_total_samples': n_samples,
        'top_domains': dict(domains.most_common(10)),
    }


# =============================================================================
# 4. 任务语义多样性 (Sentence Embedding)
# =============================================================================

def compute_task_diversity(embeddings: np.ndarray) -> Dict[str, Any]:
    """计算任务语义多样性 (基于嵌入向量的余弦相似度)。"""
    n = len(embeddings)
    if n < 2:
        return {
            'semantic_diversity': 0.0,
            'avg_pairwise_similarity': 1.0 if n == 1 else 0.0,
            'n_instructions': n,
        }

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    sim_matrix = np.dot(normalized, normalized.T)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]

    return {
        'semantic_diversity': 1.0 - float(np.mean(upper_tri)),
        'avg_pairwise_similarity': float(np.mean(upper_tri)),
        'n_instructions': n,
    }


# =============================================================================
# 5. 表达模式多样性 (Self-BLEU)
# =============================================================================

def compute_expression_diversity(
    instructions: List[str],
    batch_size: int = 200,
) -> Dict[str, Any]:
    """计算表达模式多样性 (Self-BLEU)。"""
    n = len(instructions)
    if n < 2:
        return {'self_bleu': 1.0, 'expression_diversity': 0.0, 'n_instructions': n}

    tokenized = [inst.lower().split() for inst in instructions]
    smoothing = SmoothingFunction().method1

    if n > batch_size:
        batch_bleu_scores = []
        for start in range(0, n, batch_size):
            batch = tokenized[start:start + batch_size]
            scores = []
            for i, hyp in enumerate(batch):
                refs = [batch[j] for j in range(len(batch)) if j != i]
                if hyp and refs:
                    scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))
            if scores:
                batch_bleu_scores.append(np.mean(scores))
        self_bleu = float(np.mean(batch_bleu_scores)) if batch_bleu_scores else 1.0
    else:
        scores = []
        for i, hyp in enumerate(tokenized):
            refs = [tokenized[j] for j in range(n) if j != i]
            if hyp and refs:
                scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))
        self_bleu = float(np.mean(scores)) if scores else 1.0

    return {
        'self_bleu': self_bleu,
        'expression_diversity': 1.0 - self_bleu,
        'n_instructions': n,
    }


# =============================================================================
# 工具函数
# =============================================================================

def _compute_entropy(counter: Counter) -> Tuple[float, float, int]:
    """计算归一化熵。"""
    if not counter:
        return 0.0, 0.0, 0
    n = len(counter)
    if n < 2:
        return 0.0, 0.0, n
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0, n

    entropy = -sum((c / total) * math.log(c / total) for c in counter.values() if c > 0)
    return entropy / math.log(n), entropy, n


def _compute_gini(values: List[float]) -> float:
    """计算 Gini 系数。"""
    if not values or len(values) < 2:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _extract_domain(url_or_website: str) -> str:
    """从 URL 或网站名提取域名。"""
    if not url_or_website:
        return "unknown"
    if url_or_website.startswith(('http://', 'https://')):
        try:
            return urlparse(url_or_website).netloc or url_or_website
        except Exception:
            pass
    return url_or_website


# =============================================================================
# 主函数
# =============================================================================

def compute_diversity(
    data_iterator: Iterator[Record],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress_interval: int = 100,
    debug: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    action_mapping: Optional[Dict[str, str]] = None,
    skip_actions: Optional[set] = None,
) -> Dict[str, Any]:
    """
    计算 GUI Agent 数据集多样性指标 (五个维度)。

    Args:
        data_iterator: Record 迭代器 (来自 Loader)
        dataset_name: 数据集名称
        output_file: 结果输出文件路径
        max_samples: 最大样本数
        progress_interval: 进度输出间隔
        debug: 调试模式
        embedding_model: 句子嵌入模型名称
        action_mapping: 数据集动作 → Playwright 标准动作的映射表
        skip_actions: 跳过的动作集合

    Returns:
        包含五个维度指标的字典
    """
    print("=" * 70)
    print("Diversity Evaluation")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    # ==================== 数据收集 ====================
    record_count = 0
    total_actions = 0
    action_types = Counter()
    domains = Counter()
    page_htmls = []
    instructions = []

    for record in data_iterator:
        if max_samples and record_count >= max_samples:
            break
        record_count += 1

        domain = _extract_domain(record.website or record.metadata.get('url', ''))
        domains[domain] += 1

        if record.instruction:
            instructions.append(record.instruction)

        for action in record.actions:
            total_actions += 1
            if action.action_type:
                action_types[action.action_type.lower()] += 1
            if action.cleaned_html:
                page_htmls.append(action.cleaned_html)

        if progress_interval and record_count % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = record_count / elapsed if elapsed > 0 else 0
            print(f"  [{record_count:,}] {rate:.1f} rec/s, "
                  f"{total_actions:,} actions, {len(page_htmls):,} pages")

        if debug and record_count <= 3:
            print(f"  [DEBUG] #{record_count}: domain={domain}, "
                  f"actions={len(record.actions)}")

    collect_time = time.time() - start_time
    print(f"\n数据收集完成: {record_count:,} records, {total_actions:,} actions, "
          f"{len(page_htmls):,} pages ({collect_time:.1f}s)")

    # ==================== 计算五个维度 ====================

    print(f"\n📊 [1/5] 页面结构多样性 (APTED + Vendi Score)...")
    page_div = compute_page_diversity(page_htmls)

    print(f"\n📊 [2/5] 动作模式多样性...")
    action_div = compute_action_diversity(
        action_types,
        action_mapping=action_mapping,
        skip_actions=skip_actions,
    )

    print(f"📊 [3/5] 域名多样性...")
    domain_div = compute_domain_diversity(domains)

    print(f"\n📊 [4/5] 任务语义多样性 (Sentence Embedding)...")
    model = SentenceTransformer(embedding_model)
    print(f"   编码 {len(instructions)} 条指令...")
    instruction_embeddings = model.encode(
        instructions, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    task_div = compute_task_diversity(instruction_embeddings)

    print(f"\n📊 [5/5] 表达模式多样性 (Self-BLEU)...")
    expression_div = compute_expression_diversity(instructions)

    elapsed = time.time() - start_time

    # ==================== 构建结果 ====================
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'dimensions': {
            'page_diversity': page_div,
            'action_diversity': action_div,
            'domain_diversity': domain_div,
            'task_diversity': task_div,
            'expression_diversity': expression_div,
        },
        'summary': {
            'total_records': record_count,
            'total_actions': total_actions,
            'total_pages': len(page_htmls),
            'avg_actions_per_record': total_actions / record_count if record_count > 0 else 0,
            'unique_domains': len(domains),
            'unique_action_types': len(action_types),
        },
    }

    # ==================== 输出 ====================
    _print_results(results, page_div, action_div, domain_div, task_div, expression_div, elapsed)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

    return results


def _print_results(results, page_div, action_div, domain_div, task_div, expression_div, elapsed):
    """打印结果摘要。"""
    s = results['summary']
    print()
    print("=" * 70)
    print(f"结果摘要  ({s['total_records']:,} records, "
          f"{s['total_actions']:,} actions, {s['total_pages']:,} pages, "
          f"{elapsed:.1f}s)")
    print("=" * 70)

    print(f"\n【1. 页面结构多样性】 APTED + Vendi Score")
    print(f"  ★ Vendi Score = {page_div['vendi_score']:.1f} "
          f"(等效独立页面模板数, 共 {page_div['n_pages']} 页)")
    print(f"    VS/N = {page_div['vendi_score_normalized']:.4f}  "
          f"平均相似度 = {page_div['similarity_mean']:.4f}")

    print(f"\n【2. 动作模式多样性】")
    print(f"  动作类型: {action_div['n_action_types']} 种  "
          f"熵 = {action_div['action_type_entropy']:.4f}  "
          f"覆盖率 = {action_div['action_type_coverage']:.0%}")

    print(f"\n【3. 域名多样性】")
    print(f"  唯一域名: {domain_div['n_unique_domains']} 个  "
          f"熵 = {domain_div['domain_entropy']:.4f}  "
          f"Gini = {domain_div['domain_gini']:.4f}")

    print(f"\n【4. 任务语义多样性】")
    print(f"  语义多样性 = {task_div['semantic_diversity']:.4f}  "
          f"(1 - 平均余弦相似度 {task_div['avg_pairwise_similarity']:.4f})")

    print(f"\n【5. 表达模式多样性】 Self-BLEU")
    print(f"  Self-BLEU = {expression_div['self_bleu']:.4f}  "
          f"表达多样性 = {expression_div['expression_diversity']:.4f}")
    print()
