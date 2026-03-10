#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity 多样性指标 — API Agent 数据集多维度多样性评估

从七个维度评估数据集的多样性：

    1. Embedding 语义多样性
       对 query / tools / api_calls 三个字段分别生成 embedding，
       用 Vendi Score（基于核矩阵特征值）或 KNN 平均距离衡量语义空间中的分散程度。
       数值越高说明样本之间语义差异越大。

    2. API 调用多样性
       从四个粒度检测调用模式是否单一：
       - 名称分布熵：所有 sample 调用的 API 名称分布是否均匀。
         如果大量 sample 调用同一个 API，熵低 → 名称多样性差。
       - 调用序列 bigram 熵：将每个 sample 的调用序列看成词序列，
         提取相邻两个调用组成的 bigram（如 search → get_detail），
         看这些 bigram 的分布是否丰富。大量 sample 走同一条链路 → 熵低。
       - 参数 key 组合熵：每次 API 调用传了哪些参数名（如 (city, date) vs (query, limit)），
         这些参数 key 的组合模式是否多样。所有调用都只传同样的参数集 → 熵低。
       - 调用序列编辑距离：对所有 sample 的调用序列做全量两两 Levenshtein 编辑距离计算。
         与序列熵互补——序列熵只看"有几种不同序列"，编辑距离能揭示看似不同的序列
         其实非常接近。例：80% 的序列是 [search, get_detail]，20% 是
         [search, get_detail, get_reviews]，序列熵认为"有 2 种"，但编辑距离仅 1。

    3. 表达模式多样性 (Self-BLEU)
       对所有 query 计算 Self-BLEU：每条 query 以其余 query 为参考计算 BLEU 分数后取均值。
       Self-BLEU 高 → query 之间措辞高度相似（模板化）；Self-BLEU 低 → 表达多样。
       对合成数据尤其有效，能检测 "Find me a...", "Search for..." 等模板化生成。

    4. 参数值多样性
       按参数名聚合所有 API 调用中实际传入的参数值，计算每个参数名下值的唯一率和熵。
       合成数据中参数值容易坍缩（永远是相同的城市、日期、ID），
       高唯一率 + 高熵 → 参数值丰富；低唯一率 + 低熵 → 值坍缩严重。

    5. 域名/类别多样性
       从 metadata 提取 category / domain 等字段，计算分布熵和 Gini 系数。
       熵高 + Gini 低 → 类别分布均匀；熵低 + Gini 高 → 数据集中在少数类别。
       ToolBench 有 49 个 category，xLAM 有分类信息，可直接利用。

    6. 工具组合多样性
       分析每个 sample 提供的工具集合（而非单个工具）：
       - 工具集合分布熵：有多少种不同的工具集合搭配。
       - 工具共现覆盖率：所有可能的工具两两共现中，实际出现了多少比例。
       覆盖率高 → 工具搭配丰富；覆盖率低 → 只有少数固定组合。

    7. 工具覆盖率
       每个 sample 定义了若干可用工具，agent 实际只调用了部分。
       覆盖率 = 实际调用工具数 / 可用工具数。
       覆盖率普遍低 → 数据太简单或存在大量无关工具填充。

使用方式:
    from diversity import compute_diversity
    from loaders import ToolBenchLoader

    loader = ToolBenchLoader('/path/to/toolbench.json')

    results = compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name='ToolBench',
        method='knn',  # 或 'vendi'
        field='query',
        embedding_cache_path='embeddings/toolbench_query.npy',
    )
"""

# 在任何 import 之前设置线程限制，防止 OpenBLAS 崩溃
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

import sys
import json
import math
import time
import random
import itertools
import numpy as np
from collections import Counter
from datetime import datetime
from typing import Optional, Iterator, Dict, Any, List, Tuple, Set
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import APIAgentSample


# =============================================================================
# 支持的 Embedding 模型
# =============================================================================

# 获取脚本所在目录，用于构建本地模型路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")

# 预定义的模型配置
EMBEDDING_MODELS = {
    # Sentence-Transformers 模型（轻量，CPU/GPU 都可用）
    "all-MiniLM-L6-v2": {
        "type": "sentence-transformers",
        "dim": 384,
        "description": "轻量快速，适合大规模数据",
        "local_path": os.path.join(_MODELS_DIR, "all-MiniLM-L6-v2"),
    },
    "all-mpnet-base-v2": {
        "type": "sentence-transformers",
        "dim": 768,
        "description": "效果更好，速度适中",
        "local_path": os.path.join(_MODELS_DIR, "all-mpnet-base-v2"),
    },
    # Qwen Embedding 模型（使用原生 transformers 加载，支持多 GPU）
    "Qwen/Qwen3-Embedding-8B": {
        "type": "transformers",
        "dim": 4096,
        "description": "效果最好，需要 GPU，使用原生 transformers 加载",
        "local_path": os.path.join(_MODELS_DIR, "Qwen3-Embedding-8B"),
    },
}


def _get_model_path(model_name: str) -> str:
    """
    获取模型路径：优先使用本地路径，如果不存在则使用原始名称（从 HuggingFace 下载）
    """
    if model_name in EMBEDDING_MODELS:
        local_path = EMBEDDING_MODELS[model_name].get("local_path")
        if local_path and os.path.exists(local_path):
            print(f"使用本地模型: {local_path}")
            return local_path
    # 如果传入的是路径且存在，直接使用
    if os.path.exists(model_name):
        print(f"使用指定路径: {model_name}")
        return model_name
    # 否则使用原始名称（会从 HuggingFace 下载）
    print(f"使用远程模型: {model_name}")
    return model_name


# =============================================================================
# Embedding 生成
# =============================================================================

def get_embedding_model_sbert(model_name: str):
    """
    加载 sentence-transformers 模型（用于小模型）
    
    Args:
        model_name: 模型名称或路径
    """
    import torch
    from sentence_transformers import SentenceTransformer
    
    model_path = _get_model_path(model_name)
    print(f"加载 Sentence-Transformers 模型: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = SentenceTransformer(model_path, device=device)
    
    return model


def get_embedding_model_transformers(model_name: str):
    """
    使用原生 transformers 加载大模型（支持多 GPU）
    
    Args:
        model_name: 模型名称或路径
    
    Returns:
        (model, tokenizer) 元组
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    model_path = _get_model_path(model_name)
    print(f"加载 Transformers 模型 (多 GPU): {model_path}")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"检测到 {num_gpus} 个 GPU")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    
    return model, tokenizer


def generate_embeddings_sbert(
    texts: List[str],
    model,
    batch_size: int = 64,
) -> np.ndarray:
    """
    使用 Sentence-Transformers 生成 embedding
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 归一化
    )
    return embeddings


def _last_token_pool(last_hidden_states, attention_mask):
    """
    从最后一个 token 提取 embedding（用于 Qwen3-Embedding）
    """
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_embeddings_transformers(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 8192,
) -> np.ndarray:
    """
    使用原生 transformers 生成 embedding（支持多 GPU 大模型）
    """
    import torch
    import torch.nn.functional as F
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成 embedding (transformers)"):
        batch_texts = texts[i:i + batch_size]
        
        # 分词
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 移动到模型所在设备
        # 注意：device_map="auto" 时，模型可能分布在多个设备上
        # 输入应该放在第一个设备上
        input_device = next(model.parameters()).device
        batch_dict = {k: v.to(input_device) for k, v in batch_dict.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        # 从最后一个 token 提取 embedding
        embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 转为 numpy
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def _extract_field_text(sample: APIAgentSample, field: str) -> str:
    """从 APIAgentSample 中提取指定字段的文本表示"""
    if field == "query":
        return sample.query or ""
    elif field == "tools":
        parts = []
        for tool in sample.tools:
            parts.append(f"{tool.name}: {tool.description or ''}")
        return "\n".join(parts)
    elif field == "api_calls":
        parts = []
        for call in sample.api_calls:
            name = call.name or ""
            if name.lower() in ("finish", "finalaction"):
                continue
            args_str = ", ".join(f"{k}={v}" for k, v in (call.arguments or {}).items())
            parts.append(f"{name}({args_str})")
        return "\n".join(parts) if parts else ""
    elif field == "both":
        query = sample.query or ""
        tool_parts = []
        for tool in sample.tools:
            tool_parts.append(f"{tool.name}: {tool.description or ''}")
        tools = "\n".join(tool_parts)
        return f"Query: {query}\n\nTools: {tools}"
    elif field == "all":
        query = sample.query or ""
        tool_parts = []
        for tool in sample.tools:
            tool_parts.append(f"{tool.name}: {tool.description or ''}")
        tools = "\n".join(tool_parts)
        call_parts = []
        for call in sample.api_calls:
            name = call.name or ""
            if name.lower() in ("finish", "finalaction"):
                continue
            args_str = ", ".join(f"{k}={v}" for k, v in (call.arguments or {}).items())
            call_parts.append(f"{name}({args_str})")
        calls = "\n".join(call_parts)
        return f"Query: {query}\n\nTools: {tools}\n\nAPI Calls: {calls}"
    else:
        raise ValueError(f"Unknown field: {field}. Use 'query', 'tools', 'api_calls', 'both', or 'all'.")


def generate_embeddings(
    data_iterator: Iterator[APIAgentSample],
    field: str = "query",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    max_text_length: int = 8192,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    生成 embedding
    
    Args:
        data_iterator: APIAgentSample 迭代器
        field: 要提取的字段 ('query', 'tools', 'api_calls', 'both', 'all')
        model_name: 模型名称，支持:
            - sentence-transformers: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
            - transformers: 'Qwen/Qwen3-Embedding-8B'
        batch_size: 批大小
        max_samples: 最大样本数
        max_text_length: 最大文本长度（截断）
        cache_path: embedding 缓存路径，如果存在则直接加载
    
    Returns:
        numpy array of shape (n_samples, embedding_dim)
    """
    # 检查缓存
    if cache_path and os.path.exists(cache_path):
        print(f"从缓存加载 embedding: {cache_path}")
        embeddings = np.load(cache_path)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings
    
    # 确定模型类型
    if model_name in EMBEDDING_MODELS:
        model_type = EMBEDDING_MODELS[model_name]["type"]
    elif model_name.startswith("Qwen/") or "Embedding" in model_name:
        model_type = "transformers"  # 大模型默认使用 transformers
    else:
        model_type = "sentence-transformers"
    
    print(f"模型类型: {model_type}")
    
    # 收集文本
    print(f"收集文本，字段: {field}")
    texts = []
    
    for sample in tqdm(data_iterator, desc="收集文本"):
        if max_samples and len(texts) >= max_samples:
            break
        
        # 提取文本
        text = _extract_field_text(sample, field)
        
        # 截断过长文本
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        texts.append(text)
    
    print(f"文本数量: {len(texts)}")
    
    # 根据模型类型生成 embedding
    if model_type == "transformers":
        # 大模型使用原生 transformers（支持多 GPU）
        print(f"使用原生 transformers + 多 GPU")
        model, tokenizer = get_embedding_model_transformers(model_name)
        embeddings = generate_embeddings_transformers(texts, model, tokenizer, batch_size=batch_size)
    else:
        # 小模型使用 sentence-transformers
        model = get_embedding_model_sbert(model_name)
        embeddings = generate_embeddings_sbert(texts, model, batch_size=batch_size)
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # 保存缓存
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        np.save(cache_path, embeddings)
        print(f"Embedding 已保存到: {cache_path}")
    
    return embeddings


# =============================================================================
# Vendi Score 计算
# =============================================================================

def _compute_vendi_worker(args: tuple) -> tuple:
    """
    多进程 worker 函数，在指定 GPU 上计算 Vendi Score
    
    Args:
        args: (batch_idx, embeddings, similarity_metric, gpu_id)
    
    Returns:
        (batch_idx, vendi_score)
    """
    batch_idx, embeddings, similarity_metric, gpu_id = args
    
    import torch
    
    # 设置当前进程使用的 GPU
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    n = embeddings.shape[0]
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    
    # 计算相似度矩阵（核矩阵）
    if similarity_metric == "cosine":
        K = embeddings_tensor @ embeddings_tensor.T
    elif similarity_metric == "dot_product":
        K = embeddings_tensor @ embeddings_tensor.T
    else:
        raise ValueError(f"Unknown similarity_metric: {similarity_metric}")
    
    # 归一化
    K = K / n
    
    # 计算特征值
    eigenvalues = torch.linalg.eigvalsh(K)
    
    # 只保留正特征值
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # 计算熵
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues)).item()
    
    # Vendi Score = exp(entropy)
    vendi_score = np.exp(entropy)
    
    # 清理显存
    del embeddings_tensor, K, eigenvalues
    torch.cuda.empty_cache()
    
    return (batch_idx, vendi_score)


def _compute_single_vendi_score(
    embeddings_tensor,
    similarity_metric: str = "cosine",
) -> float:
    """
    计算单个 batch 的 Vendi Score（内部函数）
    
    Args:
        embeddings_tensor: PyTorch tensor, shape (n_samples, embedding_dim)
        similarity_metric: 相似度度量
    
    Returns:
        Vendi Score (float)
    """
    import torch
    
    n = embeddings_tensor.shape[0]
    
    # 计算相似度矩阵（核矩阵）
    if similarity_metric == "cosine":
        K = embeddings_tensor @ embeddings_tensor.T
    elif similarity_metric == "dot_product":
        K = embeddings_tensor @ embeddings_tensor.T
    else:
        raise ValueError(f"Unknown similarity_metric: {similarity_metric}")
    
    # 归一化
    K = K / n
    
    # 计算特征值
    eigenvalues = torch.linalg.eigvalsh(K)
    
    # 只保留正特征值
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # 计算熵
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues)).item()
    
    # Vendi Score = exp(entropy)
    return np.exp(entropy)


def compute_vendi_score(
    embeddings: np.ndarray,
    similarity_metric: str = "cosine",
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算 Vendi Score（支持 GPU 加速、分 batch 计算和多 GPU 并行）
    
    Vendi Score = exp(entropy of eigenvalues of similarity matrix)
    数值越高，多样性越大
    
    当数据量大时，可以分 batch 计算每个 batch 的 Vendi Score，最后取平均。
    支持多 GPU 并行计算，每个 GPU 处理一个 batch。
    
    Args:
        embeddings: shape (n_samples, embedding_dim)
        similarity_metric: 相似度度量 ('cosine', 'dot_product')
        sample_size: 采样大小，None 表示全量
        random_seed: 随机种子
        use_gpu: 是否使用 GPU 加速
        batch_size: 分 batch 计算时每个 batch 的大小，None 表示不分 batch
                    建议值：10000-50000（根据 GPU 显存调整）
        num_gpus: 使用的 GPU 数量，None 表示自动检测
    
    Returns:
        包含 Vendi Score 的字典
    """
    import torch
    import torch.multiprocessing as mp
    
    n_total = embeddings.shape[0]
    
    # 采样
    if sample_size and sample_size < n_total:
        print(f"采样 {sample_size} / {n_total} 样本")
        random.seed(random_seed)
        indices = random.sample(range(n_total), sample_size)
        embeddings = embeddings[indices]
        is_sampled = True
    else:
        sample_size = n_total
        is_sampled = False
    
    n = embeddings.shape[0]
    print(f"计算 Vendi Score，样本数: {n}")
    
    # 检测可用 GPU
    device = "cpu"
    available_gpus = 0
    if use_gpu and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        device = "cuda"
        print(f"检测到 {available_gpus} 个 GPU")
        for i in range(available_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("使用 CPU 计算")
    
    # 确定使用的 GPU 数量
    if num_gpus is None:
        num_gpus = available_gpus if available_gpus > 0 else 1
    else:
        num_gpus = min(num_gpus, available_gpus) if available_gpus > 0 else 1
    
    # 如果需要分 batch 计算
    if batch_size and batch_size < n:
        num_batches = (n + batch_size - 1) // batch_size
        print(f"分 {num_batches} 个 batch 计算（每个 batch 最多 {batch_size} 样本）")
        
        # 随机打乱索引，确保每个 batch 的样本是随机的
        random.seed(random_seed)
        indices = list(range(n))
        random.shuffle(indices)
        
        # 准备所有 batch 的数据：(batch_idx, embeddings, similarity_metric, gpu_id)
        batch_args = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            batch_indices = indices[start_idx:end_idx]
            batch_embeddings = embeddings[batch_indices]
            gpu_id = i % num_gpus
            batch_args.append((i, batch_embeddings, similarity_metric, gpu_id))
            print(f"  Batch {i+1}/{num_batches} -> GPU {gpu_id} ({end_idx - start_idx} 样本)")
        
        batch_scores = [None] * num_batches
        
        # 多 GPU 并行计算（使用多进程）
        if num_gpus > 1 and num_batches > 1:
            print(f"使用 {min(num_gpus, num_batches)} 个 GPU 多进程并行计算...")
            
            # 使用 spawn 方式创建进程，避免 CUDA 初始化问题
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 已经设置过了
            
            with mp.Pool(processes=min(num_gpus, num_batches)) as pool:
                results = pool.map(_compute_vendi_worker, batch_args)
            
            for batch_idx, score in results:
                batch_scores[batch_idx] = score
                print(f"  Batch {batch_idx+1} 完成: Vendi Score = {score:.4f}")
        else:
            # 单 GPU 串行计算
            for args in batch_args:
                batch_idx, batch_emb, sim_metric, gpu_id = args
                print(f"  计算 Batch {batch_idx+1}/{num_batches}...")
                embeddings_tensor = torch.tensor(batch_emb, dtype=torch.float32, device=device)
                batch_score = _compute_single_vendi_score(embeddings_tensor, similarity_metric)
                batch_scores[batch_idx] = batch_score
                print(f"    Vendi Score: {batch_score:.4f}")
                
                del embeddings_tensor
                if device == "cuda":
                    torch.cuda.empty_cache()
        
        # 取平均
        vendi_score = np.mean(batch_scores)
        entropy = np.log(vendi_score)  # 反推熵
        
        print(f"\n各 batch Vendi Score: {[f'{s:.4f}' for s in batch_scores]}")
        print(f"平均 Vendi Score: {vendi_score:.4f}")
        
        return {
            "vendi_score": float(vendi_score),
            "entropy": float(entropy),
            "n_samples": n,
            "n_total": n_total,
            "is_sampled": is_sampled,
            "similarity_metric": similarity_metric,
            "device": device,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "num_gpus": num_gpus,
            "batch_scores": [float(s) for s in batch_scores],
        }
    
    # 不分 batch，整体计算
    print("计算相似度矩阵...")
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    
    vendi_score = _compute_single_vendi_score(embeddings_tensor, similarity_metric)
    entropy = np.log(vendi_score)
    
    # 清理显存
    del embeddings_tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"Vendi Score: {vendi_score:.4f}")
    
    return {
        "vendi_score": float(vendi_score),
        "entropy": float(entropy),
        "n_samples": n,
        "n_total": n_total,
        "is_sampled": is_sampled,
        "similarity_metric": similarity_metric,
        "device": device,
    }


# =============================================================================
# KNN 平均距离计算
# =============================================================================

def compute_knn_diversity(
    embeddings: np.ndarray,
    k: int = 10,
    distance_metric: str = "cosine",
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    计算 KNN 平均距离作为多样性指标
    
    平均距离越大，多样性越高
    
    Args:
        embeddings: shape (n_samples, embedding_dim)
        k: K 近邻的 K 值
        distance_metric: 距离度量 ('cosine', 'euclidean')
        sample_size: 采样大小，None 表示全量
        random_seed: 随机种子
    
    Returns:
        包含 KNN 多样性分数的字典
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_total = embeddings.shape[0]
    
    # 采样
    if sample_size and sample_size < n_total:
        print(f"采样 {sample_size} / {n_total} 样本")
        random.seed(random_seed)
        indices = random.sample(range(n_total), sample_size)
        embeddings = embeddings[indices]
        is_sampled = True
    else:
        sample_size = n_total
        is_sampled = False
    
    n = embeddings.shape[0]
    print(f"计算 KNN 多样性，样本数: {n}, K={k}")
    
    # 调整 k 值
    if k >= n:
        k = n - 1
        print(f"K 调整为 {k}")
    
    # sklearn 单线程版本（避免 OpenBLAS 多线程问题）
    print("构建 KNN 模型（单线程）...")
    nn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 因为包含自己
        metric=distance_metric,
        algorithm='auto',
        n_jobs=1,  # 单线程，避免 OpenBLAS 崩溃
    )
    nn.fit(embeddings)
    
    print("查询 K 近邻...")
    distances, indices = nn.kneighbors(embeddings)
    
    # 排除自己（第一个近邻是自己，距离为 0）
    k_distances = distances[:, 1:k+1]
    
    # 计算统计量
    mean_distance = float(np.mean(k_distances))
    std_distance = float(np.std(k_distances))
    median_distance = float(np.median(k_distances))
    
    print(f"KNN 平均距离: {mean_distance:.6f}")
    print(f"KNN 距离标准差: {std_distance:.6f}")
    
    return {
        "knn_mean_distance": mean_distance,
        "knn_std_distance": std_distance,
        "knn_median_distance": median_distance,
        "k": k,
        "n_samples": n,
        "n_total": n_total,
        "is_sampled": is_sampled,
        "distance_metric": distance_metric,
    }


# =============================================================================
# 辅助函数: 信息熵 / Gini
# =============================================================================

def _compute_entropy(counter: Counter) -> Tuple[float, float, int]:
    """计算归一化熵。返回 (normalized_entropy, raw_entropy, n_types)。"""
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
    """计算 Gini 系数，衡量分布均匀度。0=完全均匀，1=完全集中。"""
    if not values or len(values) < 2:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """列表级别的 Levenshtein 编辑距离"""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


# =============================================================================
# 调用序列多样性（编辑距离 + 结构分析）
# =============================================================================

def compute_call_sequence_diversity(
    call_sequences: List[List[str]],
) -> Dict[str, Any]:
    """
    计算 API 调用序列间的结构多样性。

    与序列熵互补：序列熵只看"有多少种不同序列"，
    编辑距离能揭示看似不同的序列其实非常接近。

    例：如果 80% 的 sample 调用链是 [search, get_detail, finish]，
    另外 20% 是 [search, get_detail, get_reviews, finish]，
    序列熵会认为"有 2 种不同序列"，但编辑距离会揭示
    这两种序列其实非常接近（编辑距离仅 1），实际多样性很低。

    子指标：
    1. 唯一序列比例
    2. 序列长度变异系数
    3. 序列间平均编辑距离（全量 O(N^2)）
    4. 序列间平均归一化编辑距离
    """
    n = len(call_sequences)
    if n == 0:
        return {
            'n_sequences': 0,
            'unique_sequence_ratio': 0.0,
            'length_cv': 0.0,
            'length_mean': 0.0,
            'length_std': 0.0,
            'avg_edit_distance': 0.0,
            'avg_normalized_edit_distance': 0.0,
        }

    # --- 唯一序列比例 ---
    seq_strs = ["|".join(seq) for seq in call_sequences]
    n_unique = len(set(seq_strs))
    unique_ratio = n_unique / n

    # --- 序列长度统计 ---
    lengths = [len(seq) for seq in call_sequences]
    length_mean = float(np.mean(lengths))
    length_std = float(np.std(lengths))
    length_cv = length_std / length_mean if length_mean > 0 else 0.0

    # --- 序列间编辑距离（全量） ---
    avg_edit_dist = 0.0
    avg_norm_edit_dist = 0.0
    if n >= 2:
        edit_dists = []
        norm_edit_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = _levenshtein(call_sequences[i], call_sequences[j])
                edit_dists.append(d)
                max_len = max(len(call_sequences[i]), len(call_sequences[j]), 1)
                norm_edit_dists.append(d / max_len)
        avg_edit_dist = float(np.mean(edit_dists))
        avg_norm_edit_dist = float(np.mean(norm_edit_dists))

    return {
        'n_sequences': n,
        'n_unique_sequences': n_unique,
        'unique_sequence_ratio': unique_ratio,
        'length_mean': length_mean,
        'length_std': length_std,
        'length_cv': length_cv,
        'avg_edit_distance': avg_edit_dist,
        'avg_normalized_edit_distance': avg_norm_edit_dist,
    }


# =============================================================================
# Self-BLEU 表达模式多样性
# =============================================================================

def compute_expression_diversity(
    texts: List[str],
    batch_size: int = 200,
) -> Dict[str, Any]:
    """
    计算 Self-BLEU 表达模式多样性。
    Self-BLEU 高 → 文本模板化严重；Self-BLEU 低 → 表达多样。

    对合成数据特别有效：能检测 "Find me a...", "Search for..." 等模板化 query。
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    n = len(texts)
    if n < 2:
        return {'self_bleu': 1.0, 'expression_diversity': 0.0, 'n_texts': n}

    tokenized = [t.lower().split() for t in texts]
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
        'n_texts': n,
    }


# =============================================================================
# 参数值多样性
# =============================================================================

def compute_param_value_diversity(
    samples: List[APIAgentSample],
) -> Dict[str, Any]:
    """
    分析 API 调用中参数值的多样性。

    合成数据中参数值容易坍缩（永远是相同的城市、日期、ID），
    这里按参数名聚合值的唯一率和熵。
    """
    param_values: Dict[str, Counter] = {}
    n_calls = 0

    for sample in samples:
        for call in sample.api_calls:
            name = (call.name or "").strip()
            if not name or name.lower() in ("finish", "finalaction"):
                continue
            n_calls += 1
            if not isinstance(call.arguments, dict):
                continue
            for k, v in call.arguments.items():
                v_str = str(v)
                if k not in param_values:
                    param_values[k] = Counter()
                param_values[k][v_str] += 1

    if not param_values:
        return {
            'n_calls': n_calls,
            'n_param_keys': 0,
            'avg_value_unique_ratio': 0.0,
            'avg_value_entropy_normalized': 0.0,
            'per_param': {},
        }

    per_param = {}
    unique_ratios = []
    entropies = []

    for k, counter in sorted(param_values.items(), key=lambda x: -sum(x[1].values())):
        total = sum(counter.values())
        n_unique = len(counter)
        ratio = n_unique / total if total > 0 else 0.0
        ent_norm, _, _ = _compute_entropy(counter)

        unique_ratios.append(ratio)
        entropies.append(ent_norm)

        per_param[k] = {
            'n_unique_values': n_unique,
            'n_total': total,
            'unique_ratio': ratio,
            'entropy_normalized': ent_norm,
            'top_values': dict(counter.most_common(5)),
        }

    return {
        'n_calls': n_calls,
        'n_param_keys': len(param_values),
        'avg_value_unique_ratio': float(np.mean(unique_ratios)),
        'avg_value_entropy_normalized': float(np.mean(entropies)),
        'per_param': per_param,
    }


# =============================================================================
# 域名 / 类别多样性
# =============================================================================

def compute_category_diversity(
    samples: List[APIAgentSample],
) -> Dict[str, Any]:
    """
    分析数据集的 domain / category 分布多样性。

    从 metadata 中提取 category、domain、source 等字段。
    """
    category_counter = Counter()
    domain_counter = Counter()

    for sample in samples:
        meta = sample.metadata or {}
        cat = meta.get('category') or meta.get('api_category') or meta.get('type', '')
        if cat:
            category_counter[str(cat)] += 1
        domain = meta.get('domain') or meta.get('source') or meta.get('api_provider', '')
        if domain:
            domain_counter[str(domain)] += 1

    results: Dict[str, Any] = {}

    if category_counter:
        cat_ent, _, n_cats = _compute_entropy(category_counter)
        cat_gini = _compute_gini(list(category_counter.values()))
        results['category'] = {
            'n_unique': n_cats,
            'entropy_normalized': cat_ent,
            'gini': cat_gini,
            'distribution': dict(category_counter.most_common(30)),
        }
    else:
        results['category'] = None

    if domain_counter:
        dom_ent, _, n_doms = _compute_entropy(domain_counter)
        dom_gini = _compute_gini(list(domain_counter.values()))
        results['domain'] = {
            'n_unique': n_doms,
            'entropy_normalized': dom_ent,
            'gini': dom_gini,
            'distribution': dict(domain_counter.most_common(30)),
        }
    else:
        results['domain'] = None

    return results


# =============================================================================
# 工具组合多样性
# =============================================================================

def compute_tool_combination_diversity(
    samples: List[APIAgentSample],
) -> Dict[str, Any]:
    """
    分析工具组合（共现）模式的多样性。

    - 每个 sample 提供了哪些工具？这些工具集合的分布如何？
    - 工具之间的共现关系是否丰富？
    """
    toolset_counter = Counter()
    cooccurrence: Counter = Counter()
    n_samples_with_tools = 0

    for sample in samples:
        tool_names = sorted(set(t.name for t in sample.tools if t.name))
        if not tool_names:
            continue
        n_samples_with_tools += 1
        toolset_counter[tuple(tool_names)] += 1
        for pair in itertools.combinations(tool_names, 2):
            cooccurrence[pair] += 1

    if n_samples_with_tools == 0:
        return {
            'n_samples_with_tools': 0,
            'n_unique_toolsets': 0,
            'toolset_unique_ratio': 0.0,
            'toolset_entropy_normalized': 0.0,
            'n_unique_cooccurrences': 0,
            'top_toolsets': {},
            'top_cooccurrences': {},
        }

    toolset_ent, _, n_unique_toolsets = _compute_entropy(toolset_counter)

    all_tools = set()
    for ts in toolset_counter.keys():
        all_tools.update(ts)
    max_pairs = len(all_tools) * (len(all_tools) - 1) // 2
    cooccurrence_coverage = len(cooccurrence) / max_pairs if max_pairs > 0 else 0.0

    return {
        'n_samples_with_tools': n_samples_with_tools,
        'n_unique_toolsets': n_unique_toolsets,
        'toolset_unique_ratio': n_unique_toolsets / n_samples_with_tools,
        'toolset_entropy_normalized': toolset_ent,
        'n_unique_cooccurrences': len(cooccurrence),
        'max_possible_cooccurrences': max_pairs,
        'cooccurrence_coverage': cooccurrence_coverage,
        'top_toolsets': {str(k): v for k, v in toolset_counter.most_common(10)},
        'top_cooccurrences': {str(k): v for k, v in cooccurrence.most_common(10)},
    }


# =============================================================================
# 工具覆盖率
# =============================================================================

def compute_tool_coverage(
    samples: List[APIAgentSample],
) -> Dict[str, Any]:
    """
    分析每个 sample 中实际调用的工具占可用工具的比例。

    覆盖率低 → agent 只用了一小部分工具（可能是数据太简单或工具冗余）。
    """
    coverages = []
    n_skipped = 0

    for sample in samples:
        available = set(t.name for t in sample.tools if t.name)
        if not available:
            n_skipped += 1
            continue
        called = set()
        for call in sample.api_calls:
            name = (call.name or "").strip()
            if name and name.lower() not in ("finish", "finalaction"):
                called.add(name)
        coverage = len(called & available) / len(available)
        coverages.append(coverage)

    if not coverages:
        return {
            'n_samples': 0,
            'mean_coverage': 0.0,
            'std_coverage': 0.0,
            'median_coverage': 0.0,
        }

    arr = np.asarray(coverages)
    hist_bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.01]
    hist_counts, _ = np.histogram(arr, bins=hist_bins)
    hist_labels = ['0-10%', '10-20%', '20-30%', '30-50%', '50-70%', '70-100%', '100%']

    return {
        'n_samples': len(coverages),
        'n_skipped': n_skipped,
        'mean_coverage': float(np.mean(arr)),
        'std_coverage': float(np.std(arr)),
        'median_coverage': float(np.median(arr)),
        'min_coverage': float(np.min(arr)),
        'max_coverage': float(np.max(arr)),
        'distribution': {label: int(cnt) for label, cnt in zip(hist_labels, hist_counts)},
    }


# =============================================================================
# 主函数
# =============================================================================

def compute_diversity(
    data_iterator: Iterator[APIAgentSample],
    dataset_name: str = "Unknown",
    method: str = "knn",
    field: str = "query",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_cache_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    # KNN 特有参数
    k: int = 10,
    distance_metric: str = "cosine",
    # Vendi 特有参数
    similarity_metric: str = "cosine",
    vendi_batch_size: Optional[int] = None,
    num_gpus: Optional[int] = None,
    # Embedding 生成参数
    embedding_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算数据集的多样性指标
    
    Args:
        data_iterator: 数据迭代器
        dataset_name: 数据集名称
        method: 多样性计算方法 ('knn' 或 'vendi')
        field: 主字段，决定 primary score ('query', 'tools', 'api_calls', 'both', 'all')
              注意：所有三个基础字段 (query, tools, api_calls) 都会自动计算
        embedding_model: Embedding 模型名称，支持:
            - sentence-transformers: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
            - transformers: 'Qwen/Qwen3-Embedding-8B'
        embedding_cache_path: embedding 缓存路径
        sample_size: 采样大小（用于 Vendi Score 或加速 KNN）
        output_file: 结果保存路径
        max_samples: 最大样本数（用于测试）
        k: KNN 的 K 值
        distance_metric: KNN 距离度量
        similarity_metric: Vendi Score 相似度度量
        vendi_batch_size: Vendi Score 分 batch 计算的大小
        num_gpus: Vendi Score 多 GPU 并行数量
        embedding_batch_size: Embedding 生成时的 batch 大小
        
    Returns:
        包含多样性分数的字典
    """
    print("=" * 70)
    print("Diversity 多样性评估")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"方法: {method}")
    print(f"字段: {field}")
    print(f"模型: {embedding_model}")
    print(f"采样大小: {sample_size or '全量'}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Step 0: 收集样本（供 embedding 和 API 调用多样性复用）
    print("-" * 50)
    print("Step 0: 收集样本")
    print("-" * 50)
    samples: List[APIAgentSample] = []
    for s in tqdm(data_iterator, desc="收集样本"):
        if max_samples is not None and len(samples) >= max_samples:
            break
        samples.append(s)
    print(f"共收集 {len(samples):,} 个样本")
    print()
    
    # Step 1 & 2: 对三个字段分别生成 embedding 并计算多样性
    emb_batch_size = embedding_batch_size if embedding_batch_size is not None else 8
    FIELDS = ["query", "tools", "api_calls"]
    
    per_field_results = {}
    primary_diversity_score = None
    primary_diversity_result = None
    
    for f in FIELDS:
        print("-" * 50)
        print(f"Step 1-2: 字段 '{f}' — 生成 Embedding + 计算多样性 ({method})")
        print("-" * 50)
        
        f_cache = None
        if embedding_cache_path:
            base, ext = os.path.splitext(embedding_cache_path)
            f_cache = f"{base}_{f}{ext}"
        
        f_embeddings = generate_embeddings(
            data_iterator=iter(samples),
            field=f,
            model_name=embedding_model,
            batch_size=emb_batch_size,
            max_samples=max_samples,
            cache_path=f_cache,
        )
        
        if method == "knn":
            f_diversity = compute_knn_diversity(
                embeddings=f_embeddings, k=k,
                distance_metric=distance_metric, sample_size=sample_size,
            )
            f_score = f_diversity["knn_mean_distance"]
        elif method == "vendi":
            f_diversity = compute_vendi_score(
                embeddings=f_embeddings, similarity_metric=similarity_metric,
                sample_size=sample_size, batch_size=vendi_batch_size, num_gpus=num_gpus,
            )
            f_score = f_diversity["vendi_score"]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'knn' or 'vendi'.")
        
        per_field_results[f] = {"score": f_score, **f_diversity}
        print(f"  {f} 多样性分数: {f_score:.6f}")
        print()
        
        if f == field:
            primary_diversity_score = f_score
            primary_diversity_result = f_diversity
    
    diversity_score = primary_diversity_score
    diversity_result = primary_diversity_result
    
    embedding_time = time.time() - start_time
    print(f"Embedding + 多样性总耗时: {embedding_time:.1f} 秒")
    print()
    
    # Step 3: API 调用多样性（轻量统计，复用 samples）
    print("-" * 50)
    print("Step 3: 计算 API 调用多样性")
    print("-" * 50)
    
    api_name_counter = Counter()
    sequence_counter = Counter()
    bigram_counter = Counter()
    param_key_combo_counter = Counter()
    n_calls_total = 0
    n_samples_with_calls = 0
    calls_per_sample: List[int] = []

    for sample in samples:
        names = []
        for call in sample.api_calls:
            name = (call.name or "").strip()
            if not name or name.lower() in ("finish", "finalaction"):
                continue
            names.append(name)
            api_name_counter[name] += 1
            n_calls_total += 1
            if isinstance(call.arguments, dict) and call.arguments:
                param_key_combo_counter[tuple(sorted(call.arguments.keys()))] += 1

        calls_per_sample.append(len(names))
        if names:
            n_samples_with_calls += 1
            sequence_counter[tuple(names)] += 1
            for i in range(len(names) - 1):
                bigram_counter[(names[i], names[i + 1])] += 1

    name_ent_norm, name_ent_raw, n_unique_apis = _compute_entropy(api_name_counter)
    name_gini = _compute_gini(list(api_name_counter.values()))
    seq_ent_norm, _, n_unique_seqs = _compute_entropy(sequence_counter)
    bigram_ent_norm, _, n_unique_bigrams = _compute_entropy(bigram_counter)
    param_ent_norm, _, n_unique_combos = _compute_entropy(param_key_combo_counter)

    calls_arr = np.asarray(calls_per_sample, dtype=float)
    api_call_diversity = {
        "calls_per_sample": {
            "mean": float(np.mean(calls_arr)) if len(calls_arr) > 0 else 0.0,
            "std": float(np.std(calls_arr)) if len(calls_arr) > 0 else 0.0,
            "median": float(np.median(calls_arr)) if len(calls_arr) > 0 else 0.0,
            "max": float(np.max(calls_arr)) if len(calls_arr) > 0 else 0.0,
        },
        "api_name_diversity": {
            "n_unique_apis": n_unique_apis,
            "n_total_calls": n_calls_total,
            "entropy_normalized": name_ent_norm,
            "gini": name_gini,
            "top_apis": dict(api_name_counter.most_common(20)),
        },
        "sequence_diversity": {
            "n_unique_sequences": n_unique_seqs,
            "n_samples_with_calls": n_samples_with_calls,
            "sequence_unique_ratio": n_unique_seqs / n_samples_with_calls if n_samples_with_calls > 0 else 0.0,
            "sequence_entropy_normalized": seq_ent_norm,
            "bigram_entropy_normalized": bigram_ent_norm,
            "n_unique_bigrams": n_unique_bigrams,
            "top_sequences": {str(k): v for k, v in sequence_counter.most_common(10)},
        },
        "param_combo_diversity": {
            "n_unique_param_combos": n_unique_combos,
            "param_combo_unique_ratio": n_unique_combos / n_calls_total if n_calls_total > 0 else 0.0,
            "param_combo_entropy_normalized": param_ent_norm,
            "top_param_combos": {str(k): v for k, v in param_key_combo_counter.most_common(10)},
        },
    }

    print(f"  唯一 API 数: {n_unique_apis}, 名称熵(归一化): {name_ent_norm:.4f}, Gini: {name_gini:.4f}")
    print(f"  唯一序列数: {n_unique_seqs}, 序列熵(归一化): {seq_ent_norm:.4f}")
    print(f"  唯一参数组合: {n_unique_combos}, 参数熵(归一化): {param_ent_norm:.4f}")
    print()

    # Step 3.5: 调用序列编辑距离分析
    print("-" * 50)
    print("Step 3.5: 计算调用序列编辑距离多样性")
    print("-" * 50)
    all_call_sequences = []
    for sample in samples:
        names = []
        for call in sample.api_calls:
            name = (call.name or "").strip()
            if name and name.lower() not in ("finish", "finalaction"):
                names.append(name)
        if names:
            all_call_sequences.append(names)
    call_seq_div = compute_call_sequence_diversity(all_call_sequences)
    api_call_diversity["call_sequence_edit_distance"] = call_seq_div
    print(f"  唯一序列比例: {call_seq_div['unique_sequence_ratio']:.4f}")
    print(f"  序列长度 CV: {call_seq_div['length_cv']:.4f}")
    print(f"  平均归一化编辑距离: {call_seq_div['avg_normalized_edit_distance']:.4f}")
    print()

    # Step 4: 表达模式多样性 (Self-BLEU on queries)
    print("-" * 50)
    print("Step 4: 计算 Query 表达模式多样性 (Self-BLEU)")
    print("-" * 50)
    queries = [s.query for s in samples if s.query]
    expression_div = compute_expression_diversity(queries)
    print(f"  Self-BLEU: {expression_div['self_bleu']:.4f}, "
          f"表达多样性: {expression_div['expression_diversity']:.4f}")
    print()

    # Step 5: 参数值多样性
    print("-" * 50)
    print("Step 5: 计算参数值多样性")
    print("-" * 50)
    param_value_div = compute_param_value_diversity(samples)
    print(f"  参数 key 数: {param_value_div['n_param_keys']}")
    print(f"  平均值唯一率: {param_value_div['avg_value_unique_ratio']:.4f}")
    print(f"  平均值熵(归一化): {param_value_div['avg_value_entropy_normalized']:.4f}")
    print()

    # Step 6: 域名/类别多样性
    print("-" * 50)
    print("Step 6: 计算域名/类别多样性")
    print("-" * 50)
    category_div = compute_category_diversity(samples)
    if category_div.get('category'):
        c = category_div['category']
        print(f"  类别数: {c['n_unique']}, 熵(归一化): {c['entropy_normalized']:.4f}, "
              f"Gini: {c['gini']:.4f}")
    else:
        print("  (metadata 中无 category 字段)")
    if category_div.get('domain'):
        d = category_div['domain']
        print(f"  域名数: {d['n_unique']}, 熵(归一化): {d['entropy_normalized']:.4f}, "
              f"Gini: {d['gini']:.4f}")
    else:
        print("  (metadata 中无 domain 字段)")
    print()

    # Step 7: 工具组合多样性
    print("-" * 50)
    print("Step 7: 计算工具组合多样性")
    print("-" * 50)
    tool_combo_div = compute_tool_combination_diversity(samples)
    print(f"  唯一工具集合数: {tool_combo_div['n_unique_toolsets']}")
    print(f"  工具集合熵(归一化): {tool_combo_div['toolset_entropy_normalized']:.4f}")
    print(f"  共现覆盖率: {tool_combo_div.get('cooccurrence_coverage', 0):.4f}")
    print()

    # Step 8: 工具覆盖率
    print("-" * 50)
    print("Step 8: 计算工具覆盖率")
    print("-" * 50)
    tool_cov = compute_tool_coverage(samples)
    print(f"  平均覆盖率: {tool_cov['mean_coverage']:.4f}")
    print(f"  中位覆盖率: {tool_cov['median_coverage']:.4f}")
    print()

    total_time = time.time() - start_time

    # 汇总结果
    results = {
        "dataset": dataset_name,
        "method": method,
        "primary_field": field,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "n_samples": len(samples),
        "embedding_diversity_score": diversity_score,
        "per_field_diversity": per_field_results,
        **diversity_result,
        "api_call_diversity": api_call_diversity,
        "expression_diversity": expression_div,
        "param_value_diversity": param_value_div,
        "category_diversity": category_div,
        "tool_combination_diversity": tool_combo_div,
        "tool_coverage": tool_cov,
    }

    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！总耗时 {total_time:.1f} 秒")
    print("=" * 70)
    print()
    print(f"数据集: {dataset_name} | 样本数: {len(samples):,} | 方法: {method}")
    print()
    print("【1. Embedding 语义多样性（按字段）】")
    for f in FIELDS:
        print(f"  {f:12s}: {per_field_results[f]['score']:.6f}")
    print()
    print("【2. API 调用多样性】")
    print(f"  API 名称熵(归一化):   {name_ent_norm:.6f}  (唯一 {n_unique_apis} 个)")
    print(f"  调用序列熵(归一化):   {seq_ent_norm:.6f}  (唯一 {n_unique_seqs} 个)")
    print(f"  Bigram 熵(归一化):    {bigram_ent_norm:.6f}  (唯一 {n_unique_bigrams} 个)")
    print(f"  参数 key 组合熵:      {param_ent_norm:.6f}  (唯一 {n_unique_combos} 个)")
    print(f"  序列唯一率:           {call_seq_div['unique_sequence_ratio']:.6f}")
    print(f"  序列长度 CV:          {call_seq_div['length_cv']:.6f}")
    print(f"  平均归一化编辑距离:   {call_seq_div['avg_normalized_edit_distance']:.6f}")
    print()
    print("【3. 表达模式多样性 (Self-BLEU)】")
    print(f"  Self-BLEU:            {expression_div['self_bleu']:.6f}")
    print(f"  表达多样性:           {expression_div['expression_diversity']:.6f}")
    print()
    print("【4. 参数值多样性】")
    print(f"  平均值唯一率:         {param_value_div['avg_value_unique_ratio']:.6f}")
    print(f"  平均值熵(归一化):     {param_value_div['avg_value_entropy_normalized']:.6f}")
    print()
    print("【5. 域名/类别多样性】")
    if category_div.get('category'):
        c = category_div['category']
        print(f"  类别熵(归一化):       {c['entropy_normalized']:.6f}  ({c['n_unique']} 类)")
        print(f"  类别 Gini:            {c['gini']:.6f}")
    else:
        print("  (无 category 信息)")
    if category_div.get('domain'):
        d = category_div['domain']
        print(f"  域名熵(归一化):       {d['entropy_normalized']:.6f}  ({d['n_unique']} 域)")
        print(f"  域名 Gini:            {d['gini']:.6f}")
    else:
        print("  (无 domain 信息)")
    print()
    print("【6. 工具组合多样性】")
    print(f"  唯一工具集合:         {tool_combo_div['n_unique_toolsets']}")
    print(f"  工具集合唯一率:       {tool_combo_div['toolset_unique_ratio']:.6f}")
    print(f"  工具集合熵(归一化):   {tool_combo_div['toolset_entropy_normalized']:.6f}")
    print(f"  共现覆盖率:           {tool_combo_div.get('cooccurrence_coverage', 0):.6f}")
    print()
    print("【7. 工具覆盖率】")
    print(f"  平均覆盖率:           {tool_cov['mean_coverage']:.6f}")
    print(f"  中位覆盖率:           {tool_cov['median_coverage']:.6f}")
    print()

    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diversity 多样性评估")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["toolbench", "xlam"],
                        help="数据集名称")
    parser.add_argument("--method", type=str, default="knn",
                        choices=["knn", "vendi"],
                        help="多样性计算方法")
    parser.add_argument("--field", type=str, default="query",
                        choices=["query", "tools", "api_calls", "both", "all"],
                        help="主字段（所有字段都会计算，此项决定 primary score）")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding 模型名称: all-MiniLM-L6-v2, all-mpnet-base-v2, Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="采样大小（None 表示全量）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--k", type=int, default=10,
                        help="KNN 的 K 值")
    parser.add_argument("--embedding-batch-size", type=int, default=None,
                        help="Embedding 生成时的 batch 大小")
    parser.add_argument("--vendi-batch-size", type=int, default=None,
                        help="Vendi Score 分 batch 计算的大小")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Vendi Score 多 GPU 并行数量")
    
    args = parser.parse_args()
    
    # 加载数据集
    if args.dataset == "toolbench":
        from loaders import ToolBenchLoader
        loader = ToolBenchLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/toolbench_official/toolllama_G123_dfs_train.json'
        )
        dataset_name = "ToolBench"
        embedding_cache = f"embeddings/toolbench_{args.field}.npy"
        
    elif args.dataset == "xlam":
        from loaders import XLAMLoader
        loader = XLAMLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/xlam_60k.jsonl'
        )
        dataset_name = "xLAM-60k"
        embedding_cache = f"embeddings/xlam_{args.field}.npy"
    
    # 设置输出文件
    output_file = args.output
    if output_file is None:
        output_file = f"results/{args.dataset}/diversity_{args.method}_results.json"
    
    # 运行评估
    results = compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name=dataset_name,
        method=args.method,
        field=args.field,
        embedding_model=args.model,
        embedding_cache_path=embedding_cache,
        sample_size=args.sample_size,
        output_file=output_file,
        max_samples=args.max_samples,
        k=args.k,
        embedding_batch_size=args.embedding_batch_size,
        vendi_batch_size=args.vendi_batch_size,
        num_gpus=args.num_gpus,
    )
