#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity 多样性指标

支持两种多样性计算方法：
1. Vendi Score - 基于核矩阵特征值的多样性指标，需要采样
2. KNN 平均距离 - 基于 K 近邻距离的多样性指标，可全量计算

使用方式:
    from diversity import compute_diversity
    
    results = compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name='OpenMathInstruct-1',
        method='knn',  # 或 'vendi'
        field='question',
        embedding_cache_path='embeddings/openmath_question.npy',
    )
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Optional, Iterator, Dict, Any, List
from tqdm import tqdm

from data_types import MathSample


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


def generate_embeddings(
    data_iterator: Iterator[MathSample],
    field: str = "question",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    max_text_length: int = 8192,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    生成 embedding
    
    Args:
        data_iterator: MathSample 迭代器
        field: 要提取的字段 ('question', 'solution', 'both')
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
        if field == "question":
            text = sample.question or ""
        elif field == "solution":
            solution = sample.solution
            if isinstance(solution, list):
                text = "\n".join(solution)
            else:
                text = str(solution) if solution else ""
        elif field == "both":
            question = sample.question or ""
            solution = sample.solution
            if isinstance(solution, list):
                solution = "\n".join(solution)
            else:
                solution = str(solution) if solution else ""
            text = f"Question: {question}\n\nSolution: {solution}"
        else:
            raise ValueError(f"Unknown field: {field}")
        
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
    
    # 构建 KNN 模型
    print("构建 KNN 模型...")
    nn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 因为包含自己
        metric=distance_metric,
        algorithm='auto',
        n_jobs=-1,
    )
    nn.fit(embeddings)
    
    # 查询所有样本的 K 近邻
    print("查询 K 近邻...")
    distances, indices = nn.kneighbors(embeddings)
    
    # 排除自己（第一个近邻是自己，距离为 0）
    k_distances = distances[:, 1:k+1]
    
    # 计算统计量
    mean_distance = float(np.mean(k_distances))
    std_distance = float(np.std(k_distances))
    median_distance = float(np.median(k_distances))
    
    # 每个样本的平均 K 近邻距离
    per_sample_mean = np.mean(k_distances, axis=1)
    
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
# 主函数
# =============================================================================

def compute_diversity(
    data_iterator: Iterator[MathSample],
    dataset_name: str = "Unknown",
    method: str = "knn",
    field: str = "question",
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
        field: 用于计算 embedding 的字段 ('question', 'solution', 'both')
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
    
    # Step 1: 生成或加载 embedding
    print("-" * 50)
    print("Step 1: 生成/加载 Embedding")
    print("-" * 50)
    
    # 如果指定了 embedding_batch_size，使用它；否则使用默认值
    emb_batch_size = embedding_batch_size if embedding_batch_size is not None else 8
    
    embeddings = generate_embeddings(
        data_iterator=data_iterator,
        field=field,
        model_name=embedding_model,
        batch_size=emb_batch_size,
        max_samples=max_samples,
        cache_path=embedding_cache_path,
    )
    
    embedding_time = time.time() - start_time
    print(f"Embedding 耗时: {embedding_time:.1f} 秒")
    print()
    
    # Step 2: 计算多样性
    print("-" * 50)
    print(f"Step 2: 计算多样性 ({method})")
    print("-" * 50)
    
    diversity_start = time.time()
    
    if method == "knn":
        diversity_result = compute_knn_diversity(
            embeddings=embeddings,
            k=k,
            distance_metric=distance_metric,
            sample_size=sample_size,
        )
        diversity_score = diversity_result["knn_mean_distance"]
    elif method == "vendi":
        diversity_result = compute_vendi_score(
            embeddings=embeddings,
            similarity_metric=similarity_metric,
            sample_size=sample_size,
            batch_size=vendi_batch_size,
            num_gpus=num_gpus,
        )
        diversity_score = diversity_result["vendi_score"]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn' or 'vendi'.")
    
    diversity_time = time.time() - diversity_start
    print(f"多样性计算耗时: {diversity_time:.1f} 秒")
    print()
    
    total_time = time.time() - start_time
    
    # 汇总结果
    results = {
        "dataset": dataset_name,
        "method": method,
        "field": field,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "embedding_time_seconds": embedding_time,
        "diversity_time_seconds": diversity_time,
        "diversity_score": diversity_score,
        **diversity_result,
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
    print(f"数据集: {dataset_name}")
    print(f"方法: {method}")
    print(f"多样性分数: {diversity_score:.6f}")
    print()
    
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diversity 多样性评估")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["openmathinstruct", "lila"],
                        help="数据集名称")
    parser.add_argument("--method", type=str, default="knn",
                        choices=["knn", "vendi"],
                        help="多样性计算方法")
    parser.add_argument("--field", type=str, default="question",
                        choices=["question", "solution", "both"],
                        help="用于计算 embedding 的字段")
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
    
    args = parser.parse_args()
    
    # 加载数据集
    if args.dataset == "openmathinstruct":
        from loaders import OpenMathInstructLoader
        loader = OpenMathInstructLoader(
            '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/OpenMathInstruct-1',
            use_correct=True
        )
        dataset_name = "OpenMathInstruct-1"
        embedding_cache = f"embeddings/openmath_{args.field}.npy"
    elif args.dataset == "lila":
        from loaders import LILALoader
        loader = LILALoader(
            '/mnt/petrelfs/liuhaoze/datasets/Symbolic_and_Logical_Data/LILA/lila/multi/iid/train_math_only.json'
        )
        dataset_name = "LILA"
        embedding_cache = f"embeddings/lila_{args.field}.npy"
    
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
    )

