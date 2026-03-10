import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from itertools import combinations
import numpy as np

# -----------------------------
# 初始化语义嵌入模型
# -----------------------------
model_name = "all-MiniLM-L6-v2"  # 轻量级 sentence embedding 模型
embedder = SentenceTransformer(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder.to(device)

# -----------------------------
# 工具函数：计算余弦相似度
# -----------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

# -----------------------------
# 计算 Self-CosSim
# -----------------------------
def compute_self_cosine_similarity(samples: List[Dict]) -> List[Dict]:
    """
    samples: List[{'id': str, 'gen_text': str}]
    返回每条文本的 Self-CosSim
    """
    results = []

    # 1. 获取所有文本
    texts = [sample['gen_text'] for sample in samples]

    # 2. 编码为 embeddings
    embeddings = embedder.encode(texts, convert_to_tensor=True, device=device)

    # 3. 计算所有两两余弦相似度
    M = len(embeddings)
    if M < 2:
        # 如果只有 1 条文本，Self-CosSim = 1
        for sample in samples:
            results.append({'id': sample['id'], 'self_cos_sim': 1.0})
        return results

    # 将 tensor 转为 numpy
    embeddings = embeddings.cpu().numpy()

    # 计算两两组合的余弦相似度
    sim_sum = 0
    count = 0
    for i, j in combinations(range(M), 2):
        sim_sum += cosine_similarity(embeddings[i], embeddings[j])
        count += 1

    avg_sim = sim_sum / count  # Self-CosSim

    # 4. 为每条文本返回相同的 Self-CosSim
    for sample in samples:
        results.append({'id': sample['id'], 'self_cos_sim': avg_sim})

    return results

# -----------------------------
# 示例使用
# -----------------------------
if __name__ == "__main__":
    samples = [
        {'id': '1', 'gen_text': "I love going to the gym on weekends."},
        {'id': '2', 'gen_text': "Exercise is great for your health and mood."},
        {'id': '3', 'gen_text': "Reading books can improve your knowledge."}
    ]

    results = compute_self_cosine_similarity(samples)
    for r in results:
        print(r)
