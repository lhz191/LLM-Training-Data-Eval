import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)






def compute_corpus_level_eds(
    data, 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
):
    model = SentenceTransformer(model_name)

    gen_texts = []
    ref_texts = []

    # 读取数据
    for item in data:
        gen_texts.append(item["gen_text"])
        ref_texts.append(item["ref_text"])

    # 编码 synthetic corpus
    gen_embeddings = []
    for i in tqdm(range(0, len(gen_texts), batch_size), desc="Encoding synthetic"):
        batch = gen_texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        emb = l2_normalize(emb)
        gen_embeddings.append(emb)
    gen_embeddings = np.vstack(gen_embeddings)

    # 编码 real corpus
    ref_embeddings = []
    for i in tqdm(range(0, len(ref_texts), batch_size), desc="Encoding real"):
        batch = ref_texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        emb = l2_normalize(emb)
        ref_embeddings.append(emb)
    ref_embeddings = np.vstack(ref_embeddings)

    # 语料级均值向量（不再 normalize）
    e_syn = gen_embeddings.mean(axis=0)
    e_real = ref_embeddings.mean(axis=0)

    # cosine similarity
    eds = np.dot(e_syn, e_real) / (
        np.linalg.norm(e_syn) * np.linalg.norm(e_real)
    )

    return float(eds)
