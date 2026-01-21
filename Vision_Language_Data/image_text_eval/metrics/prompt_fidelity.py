#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Fidelity 指标 - 图像-文本对齐评估

使用 CLIP 模型计算图像与文本描述之间的相似度。
衡量生成图像与对应文本提示的匹配程度。

使用方式:
    from loaders import GeneralLoader
    from metrics.prompt_fidelity import compute_prompt_fidelity
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_prompt_fidelity(data_iterator=loader.iterate(), device='cuda')
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator, List

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_types import ImageTextSample


class CLIPDataset(Dataset):
    """CLIP 数据集"""
    
    def __init__(self, data: List[ImageTextSample], transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item.image_path).convert("RGB")
        image = self.transform(image)
        text = item.caption
        return image, text


def compute_prompt_fidelity(
    data_iterator: Iterator[ImageTextSample],
    device: str = None,
    batch_size: int = 1024,
) -> float:
    """
    计算 Prompt Fidelity 指标（图像-文本对齐度）
    
    Args:
        data_iterator: ImageTextSample 迭代器
        device: torch device (cuda/cpu)
        batch_size: 批大小
    
    Returns:
        平均余弦相似度分数
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 迭代器转列表
    samples = list(data_iterator)
    
    # 加载 CLIP 模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 创建 DataLoader
    dataset = CLIPDataset(samples, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 计算相似度
    all_sims = []

    with torch.no_grad():
        for images, texts in tqdm(dataloader):
            images = images.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # 编码到统一特征空间
            img_feat = model.encode_image(images)     # [B, D]
            txt_feat = model.encode_text(text_tokens) # [B, D]

            # 归一化
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)

            # 计算余弦相似度（逐对）
            cos_sim = (img_feat * txt_feat).sum(dim=-1)  # [B]
            all_sims.extend(cos_sim.cpu().numpy())

    # 输出结果
    for i, sim in enumerate(all_sims):
        print(f"Sample {i}: Cosine Similarity = {sim:.4f}")
        
    return float(np.mean(all_sims))
