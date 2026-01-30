#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inception Score 指标 - 图像质量与多样性评估

使用 Inception V3 模型计算图像的 Inception Score。
衡量生成图像的质量和多样性。

使用方式:
    from loaders import GeneralLoader
    from metrics.inception_score import compute_inception_score
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_inception_score(data_iterator=loader.iterate(), device='cuda')
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional, Iterator
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import ImageTextSample


# -------------------------
# Dataset
# -------------------------

class ImageListDataset(Dataset):
    def __init__(self, image_paths: List[str], tfm: transforms.Compose):
        self.image_paths = image_paths
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.tfm(img)


# -------------------------
# Weights: check-before-download (offline-friendly)
# -------------------------

def _weights_cache_file(weights: Inception_V3_Weights, torch_home: Optional[str]) -> str:
    if torch_home:
        os.environ["TORCH_HOME"] = torch_home
    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    url = weights.url
    fname = os.path.basename(urlparse(url).path)
    return os.path.join(ckpt_dir, fname)


def ensure_weights_available(
    weights: Inception_V3_Weights,
    torch_home: Optional[str],
    allow_download: bool,
) -> None:
    cache_path = _weights_cache_file(weights, torch_home)
    if os.path.exists(cache_path):
        return
    if not allow_download:
        raise FileNotFoundError(
            "Inception-v3 weights not found in local cache, and downloading is disabled.\n"
            f"Expected cached file at:\n  {cache_path}\n\n"
            "Fix options:\n"
            "1) Pre-download weights on a machine with internet, then copy the cache directory to this machine.\n"
            "2) Allow download by setting allow_download=True.\n"
            "3) Set torch_home to point to a directory that already contains hub/checkpoints.\n"
        )


# -------------------------
# Core: Inception Score
# -------------------------

@torch.no_grad()
def _inception_score_from_paths(
    image_paths: List[str],
    batch_size: int = 64,
    splits: int = 10,
    device: str | None = None,
    num_workers: int = 4,
    seed: int = 0,
    torch_home: Optional[str] = None,
    allow_download: bool = True,
) -> Tuple[float, float, int]:
    """
    Returns: (mean_is, std_is, num_images)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(image_paths)
    if n == 0:
        raise ValueError("No valid images found from input.")
    if splits < 1:
        raise ValueError("splits must be >= 1")
    if splits > n:
        splits = n

    weights = Inception_V3_Weights.IMAGENET1K_V1

    tfm = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    ds = ImageListDataset(image_paths, tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    ensure_weights_available(weights, torch_home=torch_home, allow_download=allow_download)

    model = inception_v3(weights=weights, transform_input=False)
    model.eval().to(device)

    probs: List[np.ndarray] = []
    for batch in dl:
        batch = batch.to(device)
        logits = model(batch)
        pyx = F.softmax(logits, dim=1)
        probs.append(pyx.detach().cpu().numpy())

    pyx_all = np.concatenate(probs, axis=0)  # [N, 1000]

    split_scores: List[float] = []
    split_size = n // splits
    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else n
        part = pyx_all[start:end]

        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl = np.sum(kl, axis=1)
        split_scores.append(float(np.exp(np.mean(kl))))

    mean_is = float(np.mean(split_scores))
    std_is = float(np.std(split_scores, ddof=1)) if len(split_scores) > 1 else 0.0
    return mean_is, std_is, n


# -------------------------
# Public API
# -------------------------

def compute_inception_score(
    data_iterator: Iterator[ImageTextSample],
    batch_size: int = 64,
    splits: int = 10,
    device: str = None,
    num_workers: int = 4,
    seed: int = 0,
    torch_home: Optional[str] = None,
    allow_download: bool = True,
) -> float:
    """
    计算 Inception Score 指标
    
    Args:
        data_iterator: ImageTextSample 迭代器
        batch_size: 批大小
        splits: 分割数（用于计算标准差）
        device: torch device (cuda/cpu)
        num_workers: DataLoader 工作进程数
        seed: 随机种子
        torch_home: PyTorch 模型缓存目录
        allow_download: 是否允许下载模型权重
    
    Returns:
        Inception Score 分数
    """
    samples = list(data_iterator)
    image_paths = [sample.image_path for sample in samples]
    
    if not image_paths:
        raise ValueError("No valid images found from input.")
    
    mean_is, std_is, n = _inception_score_from_paths(
        image_paths=image_paths,
        batch_size=batch_size,
        splits=splits,
        device=device,
        num_workers=num_workers,
        seed=seed,
        torch_home=torch_home,
        allow_download=allow_download,
    )
    
    return mean_is
