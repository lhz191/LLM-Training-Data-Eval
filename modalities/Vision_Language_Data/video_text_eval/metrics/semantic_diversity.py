#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Diversity 指标 - 语义多样性评估

使用 Inception V3 模型计算视频帧的语义多样性。
通过计算帧间预测分布的 KL 散度来衡量语义变化。

使用方式:
    from loaders import GeneralLoader
    from metrics.semantic_diversity import compute_semantic_diversity
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_semantic_diversity(data_iterator=loader.iterate(), device='cuda')
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torchvision

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator, List
from data_types import VideoTextSample


class Inception_dataset(Dataset):
    def __init__(self, data: List[VideoTextSample]):
        super().__init__()
        
        self.data = data
        self.target_video_len = 16
        # CLIP 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(299),          # 最短边 299
            transforms.CenterCrop(299),      # 中心裁剪到 299x299
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225]    # ImageNet 标准差
            )
        ])

    
    def __getitem__(self, index):
        path = self.data[index].video_path

        vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        # Sampling video frames
        start_frame_ind, end_frame_ind = 0, total_frames-1
        assert end_frame_ind - start_frame_ind >= self.target_video_len, f"Video has too few frames: {total_frames} < {self.target_video_len}, sample:{path}"
        
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        video = vframes[frame_indice].float() / 255.0 # T C H W
        video = torch.stack([self.transform(frame) for frame in video])
        

        return {'video': video}

    def __len__(self):
        return len(self.data)



def load_inception(device):
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.to(device)
    return model


def semantic_diversity(preds):
    # preds: [B, T, C]   softmax outputs
    py = preds.mean(dim=1, keepdim=True)# p(y) [B, 1, C]
    kl = preds * (preds.log() - py.log())  # KL divergence for each sample [B, T, C]
    kl = kl.sum(dim=2) # sum over classes [B, T]
    kl = kl.mean(dim=1) # [B]
    return torch.exp(kl)


def compute_semantic_diversity(data_iterator: Iterator[VideoTextSample], device) -> float:
    """
    计算 Semantic Diversity 指标
    
    Args:
        data_iterator: VideoTextSample 迭代器
        device: torch device (cuda/cpu)
    
    Returns:
        语义多样性分数
    """
    # 迭代器转列表（PyTorch Dataset 需要支持 len 和索引）
    samples = list(data_iterator)
    
    model = load_inception(device)
    
    dataset = Inception_dataset(samples)
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    
    sd_list = []
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            v = x["video"] # [B,T,C,H,W]
            B, T, C, H, W = v.shape
            v = v.to(device)
            frames = v.reshape(B*T, C, H, W)
            preds = model(frames)
            preds = F.softmax(preds, -1)
            preds = preds.reshape(B, T, -1)
            
            sd_list.append(semantic_diversity(preds))
    sd_list = torch.cat(sd_list, dim=0).cpu().numpy()
    return np.mean(sd_list)
