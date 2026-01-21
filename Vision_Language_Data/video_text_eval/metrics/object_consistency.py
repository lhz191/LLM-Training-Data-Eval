#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Consistency 指标 - 对象一致性评估

使用 CLIP 模型计算视频中相邻帧之间的特征相似度。
衡量视频内容是否连贯（对象是否在时间上保持稳定）。

使用方式:
    from loaders import GeneralLoader
    from metrics.object_consistency import compute_object_consistency
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_object_consistency(data_iterator=loader.iterate(), device='cuda')
"""

import torch
import numpy as np
from transformers import CLIPModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator, List
from data_types import VideoTextSample


class Video_dataset(Dataset):
    def __init__(self, data: List[VideoTextSample]):
        super().__init__()
        
        self.data = data
        self.target_video_len = 16
        # CLIP 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
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
        video = vframes[frame_indice].to(torch.float32) / 255.0 # T C H W
        video = torch.stack([self.transform(frame) for frame in video])
        

        return {'video': video}

    def __len__(self):
        return len(self.data)
    

def calculate_fidelity(dataset, model, device):
    """
    Calculate FID (Fidelity) based on object feature similarity between adjacent frames.

    Parameters:
    - video_frames (list of torch.Tensor): List of frames as tensors, shape [F, C, H, W]
    - model (CLIPModel): Pre-trained CLIP model for feature extraction

    Returns:
    - float: Fidelity score (FID)
    """
    N = len(dataset) # Total number

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False, num_workers=8)
    
    # Extract features for each frame
    features = []
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():

        for x in tqdm(dataloader, desc="Extracting features"):
            frame = x['video'] # [B, T, C, H, W]
            frame = frame.to(device)
            B, T, C, H, W = frame.shape
            frame = frame.reshape(B*T, C, H, W)
            output = model.get_image_features(frame)
            features.append(output)  # Shape: [C]
    
    features = torch.cat(features, dim=0)
    features = features.reshape(N, T, -1)
    # Compute similarity between adjacent frames
    # L2-normalize for stable cosine
    features = F.normalize(features, p=2, dim=-1)

    # compute cosine similarity between adjacent frames
    sims = (features[:, :-1] * features[:, 1:]).sum(dim=-1)  # equivalent to cosine if normalized

    # per-video fidelity: mean over time
    fid_per_video = sims.mean(dim=1)  # [B]
    
    return fid_per_video.mean(dim=0).item()


def compute_object_consistency(data_iterator: Iterator[VideoTextSample], device) -> float:
    """
    计算 Object Consistency 指标
    
    Args:
        data_iterator: VideoTextSample 迭代器
        device: torch device (cuda/cpu)
    
    Returns:
        对象一致性分数
    """
    # 迭代器转列表（PyTorch Dataset 需要支持 len 和索引）
    samples = list(data_iterator)
    
    dataset = Video_dataset(samples)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    return calculate_fidelity(dataset, model, device)
