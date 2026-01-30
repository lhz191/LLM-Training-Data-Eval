#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Modal Consistency (CMC) 指标 - 跨模态一致性评估

使用 ViCLIP 模型计算视频与文本描述之间的相似度。
衡量视频内容与对应文本描述是否匹配。

使用方式:
    from loaders import GeneralLoader
    from metrics.cross_modal_consistency import compute_cross_modal_consistency
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_cross_modal_consistency(data_iterator=loader.iterate(), device='cuda')
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator, List
from data_types import VideoTextSample
from .internvid.viclip import ViCLIP
from .internvid.simple_tokenizer import SimpleTokenizer as _Tokenizer


class ViCLIP_dataset(Dataset):
    def __init__(self, data: List[VideoTextSample]):
        super().__init__()
        
        self.data = data
        self.target_video_len = 8
        # CLIP 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    
    def __getitem__(self, index):
        path = self.data[index].video_path
        text = self.data[index].text

        vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        # Sampling video frames
        start_frame_ind, end_frame_ind = 0, total_frames-1
        assert end_frame_ind - start_frame_ind >= self.target_video_len, f"Video has too few frames: {total_frames} < {self.target_video_len}, sample:{path}"
        
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        video = vframes[frame_indice].to(torch.float32) / 255.0 # T C H W
        video = torch.stack([self.transform(frame) for frame in video])
        

        return {'video': video,
                'text': text}

    def __len__(self):
        return len(self.data)


clip_candidates = {'viclip': None, 'clip': None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

def normalize(data):
    return (data/255.0-v_mean)/v_std

def retrieve_text(frames, texts, name='viclip', topk=5, device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    vid_feat = get_vid_feat(frames, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    return F.cosine_similarity(vid_feat, text_feats_tensor, dim=1)


def compute_cross_modal_consistency(data_iterator: Iterator[VideoTextSample], device) -> float:
    """
    计算 Cross Modal Consistency (CMC) 指标
    
    Args:
        data_iterator: VideoTextSample 迭代器
        device: torch device (cuda/cpu)
    
    Returns:
        跨模态一致性分数
    """
    # 迭代器转列表（PyTorch Dataset 需要支持 len 和索引）
    samples = list(data_iterator)
    
    dataset = ViCLIP_dataset(samples)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=8)
    
    cos_sims = []
    with torch.no_grad():    
        for x in tqdm(dataloader):
            v, text = x["video"], x["text"]
            v = v.to(device) #[B, T, C, H, W]
            
            cos_sim = retrieve_text(v, text, device=device) # [B, 1]
            cos_sim = cos_sim.mean(dim=0)
            cos_sims.append(cos_sim.item())
    
    res = np.mean(cos_sims)
    
    return float(res)
