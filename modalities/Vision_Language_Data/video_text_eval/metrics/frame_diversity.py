#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Diversity 指标 - 帧多样性评估

通过光流（Farneback 算法）计算视频的运动程度/帧间差异。
分数越高表示视频越动态，分数越低表示视频越静态。

使用方式:
    from loaders import GeneralLoader
    from metrics.frame_diversity import compute_frame_diversity
    
    loader = GeneralLoader('/path/to/data.jsonl')
    
    score = compute_frame_diversity(data_iterator=loader.iterate())
"""

import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator
from data_types import VideoTextSample


def compute_frame_diversity(data_iterator: Iterator[VideoTextSample]) -> float:
    """
    计算 Frame Diversity 指标
    
    Args:
        data_iterator: VideoTextSample 迭代器
    
    Returns:
        所有视频的平均帧多样性分数
    """
    res = []
    for sample in data_iterator:
        video_path = sample.video_path
        cap = cv2.VideoCapture(video_path)
        
        # 获取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            continue

        # 转为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        motion_scores = []
        
        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            
            if frame_cnt % 10 != 0:
                continue
            # 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算光流（使用Farneback算法）
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 计算光流的幅值作为运动评分
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(mag)  # 运动评分是所有光流幅值的平均值
            
            motion_scores.append(motion_score)
            
            # 更新上一帧图像
            prev_gray = gray

        cap.release()
        
        if motion_scores:
            res.append(np.mean(motion_scores))
    
    return np.mean(res) if res else 0.0
