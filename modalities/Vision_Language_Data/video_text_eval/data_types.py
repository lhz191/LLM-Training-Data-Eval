#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-Text 数据类型定义

统一的数据结构，用于视频-文本多模态数据评估。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class VideoTextSample:
    """
    视频-文本样本的统一数据结构
    
    核心字段（评测必需）：
    - video_path: 视频文件路径
    - text: 对应的文本描述/caption
    
    元信息：
    - sample_id: 样本唯一标识
    - metadata: 其他可选元数据
    """
    # === 必需字段 ===
    video_path: str                     # 视频文件路径
    text: str                           # 文本描述
    
    # === 元信息 ===
    sample_id: Optional[int] = None     # 样本唯一标识
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"VideoTextSample(id={self.sample_id}, text='{text_preview}')"
