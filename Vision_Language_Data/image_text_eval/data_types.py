#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-Text 数据类型定义

基于 COCO Caption 等数据集的 JSONL 格式：
{"id": 48, "image_id": 318556, "image_path": "...", "caption": "..."}
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ImageTextSample:
    """图像-文本样本的统一数据结构"""
    
    image_path: str                              # 必需：图像路径
    caption: str                                 # 必需：文本描述
    sample_id: Optional[int] = None              # 可选：样本ID (对应 id 字段)
    image_id: Optional[int] = None               # 可选：图像ID (COCO 等数据集特有)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 可选：扩展字段
    
    def __repr__(self) -> str:
        return f"ImageTextSample(image_path='{self.image_path}', caption='{self.caption[:50]}...')"
