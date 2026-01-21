#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-Text Data Evaluation - 数据集加载器

支持的数据集：
- General: 通用 JSONL 格式 ({"idx": 0, "video": "path", "text": "xxx"})
"""

import json
from pathlib import Path
from typing import Iterator, List
from data_types import VideoTextSample


class BaseLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def load(self) -> List[VideoTextSample]:
        """加载数据集，返回 VideoTextSample 列表"""
        return list(self.iterate())
    
    def iterate(self) -> Iterator[VideoTextSample]:
        """迭代返回 VideoTextSample，子类需实现"""
        raise NotImplementedError


class GeneralLoader(BaseLoader):
    """
    通用 JSONL 数据集加载器
    
    数据格式 (JSONL):
    {"idx": 0, "video": "/path/to/video.mp4", "text": "description"}
    
    字段说明：
    - idx: 样本索引（可选，如果没有会自动生成）
    - video: 视频文件路径
    - text: 文本描述
    """
    
    def iterate(self) -> Iterator[VideoTextSample]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                
                yield VideoTextSample(
                    video_path=record.get('video', ''),
                    text=record.get('text', ''),
                    sample_id=record.get('idx', i),
                    metadata={k: v for k, v in record.items() 
                             if k not in ('idx', 'video', 'text')}
                )
