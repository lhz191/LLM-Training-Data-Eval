#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-Text 数据加载器

支持 JSONL 格式的图像-文本数据集加载。
"""

import json
from abc import ABC, abstractmethod
from typing import Iterator, List

from data_types import ImageTextSample


class BaseLoader(ABC):
    """数据加载器基类"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    @abstractmethod
    def load(self) -> List[ImageTextSample]:
        """加载全部数据到内存"""
        pass
    
    @abstractmethod
    def iterate(self) -> Iterator[ImageTextSample]:
        """迭代方式加载数据（节省内存）"""
        pass


class GeneralLoader(BaseLoader):
    """
    通用 JSONL 加载器
    
    期望格式：
    {"id": 48, "image_id": 318556, "image_path": "...", "caption": "..."}
    """
    
    def load(self) -> List[ImageTextSample]:
        return list(self.iterate())
    
    def iterate(self) -> Iterator[ImageTextSample]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                yield ImageTextSample(
                    image_path=data.get('image_path', ''),
                    caption=data.get('caption', ''),
                    sample_id=data.get('id'),
                    image_id=data.get('image_id'),
                    metadata={k: v for k, v in data.items() 
                              if k not in ('image_path', 'caption', 'id', 'image_id')}
                )
