"""
Video-Text Data Evaluation

评估 Video-Text 数据集的质量，包括：
- Frame Diversity: 帧多样性
- Semantic Diversity: 语义多样性
- Object Consistency: 对象一致性
- Cross-Modal Consistency: 跨模态一致性
- Safety Bench: 安全性评估
- Holistic Fidelity: 整体保真度 (VBench)
"""

from .data_types import VideoTextSample
from .loaders import BaseLoader, GeneralLoader

__all__ = [
    'VideoTextSample',
    'BaseLoader',
    'GeneralLoader',
]
