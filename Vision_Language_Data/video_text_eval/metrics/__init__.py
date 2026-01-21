"""
Video-Text Evaluation Metrics

Available metrics:
- compute_frame_diversity: 帧多样性（光流）
- compute_semantic_diversity: 语义多样性（Inception V3）
- compute_object_consistency: 对象一致性（CLIP）
- compute_cross_modal_consistency: 跨模态一致性（ViCLIP）
- compute_safety_bench: 安全性评估（GPT-4 Vision）
- compute_holistic_fidelity: 整体保真度（VBench）
"""

from .frame_diversity import compute_frame_diversity
from .semantic_diversity import compute_semantic_diversity
from .object_consistency import compute_object_consistency
from .cross_modal_consistency import compute_cross_modal_consistency
from .safety_bench import compute_safety_bench
from .holistic_fidelity import compute_holistic_fidelity

__all__ = [
    'compute_frame_diversity',
    'compute_semantic_diversity',
    'compute_object_consistency',
    'compute_cross_modal_consistency',
    'compute_safety_bench',
    'compute_holistic_fidelity',
]
