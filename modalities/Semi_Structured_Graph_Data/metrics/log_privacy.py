"""
Log Privacy — Qualitative Effectiveness Score  (Survey §5.3.3)

指标来源: Aghili et al., 2025; Pilán et al., 2022

人工评估指标：由人类专家对匿名化日志的隐私保护效果和可用性进行
Likert 量表打分 (如 1-5 分)，取平均。

ScoreQualitative = (1/|R|) Σ ri

依赖: numpy
"""

from typing import Dict, Any, List
import numpy as np


def compute_qualitative_score(
    ratings: List[float],
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> Dict[str, Any]:
    """
    计算人工评估的 Qualitative Effectiveness Score。

    参数:
        ratings:   人工打分列表 (如 Likert 1-5)
        scale_min: 量表最小值
        scale_max: 量表最大值

    返回:
        dict: 均值、标准差、归一化均值、评估人数
    """
    if not ratings:
        return {
            'qualitative_score': 0.0,
            'qualitative_score_normalized': 0.0,
            'std': 0.0,
            'num_raters': 0,
        }

    arr = np.array(ratings, dtype=np.float64)
    mean_score = float(np.mean(arr))
    scale_range = scale_max - scale_min

    return {
        'qualitative_score': mean_score,
        'qualitative_score_normalized': (mean_score - scale_min) / scale_range if scale_range > 0 else 0.0,
        'std': float(np.std(arr)),
        'num_raters': len(ratings),
        'scale': (scale_min, scale_max),
    }
