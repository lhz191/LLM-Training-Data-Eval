"""
Win Rate 指标计算

WinRate = (W + 0.5 × T) / (W + T + L)

从 evaluation_results 目录读取标注结果并计算 Win Rate。
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path


def calculate_win_rate(comparisons: List[str]) -> Dict[str, Any]:
    """
    计算 Win Rate
    
    Args:
        comparisons: 比较结果列表（"win", "tie", "loss"）
    
    Returns:
        dict: 包含 Win Rate 和统计信息的字典
    """
    if not comparisons:
        return {"win_rate": 0.0, "wins": 0, "ties": 0, "losses": 0, "total": 0}
    
    wins = comparisons.count("win")
    ties = comparisons.count("tie")
    losses = comparisons.count("loss")
    total = len(comparisons)
    
    win_rate = (wins + 0.5 * ties) / total if total > 0 else 0.0
    
    return {
        "win_rate": win_rate,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "total": total,
    }


def load_annotation_file(file_path: str) -> Dict[str, Any]:
    """加载标注文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_win_rate_from_file(file_path: str) -> Dict[str, Any]:
    """
    从标注文件计算 Win Rate
    
    Args:
        file_path: 标注文件路径（JSON 格式）
    
    Returns:
        dict: 包含 Win Rate 和统计信息的字典
    """
    data = load_annotation_file(file_path)
    
    # 提取比较结果
    items = data.get("items", [])
    comparisons = [
        item.get("comparison")
        for item in items
        if item.get("comparison") in ["win", "tie", "loss"]
    ]
    
    if not comparisons:
        raise ValueError("No valid comparisons found in annotation file")
    
    stats = calculate_win_rate(comparisons)
    stats["model"] = data.get("model", "unknown")
    stats["file"] = file_path
    
    return stats


def get_default_evaluation_results_dir() -> str:
    """获取默认的 evaluation_results 目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    return os.path.join(project_root, "evaluation_results")


def get_win_rate(data: List[Dict[str, Any]], cfg: Any = None) -> Dict[str, Any]:
    """
    统一的接口，用于从数据或配置文件计算 Win Rate，模仿其它 metrics 的 `get_*` 风格。

    Args:
        data: 如果没有在 cfg 中指定文件，将尝试从此 `data` 中提取 `comparison` 字段
        cfg: 配置对象（通常来自 `config.win_rate`），包含可选字段 `file` 或 `model`

    Returns:
        dict: 计算好的统计信息（win_rate, wins, ties, losses, total, model, file）
    """
    # 优先使用 cfg 中指定的文件
    file_path = None
    model = None
    if cfg is not None:
        file_path = getattr(cfg, "file", None)
        model = getattr(cfg, "model", None)

    if file_path:
        # 如果是相对路径，按 evaluation_results 目录解析
        if not os.path.isabs(file_path):
            eval_dir = get_default_evaluation_results_dir()
            file_path = os.path.join(eval_dir, file_path)
        return calculate_win_rate_from_file(file_path)

    # 否则尝试从传入的 data 中提取 comparison 字段
    comparisons = []
    for item in data:
        c = None
        # 支持多种可能的字段位置
        if isinstance(item, dict):
            c = item.get("comparison") or item.get("result")
        if c in ["win", "tie", "loss"]:
            comparisons.append(c)

    if comparisons:
        stats = calculate_win_rate(comparisons)
        stats["file"] = None
        stats["model"] = model or "unknown"
        return stats

    raise ValueError("No comparison data found and no file specified in configuration for win_rate")
