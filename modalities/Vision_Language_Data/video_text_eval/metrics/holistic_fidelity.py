"""
Holistic Fidelity - 整体保真度评估 (基于 VBench)

使用 VBench 框架对视频进行多维度质量评估，
包括主体一致性、背景一致性、美学质量、成像质量等。

原始文件: LLMDataBenchmark/Multimodal/metrics/holistic_fidelity.py
"""

import os
import sys
import json
import torch
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_types import VideoTextSample

# VBench 路径（在 metrics/ 目录下）
VBENCH_PATH = os.path.dirname(os.path.abspath(__file__))


def compute_holistic_fidelity(
    data_iterator: Iterator[VideoTextSample],
    dimension_list: Optional[List[str]] = None,
    output_path: str = "./evaluation_results",
    full_json_dir: Optional[str] = None,
    device: Optional[str] = None,
    load_ckpt_from_local: bool = False,
    read_frame: bool = False,
    mode: str = 'custom_input',
    max_samples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算 Holistic Fidelity 指标（基于 VBench）
    
    Args:
        data_iterator: VideoTextSample 迭代器
        dimension_list: 要评估的维度列表，默认为全部支持的维度
        output_path: VBench 输出目录
        full_json_dir: VBench 完整信息 JSON 文件路径
        device: 计算设备 ('cuda' 或 'cpu')
        load_ckpt_from_local: 是否从本地加载模型
        read_frame: 是否读取帧
        mode: 评估模式
        max_samples: 最大处理样本数
        output_file: 结果输出文件路径
    
    Returns:
        包含各维度评估结果的字典
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # 设置 VBench JSON 路径
    if full_json_dir is None:
        full_json_dir = os.path.join(VBENCH_PATH, 'vbench/VBench_full_info.json')
    
    # 收集样本，转换为 VBench 需要的格式
    samples = list(data_iterator)
    if max_samples is not None:
        samples = samples[:max_samples]
    
    video_data = []
    for sample in samples:
        if os.path.exists(sample.video_path):
            video_data.append({'video': sample.video_path})
    
    if not video_data:
        return {
            "metric_name": "Holistic Fidelity (VBench)",
            "error": "No valid video files found",
            "total_samples": len(samples),
            "valid_samples": 0,
        }
    
    # 尝试导入 VBench
    try:
        sys.path.insert(0, VBENCH_PATH)
        from vbench import VBench
        from vbench.distributed import dist_init, print0
    except ImportError as e:
        return {
            "metric_name": "Holistic Fidelity (VBench)",
            "error": f"Failed to import VBench: {e}",
            "hint": "Please ensure VBench is installed and accessible",
        }
    
    # 初始化分布式环境
    try:
        dist_init()
    except:
        pass  # 非分布式环境下可能会失败
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化 VBench
    my_VBench = VBench(device, full_json_dir, output_path)
    
    # 默认维度列表（排除需要额外信息的维度）
    if dimension_list is None:
        # 这些维度不需要额外的标注信息
        dimension_list = [
            "subject_consistency",
            "background_consistency", 
            "aesthetic_quality",
            "imaging_quality",
            "temporal_flickering",
            "motion_smoothness",
            "dynamic_degree",
            "overall_consistency",
            "human_action",
        ]
    
    # 运行评估
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    try:
        results_dict = my_VBench.evaluate(
            video_data=video_data,
            name=f'results_{current_time}',
            prompt_list=[],
            dimension_list=dimension_list,
            local=load_ckpt_from_local,
            read_frame=read_frame,
            mode=mode,
        )
    except Exception as e:
        return {
            "metric_name": "Holistic Fidelity (VBench)",
            "error": f"VBench evaluation failed: {e}",
            "total_samples": len(samples),
            "valid_samples": len(video_data),
        }
    
    # 组织结果
    final_results = {
        "metric_name": "Holistic Fidelity (VBench)",
        "total_samples": len(samples),
        "valid_samples": len(video_data),
        "dimensions_evaluated": dimension_list,
        "results": results_dict,
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    return final_results


# VBench 支持的所有维度（供参考）
VBENCH_ALL_DIMENSIONS = [
    "subject_consistency",      # 主体一致性
    "background_consistency",   # 背景一致性
    "aesthetic_quality",        # 美学质量
    "imaging_quality",          # 成像质量
    "object_class",             # 对象类别（需要标注）
    "multiple_objects",         # 多对象（需要标注）
    "color",                    # 颜色（需要标注）
    "spatial_relationship",     # 空间关系（需要标注）
    "scene",                    # 场景（需要标注）
    "temporal_style",           # 时间风格
    "overall_consistency",      # 整体一致性
    "human_action",             # 人类动作
    "temporal_flickering",      # 时间闪烁
    "motion_smoothness",        # 运动平滑度
    "dynamic_degree",           # 动态程度
    "appearance_style",         # 外观风格（需要标注）
]

# 不需要额外标注的维度
VBENCH_CUSTOM_SUPPORTED_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
]


if __name__ == '__main__':
    # 示例用法
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loaders import GeneralLoader
    
    test_jsonl_path = '/mnt/petrelfs/liuhaoze/main/Vision_Language_Data/LLMDataBenchmark/Multimodal/data_utils/test.jsonl'
    
    loader = GeneralLoader(test_jsonl_path)
    
    print("\n" + "="*60)
    print("Running Holistic Fidelity (VBench) Evaluation")
    print("="*60)
    
    results = compute_holistic_fidelity(
        data_iterator=loader.iterate(),
        dimension_list=["aesthetic_quality"],  # 只测试一个维度
        max_samples=2,
    )
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))
