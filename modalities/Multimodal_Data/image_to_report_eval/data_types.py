#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-to-Report Data Evaluation - 数据类型定义

统一的 Image-to-Report 数据格式，支持：
- IU X-Ray（医学报告生成）
- ShareGPT4V（通用图像描述）

两个数据集的原始格式：

IU X-Ray (JSONL, 每行一条):
{
    "query": "<image><image>Please review the chest X-ray image and create a report...",
    "response": "The heart size and pulmonary vascularity appear within normal limits...",
    "images": ["/iu_xray/image/CXR2384_IM-0942/0.png", ...]
}

ShareGPT4V (JSON, report subset, cap100k + captioner1246k 合并):
{
    "id": "000000000009",
    "image": "coco/train2017/000000000009.jpg",
    "conversations": [
        {"from": "human", "value": "What do you see happening in this image?\\n<image>"},
        {"from": "gpt",   "value": "In the center of the image, a vibrant blue lunch tray..."}
    ]
}
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class ImageToReportSample:
    """
    Image-to-Report 统一数据格式

    字段映射：
                        IU X-Ray              ShareGPT4V
    sample_id       自动生成 (idx)         id 字段
    instruction     query                  conversations[0]["value"]
    report          response               conversations[1]["value"]
    images          images (list)          [image] (单元素 list)

    metadata 可存放来源信息、领域标记等，由 Loader 填充。
    """
    sample_id: str                      # 样本唯一标识
    instruction: str                    # 指令/提示词（含 <image> token）
    report: str                         # 参考报告/描述文本
    images: List[str]                   # 图片路径列表
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        instr = self.instruction[:50] + "..." if len(self.instruction) > 50 else self.instruction
        return (f"ImageToReportSample(id='{self.sample_id}', "
                f"images={len(self.images)}, report_len={len(self.report)}, "
                f"instruction='{instr}')")

    @property
    def image_count(self) -> int:
        """图片数量"""
        return len(self.images)

    @property
    def image_token_count(self) -> int:
        """instruction 中 <image> token 的数量"""
        return self.instruction.count("<image>")

    @property
    def report_length(self) -> int:
        """报告文本字符数"""
        return len(self.report)