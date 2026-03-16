#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-to-Report Data Loaders

将不同格式的数据集转换为统一的 ImageToReportSample 格式。

支持：
- IU X-Ray:    JSONL 格式，医学报告生成（query/response/images）
- ShareGPT4V:  JSON 格式，通用图像描述（id/image/conversations）
"""

import json
from typing import List, Iterator
from tqdm import tqdm

from data_types import ImageToReportSample


# =============================================================================
# Base Loader
# =============================================================================

class BaseLoader:
    """数据集加载器基类"""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> List[ImageToReportSample]:
        """加载数据集，返回 ImageToReportSample 列表"""
        return list(self.iterate())

    def iterate(self) -> Iterator[ImageToReportSample]:
        """迭代返回 ImageToReportSample，子类需实现"""
        raise NotImplementedError


# =============================================================================
# IU X-Ray Loader
# =============================================================================

class IUXRayLoader(BaseLoader):
    """
    IU X-Ray 数据集加载器

    数据格式 (JSONL, 每行一条):
    {
        "query": "<image><image>Please review the chest X-ray image and create a report...",
        "response": "The heart size and pulmonary vascularity appear within normal limits...",
        "images": ["/iu_xray/image/CXR2384_IM-0942/0.png", "/iu_xray/image/CXR2384_IM-0942/1.png"]
    }

    数据集有 train/val/test 三个 split，每个 split 一个 JSONL 文件。
    """

    def __init__(self, data_path: str, split: str = "train"):
        """
        Args:
            data_path: IU X-Ray 数据根目录 (包含 train.jsonl, val.jsonl, test.jsonl)
            split: 数据分片 (train / val / test)
        """
        super().__init__(data_path)
        self.split = split
        self.jsonl_path = f"{data_path}/{split}.jsonl"

    # 原始数据中图片路径为 /iu_xray/image/CXR2384_IM-0942/0.png
    # 本地目录结构为 images/CXR2384_IM-0942/0.png
    _IMAGE_PREFIX = "/iu_xray/image/"

    def _normalize_image_path(self, raw_path: str) -> str:
        """将数据集中的绝对风格路径转为相对于 data_path 的本地路径"""
        if raw_path.startswith(self._IMAGE_PREFIX):
            return "images/" + raw_path[len(self._IMAGE_PREFIX):]
        return raw_path

    def iterate(self) -> Iterator[ImageToReportSample]:
        with open(self.jsonl_path) as f:
            for idx, line in enumerate(tqdm(f, desc=f"Loading IU X-Ray ({self.split})")):
                record = json.loads(line)
                yield ImageToReportSample(
                    sample_id=f"iu_xray_{self.split}_{idx}",
                    instruction=record["query"],
                    report=record["response"],
                    images=[self._normalize_image_path(p) for p in record["images"]],
                    metadata={
                        "source_dataset": "iu_xray",
                        "domain": "medical",
                        "task": "radiology_report_generation",
                        "split": self.split,
                    },
                )


# =============================================================================
# ShareGPT4V Loader
# =============================================================================

class ShareGPT4VLoader(BaseLoader):
    """
    ShareGPT4V 数据集加载器 (report subset)

    数据格式 (JSON, 合并后的 report_all.json):
    {
        "id": "000000000009",
        "image": "coco/train2017/000000000009.jpg",
        "conversations": [
            {"from": "human", "value": "What do you see happening in this image?\\n<image>"},
            {"from": "gpt",   "value": "In the center of the image, a vibrant blue lunch tray..."}
        ]
    }

    report_all.json 由 cap100k (102k, GPT-4V) + captioner1246k (1.25M, ShareCaptioner) 合并而成。
    全部为单轮、有图、report 风格。
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: 合并后的 JSON 文件路径 (sharegpt4v_report_all.json)
        """
        super().__init__(data_path)

    CAP100K_SIZE = 102025

    def iterate(self) -> Iterator[ImageToReportSample]:
        with open(self.data_path) as f:
            data = json.load(f)

        for idx, record in enumerate(tqdm(data, desc="Loading ShareGPT4V (report)")):
            image_path = record["image"]
            image_source = image_path.split("/")[0]  # coco, sam, llava, wikiart, ...
            caption_source = "cap100k" if idx < self.CAP100K_SIZE else "captioner1246k"

            yield ImageToReportSample(
                sample_id=record["id"],
                instruction=record["conversations"][0]["value"],
                report=record["conversations"][1]["value"],
                images=[image_path],
                metadata={
                    "source": "sharegpt4v",
                    "image_source": image_source,
                    "caption_source": caption_source,
                },
            )
