"""
Well-Formed Rate (WFR) 指标计算

WFR = (1/|M_gen|) * Σ V(m, S)
其中 V(m, S) ∈ {0, 1} 指示生成结果 m 是否符合 schema S。
"""

import json
import os
import re
from typing import Dict, List, Any, Callable, Optional


# ==================== Schema Validators ====================

# 用于解析 image_path 中的 image_id
_COCO_PATH_RE = re.compile(r"^train2014/COCO_train2014_0{6}(\d{6})\.jpg$")


def validate_coco_caption(
    item: Dict[str, Any],
    strict: bool = True,
    dataset_root: Optional[str] = None,
) -> bool:
    """验证 COCO Caption 数据项（包含 grounded + image_id 绑定）"""

    required_fields = ("id", "image_id", "image_path", "caption")

    # 1) 必需字段是否齐全（结构良构）
    if not all(k in item for k in required_fields):
        return False

    # 2) 字段类型是否正确（可解析）
    if not (
        isinstance(item["id"], int)
        and isinstance(item["image_id"], int)
        and isinstance(item["image_path"], str)
        and isinstance(item["caption"], str)
    ):
        return False

    # 3) 严格模式下的基础值检查
    if strict:
        if item["id"] <= 0 or item["image_id"] <= 0:
            return False
        if not item["caption"].strip():
            return False

    # 4) image_path 格式 + 提取 path 中的 image_id
    path = item["image_path"].replace("\\", "/")
    match = _COCO_PATH_RE.match(path)
    if strict and match is None:
        return False

    # 5) 正确绑定：image_id 必须与路径中的 ID 一致
    if match is not None:
        if int(match.group(1)) != item["image_id"]:
            return False

    # 6) grounded：图片文件必须真实存在
    if dataset_root is not None:
        if not os.path.exists(os.path.join(dataset_root, path)):
            return False

    return True


# ==================== Validator Registry ====================

VALIDATORS: Dict[str, Callable] = {
    "coco_caption": validate_coco_caption,
    # 以后可以在这里继续加
    # "flickr30k": validate_flickr30k,
}


# ==================== Core Functions ====================

def calculate_wfr_from_file(
    json_file: str,
    dataset: str,
    strict: bool = True,
    dataset_root: Optional[str] = None,
) -> Dict[str, Any]:
    """从 JSON 文件计算 WFR"""

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file should contain a list, but got {type(data)}")

    validator = VALIDATORS.get(dataset)
    if validator is None:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(VALIDATORS.keys())}"
        )

    valid_count = sum(
        1 for item in data
        if validator(item, strict=strict, dataset_root=dataset_root)
    )

    return {
        "wfr": valid_count / len(data) if data else 0.0,
        "total_items": len(data),
        "valid_count": valid_count,
        "invalid_count": len(data) - valid_count,
        "dataset": dataset,
        "strict": strict,
    }


# ==================== Main ====================

def get_well_formed_rate(data: List[Dict[str, Any]], cfg: Optional[Any] = None) -> float:
    """
    计算 Well-Formed Rate (WFR)。

    新的接口接受 (data, cfg) 形式：
      - `data`：解析后的 JSONL 列表（list[dict])
      - `cfg`：可选配置对象（例如从 `get_cfg()` 返回的 CfgNode），
               函数会尝试从 `cfg.well_formed_rate` 中读取 `dataset`, `strict`, `dataset_root` 等字段。

    为向后兼容，如果传入 `cfg` 为字符串（表示 dataset 名称），仍然可用。

    返回 WFR（0-1 的 float）。
    """
    # 默认值
    dataset = "coco_caption"
    strict = True
    dataset_root = None

    # 处理 cfg 参数：支持 CfgNode、dict 或直接传入 dataset 名称的旧用法
    if cfg is not None:
        # 如果传入的是字符串，认为是 dataset 名称（向后兼容）
        if isinstance(cfg, str):
            dataset = cfg
        else:
            # 尝试从 cfg.well_formed_rate 中读取参数
            try:
                wcfg = getattr(cfg, "well_formed_rate", cfg)
            except Exception:
                wcfg = cfg

            # wcfg 可能是 CfgNode 或 dict
            if hasattr(wcfg, "get") and not hasattr(wcfg, "dataset"):
                # dict-like
                dataset = wcfg.get("dataset", dataset)
                strict = wcfg.get("strict", strict)
                dataset_root = wcfg.get("dataset_root", dataset_root)
            else:
                # attribute access
                dataset = getattr(wcfg, "dataset", dataset)
                strict = getattr(wcfg, "strict", strict)
                dataset_root = getattr(wcfg, "dataset_root", dataset_root)

    validator = VALIDATORS.get(dataset)
    if validator is None:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(VALIDATORS.keys())}"
        )

    if not data:
        return 0.0

    valid_count = sum(
        1 for item in data
        if validator(item, strict=strict, dataset_root=dataset_root)
    )

    wfr = valid_count / len(data)
    return wfr
