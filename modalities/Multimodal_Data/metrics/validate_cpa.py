import argparse
import json
import os
import subprocess
import datetime
from typing import Any, Dict, List
"""
==========================
多模态内容来源可信度评估指标（C2PA）
==========================

本指标用于评估多模态数据集中【生成图像的可验证来源（Provenance）】，
基于 C2PA（Coalition for Content Provenance and Authenticity）标准。

指标定义：
- C2PA Validation Rate（来源凭证验证率）

设 M 为输入 JSON 中所有包含 image_path 的样本集合，
对每张图片 m ∈ M，使用官方 c2patool 进行验证：

  V_c2pa(m) = 1  若图片包含完整且可验证的 C2PA 凭证
             0  否则

则：
  C2PA_Validation_Rate = (1 / |M|) * ∑_{m∈M} V_c2pa(m)

该指标衡量生成图像是否具备：
- 可审计的来源链（provenance chain）
- 完整的签名与 manifest 完整性

注意：
- 对于 Stable Diffusion / 常规生成图像，通常该值为 0（无 C2PA 元数据）
- 对于支持内容凭证的生成系统（如带 C2PA/SynthID 的模型），该指标可显著大于 0

==========================
使用方法（示例）
==========================

1) 准备官方验证工具（一次即可）：
https://github.com/contentauth/c2pa-rs/releases?q=c2patool

2) 对单个 multimodal JSON 文件进行评估：

python metrics/validate_cpa.py \
  --input ../Datasets/Multimodal/coco_caption.json \
  --output_dir LLMDataBenchmark/Multimodal/evaluation_results \
  --c2pa_tool ./tools/c2patool

输出：
- 一个包含 C2PA_validation_rate 的 metric JSON 文件
- 可选逐样本验证结果（image-level）

==========================
"""


# =========================
# C2PA Verification
# =========================

def verify_c2pa(image_path: str, tool_path: str = "./c2patool") -> bool:
    """
    Use official c2patool to verify C2PA provenance.
    Return True iff full validation passes.
    """
    if not os.path.exists(image_path):
        return False

    try:
        res = subprocess.run(
            [tool_path, "verify", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        output = (res.stdout + res.stderr).lower()

        if "no c2pa manifest" in output:
            return False
        if "signature valid" in output and "manifest integrity valid" in output:
            return True
        return False
    except Exception:
        return False


# =========================
# Utilities
# =========================

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_items(data: Any) -> List[Dict[str, Any]]:
    """
    Support both:
    - list[dict]
    - dict[str, dict]
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = list(data.keys())
        try:
            keys.sort(key=lambda x: int(x))
        except Exception:
            keys.sort()
        return [data[k] for k in keys]
    raise ValueError("Unsupported JSON format")


# =========================
# Metric Computation
# =========================

def compute_c2pa_validation_rate(
    items: List[Dict[str, Any]],
    tool_path: str,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    Compute:
      C2PA_Validation_Rate = (1 / |M|) * sum_{m in M} V_c2pa(m)

    Where M is the set of samples that contain image_path (non-empty).
    For each m in M:
      V_c2pa(m) = 1 iff c2patool verifies full C2PA credentials, else 0.
    """
    M = [it for it in items if it.get("image_path")]  # M: has image_path
    total = len(M)

    validated = 0
    details = []

    for it in M:
        img = it["image_path"]
        ok = verify_c2pa(img, tool_path)  # True/False
        validated += 1 if ok else 0

        if return_details:
            details.append(
                {
                    "id": it.get("id"),
                    "image_path": img,
                    "V_c2pa": 1 if ok else 0,
                }
            )

    rate = (validated / total) if total > 0 else None

    result = {
        "C2PA_Validation_Rate": rate,   # 你要的最终指标
        "|M|": total,                  # 可选：集合大小（有助于核对）
        "validated": validated,         # 可选：通过数量（有助于核对）
    }
    if return_details:
        result["details"] = details

    return result


def get_validate_cpa(data: Any, cfg: Any = None) -> Dict[str, Any]:
    """
    统一接口：返回 C2PA Validation Rate（来源凭证验证率）结果。

    Returns:
        dict: 只包含该指标定义所需的结果字段（默认不带 details）。
              结构示例：
              {
                "C2PA_Validation_Rate": 0.0,
                "|M|": 100,
                "validated": 0
              }
    """
    c2pa_tool = "./c2patool"
    output_dir = None

    # 默认：只返回指标本身（不返回逐样本 details）
    return_details = False

    if cfg is not None:
        c2pa_tool = getattr(cfg, "c2pa_tool", c2pa_tool)
        output_dir = getattr(cfg, "output_dir", output_dir)
        return_details = getattr(cfg, "return_details", return_details)

    items = normalize_items(data)

    metric = compute_c2pa_validation_rate(
        items=items,
        tool_path=c2pa_tool,
        return_details=return_details,
    )

    # 如果需要写文件（可选）
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            out_name = f"c2pa_validation_rate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path = os.path.join(output_dir, out_name)
            write_json(out_path, metric)
            metric["output_file"] = out_path
        except Exception:
            pass

    return metric



# =========================
# Main
# =========================

def get_validate_cpa(data: Any, cfg: Any = None) -> Dict[str, float]:
    """
    Return only:
      { "C2PA_Validation_Rate": value }

    Definition:
      M = all samples that contain image_path
      V_c2pa(m) = 1 if verified, else 0
      Rate = (1 / |M|) * sum V_c2pa(m)
    """
    c2pa_tool = "./c2patool"
    if cfg is not None:
        c2pa_tool = getattr(cfg, "c2pa_tool", c2pa_tool)

    items = normalize_items(data)

    # M: samples that contain image_path
    M = [it for it in items if it.get("image_path")]
    total = len(M)

    if total == 0:
        return {"C2PA_Validation_Rate": None}

    validated = 0
    for it in M:
        if verify_c2pa(it["image_path"], c2pa_tool):
            validated += 1

    rate = validated / total

    return rate
