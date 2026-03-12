"""
JSON Fidelity — Mean Match Percentage (MMP)  (Survey §5.2.2)

    MMP = (1/N) Σ |fields(J_i) ∩ fields(J_gt_i)| / |fields(J_gt_i)|

比较生成 JSON 的字段结构与 ground-truth 的匹配程度。
额外提供 value-level match rate（Survey 提到可扩展到 value-level checks）。
"""

import json
from typing import List, Dict, Any, Set


def _flatten_keys(obj, prefix: str = '') -> Set[str]:
    """Recursively flatten a nested dict/list into dot-separated key paths."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            keys |= _flatten_keys(v, full_key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            keys |= _flatten_keys(v, f"{prefix}[]")
    return keys


def _get_value(obj, key_path: str):
    """Navigate a nested structure using a dot-separated key path."""
    parts = key_path.split('.')
    current = obj
    for part in parts:
        if part.endswith('[]'):
            part = part[:-2]
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def compute_json_fidelity(
    data: List[Dict[str, Any]],
    cfg: Any = None,
) -> Dict[str, Any]:
    """
    Compute Mean Match Precision for generated JSON against reference.

    Args:
        data: list of dicts with "output" (generated) and "reference" (ground truth) fields
        cfg: optional config
    """
    json_field = "output"
    ref_field = "reference"

    if cfg is not None:
        jcfg = getattr(cfg, 'json_eval', None)
        if jcfg:
            json_field = getattr(jcfg, 'json_field', json_field)

    total = 0
    key_precisions = []
    value_matches = []

    for item in data:
        gen_text = item.get(json_field, '')
        ref_text = item.get(ref_field, item.get('ground_truth', ''))

        if isinstance(gen_text, str):
            try:
                gen_obj = json.loads(gen_text)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            gen_obj = gen_text

        if isinstance(ref_text, str):
            try:
                ref_obj = json.loads(ref_text)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            ref_obj = ref_text

        if ref_obj is None:
            continue

        total += 1
        gen_keys = _flatten_keys(gen_obj)
        ref_keys = _flatten_keys(ref_obj)

        if ref_keys:
            precision = len(gen_keys & ref_keys) / len(ref_keys)
        else:
            precision = 1.0 if not gen_keys else 0.0
        key_precisions.append(precision)

        matched = gen_keys & ref_keys
        if matched:
            vm = sum(
                1 for k in matched
                if str(_get_value(gen_obj, k)) == str(_get_value(ref_obj, k))
            ) / len(matched)
            value_matches.append(vm)

    mmp = float(sum(key_precisions) / len(key_precisions)) if key_precisions else 0.0
    avg_val_match = float(sum(value_matches) / len(value_matches)) if value_matches else 0.0

    return {
        'mean_match_precision': mmp,
        'avg_value_match': avg_val_match,
        'total': total,
    }
