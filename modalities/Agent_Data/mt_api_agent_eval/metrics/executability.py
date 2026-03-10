#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 API Agent Executability

展平所有 Session 为 APIAgentSample 列表，调用单轮 executability。
"""

import os
import sys
import importlib.util
from typing import Iterator, Dict, Any, Optional, List

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)
_agent_data_dir = os.path.dirname(_mt_dir)
_st_dir = os.path.join(_agent_data_dir, 'api_agent_eval')

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)
from data_types import Session, APIAgentSample  # noqa: E402

_st_path = os.path.join(_st_dir, 'metrics', 'executability.py')
_spec = importlib.util.spec_from_file_location('st_executability', _st_path)
_st_mod = importlib.util.module_from_spec(_spec)
if _st_dir not in sys.path:
    sys.path.insert(0, _st_dir)
_spec.loader.exec_module(_st_mod)

st_compute_executability = _st_mod.compute_executability


def _flatten(session_iter: Iterator[Session], max_sessions: Optional[int] = None) -> List[APIAgentSample]:
    samples = []
    n = 0
    for session in session_iter:
        if max_sessions and n >= max_sessions:
            break
        n += 1
        samples.extend(session.get_all_samples())
    return samples


def compute_executability(
    session_iterator: Iterator[Session],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """展平 sessions → 调用单轮 executability。"""
    samples = _flatten(session_iterator, max_sessions)
    return st_compute_executability(
        data_iterator=iter(samples),
        dataset_name=dataset_name,
        output_file=output_file,
        **kwargs,
    )
