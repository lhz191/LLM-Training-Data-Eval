#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-GUI Agent Trajectory Validity

展平所有 Session 为 Record 列表，调用单轮 trajectory_validity。
"""

import os
import sys
import importlib.util
from typing import Iterator, Dict, Any, Optional, List

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)
_agent_data_dir = os.path.dirname(_mt_dir)
_st_dir = os.path.join(_agent_data_dir, 'text_gui_agent_eval')

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)
from data_types import Session, Record  # noqa: E402

_st_path = os.path.join(_st_dir, 'metrics', 'trajectory_validity.py')
_spec = importlib.util.spec_from_file_location('st_trajectory_validity', _st_path)
_st_mod = importlib.util.module_from_spec(_spec)
if _st_dir not in sys.path:
    sys.path.insert(0, _st_dir)
_spec.loader.exec_module(_st_mod)

st_compute_trajectory_validity = _st_mod.compute_trajectory_validity


def _flatten(session_iter: Iterator[Session], max_sessions: Optional[int] = None) -> List[Record]:
    records = []
    n = 0
    for session in session_iter:
        if max_sessions and n >= max_sessions:
            break
        n += 1
        records.extend(session.get_all_records())
    return records


def compute_trajectory_validity(
    session_iterator: Iterator[Session],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """展平 sessions → 调用单轮 trajectory_validity。"""
    records = _flatten(session_iterator, max_sessions)
    return st_compute_trajectory_validity(
        data_iterator=iter(records),
        dataset_name=dataset_name,
        output_file=output_file,
        **kwargs,
    )
