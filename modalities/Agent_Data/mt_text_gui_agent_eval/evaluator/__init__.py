#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard Model Evaluator 包 — 多轮 Text GUI Agent 版本

主入口是 evaluate_session()，对整个 Session（多轮会话）进行安全性评估。
也保留 evaluate() 用于评估单个 Record。

支持的 Guard 模型:
- AgentDoG: 上海 AI Lab 的 Agent 安全诊断框架，细粒度分类 (已实现)
- ShieldAgent: 浙大/蚂蚁，多阶段 Agent 防护 (TODO)
- WebGuard: 专门 Web Agent 安全，网页操作场景风险分类 (TODO)
- LlamaGuard: Meta 通用内容安全模型，可做 baseline (TODO)
- WildGuard: Allen AI 通用 safety，adversarial prompt 检测 (TODO)
- Qwen3Guard: 阿里通用安全模型 (TODO)

使用方式：
    from mt_text_gui_agent_eval.evaluator.agentdog import AgentDoGEvaluator

    evaluator = AgentDoGEvaluator(model_path='/path/to/model')

    # 多轮评估（主接口）
    result = evaluator.evaluate_session(session)

    # 单 Record 评估（兼容接口）
    result = evaluator.evaluate(record)

    # 通过工厂函数
    from mt_text_gui_agent_eval.evaluator import get_evaluator
    evaluator = get_evaluator('agentdog', model_path='/path/to/model')
"""

import os
import sys

_current_file = os.path.abspath(__file__)
_evaluator_dir = os.path.dirname(_current_file)
_mt_eval_dir = os.path.dirname(_evaluator_dir)
_agent_data_dir = os.path.dirname(_mt_eval_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)

_project_root = os.path.dirname(os.path.dirname(_agent_data_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.base import BaseGuardEvaluator

# =============================================================================
# 注册表
# =============================================================================

_EVALUATOR_REGISTRY = {}


def register_evaluator(name: str, evaluator_class):
    """注册 Guard 模型评估器"""
    _EVALUATOR_REGISTRY[name.lower()] = evaluator_class


def get_evaluator(name: str, **kwargs):
    """
    获取 Guard 模型评估器实例

    Args:
        name: 评估器名称 ('agentdog', ...)
        **kwargs: 传递给评估器构造函数的参数
    """
    name = name.lower()
    if name not in _EVALUATOR_REGISTRY:
        available = list(_EVALUATOR_REGISTRY.keys())
        raise ValueError(f"Unknown evaluator: {name}. Available: {available}")
    return _EVALUATOR_REGISTRY[name](**kwargs)


def list_evaluators():
    """列出所有已注册的评估器"""
    return list(_EVALUATOR_REGISTRY.keys())


# =============================================================================
# 导入并注册 AgentDoG 评估器
# =============================================================================

from .agentdog import AgentDoGEvaluator

register_evaluator('agentdog', AgentDoGEvaluator)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'BaseGuardEvaluator',
    'AgentDoGEvaluator',
    'get_evaluator',
    'register_evaluator',
    'list_evaluators',
]
