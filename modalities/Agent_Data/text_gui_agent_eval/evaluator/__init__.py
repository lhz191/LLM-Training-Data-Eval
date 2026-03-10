#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard Model Evaluator 包 — Text GUI Agent 版本

支持的 Guard 模型:
- AgentDoG: 上海人工智能实验室的 Agent 安全诊断框架 (SOTA)
  只提取行为语义（action_type, target_element, action_repr），不传 HTML。
- LlamaGuard: Meta 的内容安全模型 (TODO)
- ShieldAgent: Agent 安全防护模型 (TODO)

使用方式：
    # 方式1：直接导入类
    from evaluator.agentdog import AgentDoGEvaluator
    
    evaluator = AgentDoGEvaluator(model_path='/path/to/model')
    result = evaluator.evaluate(record)
    
    # 方式2：通过工厂函数
    from evaluator import get_evaluator
    
    evaluator = get_evaluator('agentdog', model_path='/path/to/model')
"""

import os
import sys

_current_file = os.path.abspath(__file__)
_evaluator_dir = os.path.dirname(_current_file)
_text_gui_eval_dir = os.path.dirname(_evaluator_dir)

if _text_gui_eval_dir not in sys.path:
    sys.path.insert(0, _text_gui_eval_dir)

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_text_gui_eval_dir)))
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
