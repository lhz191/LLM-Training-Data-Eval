#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard Model Evaluator 包

支持的 Guard 模型:
- AgentDoG: 上海 AI Lab 的 Agent 安全诊断框架，细粒度分类 (已实现)
- ShieldAgent: 浙大/蚂蚁，多阶段 Agent 防护 (TODO)
- WebGuard: 专门 Web Agent 安全，网页操作场景风险分类 (TODO)
- LlamaGuard: Meta 通用内容安全模型，可做 baseline (TODO)
- WildGuard: Allen AI 通用 safety，adversarial prompt 检测 (TODO)
- Qwen3Guard: 阿里通用安全模型 (TODO)

使用方式：
    # 方式1：直接导入类
    from evaluator.agentdog import AgentDoGEvaluator
    
    evaluator = AgentDoGEvaluator(model_path='/path/to/model')
    result = evaluator.evaluate(sample)
    
    # 方式2：通过工厂函数
    from evaluator import get_evaluator
    
    evaluator = get_evaluator('agentdog', model_path='/path/to/model')
"""

import os
import sys

# 使用绝对路径，避免 importlib 导入时的相对路径问题
_current_file = os.path.abspath(__file__)
_evaluator_dir = os.path.dirname(_current_file)
_api_agent_eval_dir = os.path.dirname(_evaluator_dir)

# 确保 api_agent_eval 目录在 path 中（用于导入 data_types, loaders 等）
if _api_agent_eval_dir not in sys.path:
    sys.path.insert(0, _api_agent_eval_dir)

# 添加项目根目录（用于导入 common 模块）
# api_agent_eval -> Agent_Data -> modalities -> main_new
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_api_agent_eval_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入基类
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
        name: 评估器名称 ('agentdog', 'llamaguard', ...)
        **kwargs: 传递给评估器构造函数的参数
        
    Returns:
        评估器实例
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
    # 基类
    'BaseGuardEvaluator',
    # 评估器
    'AgentDoGEvaluator',
    # 工厂函数
    'get_evaluator',
    'register_evaluator',
    'list_evaluators',
]

