#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - 统一注册表

提供跨模态的检查器注册和获取机制。

使用方式：
    # 注册检查器
    from common import register_checker
    
    @register_checker('api', 'format', 'toolbench')
    class ToolBenchFormatChecker(BaseFormatChecker):
        ...
    
    # 获取检查器
    from common import get_checker
    checker = get_checker('api', 'format', 'toolbench')
    
    # 列出所有检查器
    from common import list_checkers
    print(list_checkers())
"""

from typing import Dict, Type, Optional, List, Tuple, Any


# =============================================================================
# 全局注册表
# =============================================================================

# 三层字典: modality -> checker_type -> dataset -> checker_class
_REGISTRY: Dict[str, Dict[str, Dict[str, Type]]] = {}


# =============================================================================
# 注册函数
# =============================================================================

def register_checker(modality: str, checker_type: str, dataset: str):
    """
    注册检查器的装饰器
    
    Args:
        modality: 模态 ('api', 'gui', 'math', 'image', 'video')
        checker_type: 检查器类型 ('format', 'executability', 'dynamic', etc.)
        dataset: 数据集名称 ('toolbench', 'mind2web', 'lila', etc.)
    
    Usage:
        @register_checker('api', 'format', 'toolbench')
        class ToolBenchFormatChecker(BaseFormatChecker):
            ...
    """
    def decorator(cls: Type) -> Type:
        if modality not in _REGISTRY:
            _REGISTRY[modality] = {}
        if checker_type not in _REGISTRY[modality]:
            _REGISTRY[modality][checker_type] = {}
        
        _REGISTRY[modality][checker_type][dataset] = cls
        return cls
    
    return decorator


def register_checker_class(modality: str, checker_type: str, dataset: str, cls: Type):
    """
    直接注册检查器类（非装饰器方式）
    
    Args:
        modality: 模态
        checker_type: 检查器类型
        dataset: 数据集名称
        cls: 检查器类
    """
    if modality not in _REGISTRY:
        _REGISTRY[modality] = {}
    if checker_type not in _REGISTRY[modality]:
        _REGISTRY[modality][checker_type] = {}
    
    _REGISTRY[modality][checker_type][dataset] = cls


# =============================================================================
# 获取函数
# =============================================================================

def get_checker(modality: str, checker_type: str, dataset: str, **kwargs) -> Any:
    """
    获取检查器实例
    
    Args:
        modality: 模态
        checker_type: 检查器类型
        dataset: 数据集名称
        **kwargs: 传递给检查器构造函数的参数
    
    Returns:
        检查器实例
    
    Raises:
        ValueError: 如果找不到对应的检查器
    """
    if modality not in _REGISTRY:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    
    if checker_type not in _REGISTRY[modality]:
        raise ValueError(
            f"Unknown checker type '{checker_type}' for modality '{modality}'. "
            f"Available: {list(_REGISTRY[modality].keys())}"
        )
    
    if dataset not in _REGISTRY[modality][checker_type]:
        raise ValueError(
            f"Unknown dataset '{dataset}' for {modality}/{checker_type}. "
            f"Available: {list(_REGISTRY[modality][checker_type].keys())}"
        )
    
    checker_class = _REGISTRY[modality][checker_type][dataset]
    return checker_class(**kwargs)


def get_checker_class(modality: str, checker_type: str, dataset: str) -> Optional[Type]:
    """
    获取检查器类（不实例化）
    
    Returns:
        检查器类，如果不存在返回 None
    """
    try:
        return _REGISTRY[modality][checker_type][dataset]
    except KeyError:
        return None


# =============================================================================
# 查询函数
# =============================================================================

def list_modalities() -> List[str]:
    """列出所有已注册的模态"""
    return list(_REGISTRY.keys())


def list_checker_types(modality: str) -> List[str]:
    """列出某个模态下所有已注册的检查器类型"""
    if modality not in _REGISTRY:
        return []
    return list(_REGISTRY[modality].keys())


def list_datasets(modality: str, checker_type: str) -> List[str]:
    """列出某个模态/检查器类型下所有已注册的数据集"""
    try:
        return list(_REGISTRY[modality][checker_type].keys())
    except KeyError:
        return []


def list_checkers() -> Dict[str, Dict[str, List[str]]]:
    """
    列出所有已注册的检查器
    
    Returns:
        嵌套字典: {modality: {checker_type: [datasets]}}
    """
    result = {}
    for modality, types in _REGISTRY.items():
        result[modality] = {}
        for checker_type, datasets in types.items():
            result[modality][checker_type] = list(datasets.keys())
    return result


def print_registry():
    """打印注册表（调试用）"""
    print("=" * 60)
    print("LLM Training Data Evaluation - Registry")
    print("=" * 60)
    
    for modality in sorted(_REGISTRY.keys()):
        print(f"\n📁 {modality}")
        for checker_type in sorted(_REGISTRY[modality].keys()):
            print(f"  └─ {checker_type}")
            for dataset in sorted(_REGISTRY[modality][checker_type].keys()):
                cls = _REGISTRY[modality][checker_type][dataset]
                print(f"      └─ {dataset}: {cls.__name__}")
