#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-Text 数据集执行器基类和工厂函数

定义格式检查器的抽象基类，以及工厂函数。
具体的数据集实现在各自的 *_executor.py 文件中。
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from data_types import ImageTextSample


# ==================== 基类定义 ====================

class FormatChecker(ABC):
    """格式检查器基类"""
    
    @abstractmethod
    def check(self, sample: ImageTextSample) -> bool:
        """
        检查样本格式是否正确
        
        Args:
            sample: ImageTextSample 样本
        
        Returns:
            是否通过检查
        """
        pass


# ==================== 注册机制 ====================

_FORMAT_CHECKER_REGISTRY: Dict[str, Type[FormatChecker]] = {}


def register_format_checker(name: str):
    """注册格式检查器的装饰器"""
    def decorator(cls: Type[FormatChecker]):
        _FORMAT_CHECKER_REGISTRY[name] = cls
        return cls
    return decorator


def get_format_checker(dataset_name: str, **kwargs) -> FormatChecker:
    """
    根据数据集名称获取对应的格式检查器
    
    Args:
        dataset_name: 数据集名称
        **kwargs: 传递给检查器构造函数的参数
    
    Returns:
        FormatChecker 实例
    """
    if dataset_name not in _FORMAT_CHECKER_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(_FORMAT_CHECKER_REGISTRY.keys())}"
        )
    return _FORMAT_CHECKER_REGISTRY[dataset_name](**kwargs)


def list_format_checkers():
    """列出所有已注册的格式检查器"""
    return list(_FORMAT_CHECKER_REGISTRY.keys())
