#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent 执行器 - 基础模块

支持从不同格式的 API Agent 数据中进行格式检查和可执行性检查。
设计为可扩展架构，支持不同数据集的定制化。

本文件包含：
- 基类定义（FormatChecker, ExecutabilityChecker, DynamicChecker）
- 工厂函数和注册表

具体实现：
- ToolBench 相关: toolbench_executor.py
- xLAM 相关: xlam_executor.py

使用方式:
    # 获取格式检查器
    format_checker = get_format_checker('toolbench')
    errors, warnings = format_checker.check(sample)
    
    # 获取可执行性检查器
    exec_checker = get_executability_checker('toolbench', toolenv_path='/path/to/toolenv')
    errors, warnings, stats = exec_checker.check(sample)
    
    # 获取动态检查器
    dynamic_checker = get_dynamic_checker('toolbench', rapidapi_key='xxx')
    result = dynamic_checker.check_sample(sample)
"""
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from data_types import APIAgentSample


# =============================================================================
# 基类
# =============================================================================

class FormatChecker(ABC):
    """
    格式检查器基类
    
    每个数据集实现自己的 check 方法，返回 (errors, warnings) 元组。
    """
    
    @abstractmethod
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str]]:
        """
        检查样本的格式正确性
        
        Args:
            sample: API Agent 样本
            
        Returns:
            (errors, warnings) 元组
            - errors: 错误列表（严重问题，影响数据可用性）
            - warnings: 警告列表（轻微问题，不影响数据可用性）
        """
        pass


class ExecutabilityChecker(ABC):
    """
    可执行性检查器基类（静态检查）
    
    每个数据集实现自己的 check 方法。
    """
    
    @abstractmethod
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查样本的可执行性
        
        Args:
            sample: API Agent 样本
            
        Returns:
            (errors, warnings, stats) 元组
            - errors: 错误列表
            - warnings: 警告列表
            - stats: 统计信息（如 derivability 结果等）
        """
        pass


class DynamicChecker(ABC):
    """
    动态可执行性检查器基类
    
    通过实际调用 API 来验证可执行性。
    """
    
    @abstractmethod
    def check_sample(self, sample: APIAgentSample) -> Dict[str, Any]:
        """
        检查单个样本的动态可执行性
        
        Args:
            sample: API Agent 样本
            
        Returns:
            检查结果字典，包含：
            - sample_id: 样本 ID
            - passed: 是否通过
            - api_results: 各 API 调用结果
            - errors: 错误列表
        """
        pass


# =============================================================================
# 注册表
# =============================================================================

FORMAT_CHECKERS = {}
EXECUTABILITY_CHECKERS = {}
DYNAMIC_CHECKERS = {}


# =============================================================================
# 注册函数
# =============================================================================

def register_format_checker(name: str, checker_class: type):
    """注册格式检查器"""
    FORMAT_CHECKERS[name.lower()] = checker_class


def register_executability_checker(name: str, checker_class: type):
    """注册可执行性检查器"""
    EXECUTABILITY_CHECKERS[name.lower()] = checker_class


def register_dynamic_checker(name: str, checker_class: type):
    """注册动态检查器"""
    DYNAMIC_CHECKERS[name.lower()] = checker_class


# =============================================================================
# 工厂函数（获取检查器实例）
# =============================================================================

def get_format_checker(dataset_name: str) -> FormatChecker:
    """
    获取指定数据集的格式检查器
    
    Args:
        dataset_name: 数据集名称 (toolbench, xlam, xlam-60k)
        
    Returns:
        格式检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in FORMAT_CHECKERS:
        available = list(FORMAT_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return FORMAT_CHECKERS[name_lower]()


def get_executability_checker(dataset_name: str, **kwargs) -> ExecutabilityChecker:
    """
    获取指定数据集的可执行性检查器
    
    Args:
        dataset_name: 数据集名称 (toolbench, xlam, xlam-60k)
        **kwargs: 传递给检查器的额外参数
                  - toolenv_path: ToolBench 需要的 toolenv 路径
                  - cache_dir: 缓存目录
        
    Returns:
        可执行性检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in EXECUTABILITY_CHECKERS:
        available = list(EXECUTABILITY_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return EXECUTABILITY_CHECKERS[name_lower](**kwargs)


def get_dynamic_checker(dataset_name: str, **kwargs) -> DynamicChecker:
    """
    获取指定数据集的动态检查器
    
    Args:
        dataset_name: 数据集名称 (toolbench)
        **kwargs: 传递给检查器的额外参数
                  - rapidapi_key: RapidAPI Key
                  - toolenv_path: ToolBench 需要的 toolenv 路径
                  - cache_dir: 缓存目录
        
    Returns:
        动态检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in DYNAMIC_CHECKERS:
        available = list(DYNAMIC_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return DYNAMIC_CHECKERS[name_lower](**kwargs)


# =============================================================================
# 列出可用检查器
# =============================================================================

def list_format_checkers() -> List[str]:
    """列出所有可用的格式检查器"""
    return list(FORMAT_CHECKERS.keys())


def list_executability_checkers() -> List[str]:
    """列出所有可用的可执行性检查器"""
    return list(EXECUTABILITY_CHECKERS.keys())


def list_dynamic_checkers() -> List[str]:
    """列出所有可用的动态检查器"""
    return list(DYNAMIC_CHECKERS.keys())


# =============================================================================
# 自动导入子模块以注册检查器
# =============================================================================

def _auto_import_submodules():
    """自动导入子模块，触发它们的注册逻辑"""
    try:
        import toolbench_executor
    except ImportError:
        pass
    
    try:
        import xlam_executor
    except ImportError:
        pass


# 在模块加载时自动导入
_auto_import_submodules()
