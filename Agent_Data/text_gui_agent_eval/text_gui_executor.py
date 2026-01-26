#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-based GUI Agent 执行器 - 基础模块

支持对 Mind2Web、WebShop、WebLINX 等数据集进行静态可执行性检查。
设计为可扩展架构，支持不同数据集的定制化。

本文件包含：
- 基类定义（StaticExecutabilityChecker）
- 工厂函数和注册表

具体实现：
- Mind2Web 相关: mind2web_executor.py
- WebShop 相关: webshop_executor.py
- WebLINX 相关: weblinx_executor.py

使用方式:
    # 获取静态可执行性检查器
    checker = get_static_checker('mind2web')
    result = checker.check(record)
"""
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from data_types import Record


# =============================================================================
# 基类
# =============================================================================

class StaticExecutabilityChecker(ABC):
    """
    静态可执行性检查器基类
    
    用于验证 Record 中的 action 序列是否可以在静态 HTML 快照上执行。
    每个数据集实现自己的 check 方法。
    
    检查内容（因数据集而异）：
    - Mind2Web: 用 backend_node_id 或属性在 HTML 中定位元素
    - WebShop: 检查 action 是否在 available_actions 中
    - WebLINX: 用 data-webtasks-id 在 HTML 中定位元素
    """
    
    @abstractmethod
    def check(self, record: Record) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查单个 Record 的静态可执行性
        
        Args:
            record: GUI Agent Record（包含 actions 列表）
            
        Returns:
            (errors, warnings, stats) 元组
            - errors: 错误列表（严重问题，如元素找不到）
            - warnings: 警告列表（轻微问题，不影响可用性）
            - stats: 统计信息，包含：
                - total_actions: 总 action 数
                - success_count: 成功定位/验证的 action 数
                - failed_count: 失败的 action 数
                - success_rate: 成功率
                - action_results: 每个 action 的详细结果
        """
        pass


# =============================================================================
# 注册表
# =============================================================================

STATIC_CHECKERS = {}


# =============================================================================
# 注册函数
# =============================================================================

def register_static_checker(name: str, checker_class: type):
    """注册静态可执行性检查器"""
    STATIC_CHECKERS[name.lower()] = checker_class


# =============================================================================
# 工厂函数
# =============================================================================

def get_static_checker(dataset_name: str, **kwargs) -> StaticExecutabilityChecker:
    """
    获取指定数据集的静态可执行性检查器
    
    Args:
        dataset_name: 数据集名称 (mind2web, webshop, weblinx)
        **kwargs: 传递给检查器的额外参数
                  - headless: 是否使用无头浏览器模式 (Playwright)
                  - use_attrs_mode: Mind2Web 是否使用属性定位模式
        
    Returns:
        静态可执行性检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in STATIC_CHECKERS:
        available = list(STATIC_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return STATIC_CHECKERS[name_lower](**kwargs)


# =============================================================================
# 列出可用检查器
# =============================================================================

def list_static_checkers() -> List[str]:
    """列出所有可用的静态可执行性检查器"""
    return list(STATIC_CHECKERS.keys())


# =============================================================================
# 自动导入子模块以注册检查器
# =============================================================================

def _auto_import_submodules():
    """自动导入子模块，触发它们的注册逻辑"""
    try:
        import mind2web_executor
    except ImportError:
        pass
    
    try:
        import webshop_executor
    except ImportError:
        pass
    
    try:
        import weblinx_executor
    except ImportError:
        pass


# 在模块加载时自动导入
_auto_import_submodules()
