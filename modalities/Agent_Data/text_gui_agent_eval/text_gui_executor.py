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
import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加 common 模块路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from data_types import Record

# 导入统一基类
from common.base import (
    BaseFormatChecker,
    BaseExecutabilityChecker,
    BaseDynamicChecker,
    BaseHTMLLocator,
)


# =============================================================================
# 基类
# =============================================================================

class StaticExecutabilityChecker(BaseExecutabilityChecker):
    """
    静态可执行性检查器基类（GUI Agent）
    
    继承自 common.base.BaseExecutabilityChecker
    
    用于验证 Record 中的 action 序列是否可以在静态 HTML 快照上执行。
    每个数据集实现自己的 check 方法。
    
    检查内容（因数据集而异）：
    - Mind2Web: 用 backend_node_id 或属性在 HTML 中定位元素
    - WebShop: 检查 action 是否在 available_actions 中
    - WebLINX: 用 data-webtasks-id 在 HTML 中定位元素
    """
    
    @property
    def modality(self) -> str:
        return 'gui'
    
    @property
    def checker_type(self) -> str:
        return 'static_executability'
    
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


class DynamicExecutabilityChecker(BaseDynamicChecker):
    """
    动态可执行性检查器基类（GUI Agent）
    
    继承自 common.base.BaseDynamicChecker
    
    用于在真实网站上执行 Record 中的 action 序列，验证是否可以成功执行。
    每个数据集实现自己的 check 方法。
    
    与静态检查的区别：
    - 静态检查：在 MHTML/HTML 快照上验证元素是否存在
    - 动态检查：在真实网站上实际执行操作（click, type 等）
    
    注意事项：
    - 网站可能已变化，数据集记录的操作可能失效
    - 需要处理登录、验证码、弹窗等干扰
    - 操作会产生真实副作用
    """
    
    @property
    def modality(self) -> str:
        return 'gui'
    
    @property
    def checker_type(self) -> str:
        return 'dynamic_executability'
    
    @abstractmethod
    def check(self, record: Record) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        在真实网站上执行 Record 的 action 序列
        
        Args:
            record: GUI Agent Record（包含 actions 列表）
            
        Returns:
            (errors, warnings, stats) 元组
            - errors: 错误列表（执行失败）
            - warnings: 警告列表（部分成功或需要注意）
            - stats: 统计信息，包含：
                - total_actions: 总 action 数
                - executed_actions: 成功执行的 action 数
                - failed_actions: 执行失败的 action 数
                - execution_rate: 执行成功率
                - action_results: 每个 action 的执行结果
        """
        pass


class FormatChecker(BaseFormatChecker):
    """
    数据格式检查器基类（GUI Agent）
    
    继承自 common.base.BaseFormatChecker
    
    用于验证 Record 的数据格式是否正确，包括：
    - 必需字段是否存在
    - 字段格式是否正确
    - 数据一致性检查（如 target 在 candidates 中存在）
    
    这是一个轻量级检查，不需要浏览器，速度快。
    """
    
    @property
    def modality(self) -> str:
        return 'gui'
    
    @property
    def checker_type(self) -> str:
        return 'format_check'
    
    @abstractmethod
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """
        检查单个 Record 的数据格式
        
        Args:
            record: GUI Agent Record
            
        Returns:
            (errors, warnings) 元组
            - errors: 错误列表（格式问题，必须修复）
            - warnings: 警告列表（建议修复但不影响使用）
        """
        pass


class HTMLLocator(BaseHTMLLocator):
    """
    HTML 定位器基类（GUI Agent）
    
    继承自 common.base.BaseHTMLLocator
    
    用于检查是否能在 HTML 中定位到目标元素，计算信息保留率：
    - raw_html 定位成功率
    - cleaned_html 定位成功率  
    - 保留率 = cleaned_html 成功数 / raw_html 成功数
    
    每个数据集的定位方式不同：
    - Mind2Web: 通过 backend_node_id（如 backend_node_id="136"）
    - WebShop: 通过 [button] xxx [button_] 模式
    - WebLINX: 通过 data-webtasks-id（如 data-webtasks-id="xxx"）
    """
    
    @property
    def modality(self) -> str:
        return 'gui'
    
    @property
    def checker_type(self) -> str:
        return 'html_locator'
    
    @abstractmethod
    def can_locate(self, action, html: str) -> Tuple[bool, str]:
        """
        检查是否能在 HTML 中定位到目标元素
        
        Args:
            action: Action 对象
            html: HTML 字符串（可以是 raw_html 或 cleaned_html）
            
        Returns:
            (success, reason) 元组
            - success: 是否定位成功
            - reason: 原因说明（如 "found", "not_found", "no_target"）
        """
        pass


# =============================================================================
# 注册表
# =============================================================================

STATIC_CHECKERS = {}
DYNAMIC_CHECKERS = {}
FORMAT_CHECKERS = {}
HTML_LOCATORS = {}


# =============================================================================
# 注册函数
# =============================================================================

def register_static_checker(name: str, checker_class: type):
    """注册静态可执行性检查器"""
    STATIC_CHECKERS[name.lower()] = checker_class


def register_dynamic_checker(name: str, checker_class: type):
    """注册动态可执行性检查器"""
    DYNAMIC_CHECKERS[name.lower()] = checker_class


def register_format_checker(name: str, checker_class: type):
    """注册格式检查器"""
    FORMAT_CHECKERS[name.lower()] = checker_class


def register_html_locator(name: str, locator_class: type):
    """注册 HTML 定位器"""
    HTML_LOCATORS[name.lower()] = locator_class


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


def get_dynamic_checker(dataset_name: str, **kwargs) -> DynamicExecutabilityChecker:
    """
    获取指定数据集的动态可执行性检查器
    
    Args:
        dataset_name: 数据集名称 (mind2web, webshop, weblinx)
        **kwargs: 传递给检查器的额外参数
                  - headless: 是否使用无头浏览器模式
                  - timeout: 页面加载超时时间
        
    Returns:
        动态可执行性检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in DYNAMIC_CHECKERS:
        available = list(DYNAMIC_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return DYNAMIC_CHECKERS[name_lower](**kwargs)


def get_format_checker(dataset_name: str, **kwargs) -> FormatChecker:
    """
    获取指定数据集的格式检查器
    
    Args:
        dataset_name: 数据集名称 (mind2web, webshop, weblinx)
        **kwargs: 传递给检查器的额外参数
        
    Returns:
        格式检查器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in FORMAT_CHECKERS:
        available = list(FORMAT_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return FORMAT_CHECKERS[name_lower](**kwargs)


def get_html_locator(dataset_name: str, **kwargs) -> HTMLLocator:
    """
    获取指定数据集的 HTML 定位器
    
    Args:
        dataset_name: 数据集名称 (mind2web, webshop, weblinx)
        **kwargs: 传递给定位器的额外参数
        
    Returns:
        HTML 定位器实例
    """
    name_lower = dataset_name.lower().replace('_', '-').replace(' ', '')
    
    if name_lower not in HTML_LOCATORS:
        available = list(HTML_LOCATORS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return HTML_LOCATORS[name_lower](**kwargs)


# =============================================================================
# 列出可用检查器
# =============================================================================

def list_static_checkers() -> List[str]:
    """列出所有可用的静态可执行性检查器"""
    return list(STATIC_CHECKERS.keys())


def list_dynamic_checkers() -> List[str]:
    """列出所有可用的动态可执行性检查器"""
    return list(DYNAMIC_CHECKERS.keys())


def list_format_checkers() -> List[str]:
    """列出所有可用的格式检查器"""
    return list(FORMAT_CHECKERS.keys())


def list_html_locators() -> List[str]:
    """列出所有可用的 HTML 定位器"""
    return list(HTML_LOCATORS.keys())


# =============================================================================
# 自动导入子模块以注册检查器
# =============================================================================

def _auto_import_submodules():
    """自动导入子模块，触发它们的注册逻辑"""
    # 所有检查器已迁移到 executor/
    try:
        import executor
    except ImportError:
        pass


# 在模块加载时自动导入
_auto_import_submodules()
