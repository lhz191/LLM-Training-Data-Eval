#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-based GUI Agent 执行器 - 基础模块

与单轮 text_gui_executor.py 对应，但操作对象是 Session 而非 Record。
基类继承自 common.base，保持项目统一的类型体系。

本文件包含：
- Session 级别的基类定义（继承 common.base）
- 工厂函数和注册表

使用方式:
    checker = get_session_format_checker('weblinx')
    errors, warnings, stats = checker.check_session(session)
"""
import sys
import os
from abc import abstractmethod
from typing import List, Dict, Any, Tuple

_current_file = os.path.abspath(__file__)
_executor_dir = os.path.dirname(_current_file)
_mt_eval_dir = os.path.dirname(_executor_dir)
_agent_data_dir = os.path.dirname(_mt_eval_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)

_project_root = os.path.dirname(os.path.dirname(_agent_data_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.base import BaseFormatChecker, BaseExecutabilityChecker, BaseDynamicChecker
from data_types import Session


# =============================================================================
# Session 级别基类
# =============================================================================

class SessionFormatChecker(BaseFormatChecker):
    """
    多轮数据格式检查器基类

    继承自 common.base.BaseFormatChecker。
    对整个 Session 进行格式检查，包括：
    - Session 级别检查（rounds 非空、session_id 存在等）
    - 每轮 Record 级别检查
    - 跨轮一致性检查（如 website 一致性）
    """

    @property
    def modality(self) -> str:
        return 'gui'

    @property
    def checker_type(self) -> str:
        return 'format_check'

    def check(self, data) -> Tuple[List[str], List[str]]:
        """BaseFormatChecker 要求的接口，委托给 check_session"""
        errors, warnings, _ = self.check_session(data)
        return errors, warnings

    @abstractmethod
    def check_session(self, session: Session) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查 Session 的数据格式

        Args:
            session: 多轮会话

        Returns:
            (errors, warnings, stats) 元组
        """
        pass


class SessionStaticChecker(BaseExecutabilityChecker):
    """
    多轮静态可执行性检查器基类

    继承自 common.base.BaseExecutabilityChecker。
    逐轮验证 Action 是否可以在静态 HTML 快照上被定位和执行。
    """

    @property
    def modality(self) -> str:
        return 'gui'

    @property
    def checker_type(self) -> str:
        return 'static_executability'

    def check(self, data) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """BaseExecutabilityChecker 要求的接口，委托给 check_session"""
        return self.check_session(data)

    @abstractmethod
    def check_session(self, session: Session, execute: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查 Session 所有轮次的静态可执行性

        Args:
            session: 多轮会话
            execute: 是否执行操作

        Returns:
            (errors, warnings, stats) 元组
        """
        pass


class SessionDynamicChecker(BaseDynamicChecker):
    """
    多轮动态可执行性检查器基类

    继承自 common.base.BaseDynamicChecker。
    在真实网站上顺序执行 Session 的所有轮次 Action。
    浏览器状态在轮次间保持（跟真实多轮交互一致）。
    """

    @property
    def modality(self) -> str:
        return 'gui'

    @property
    def checker_type(self) -> str:
        return 'dynamic_executability'

    def check(self, data) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """BaseDynamicChecker 要求的接口，委托给 check_session"""
        return self.check_session(data)

    @abstractmethod
    def check_session(self, session: Session, execute: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        在真实网站上验证并执行 Session 的 action 序列

        Args:
            session: 多轮会话
            execute: 是否执行操作

        Returns:
            (errors, warnings, stats) 元组
        """
        pass


# =============================================================================
# 注册表
# =============================================================================

SESSION_FORMAT_CHECKERS = {}
SESSION_STATIC_CHECKERS = {}
SESSION_DYNAMIC_CHECKERS = {}


def register_session_format_checker(name: str, checker_class: type):
    SESSION_FORMAT_CHECKERS[name.lower()] = checker_class


def register_session_static_checker(name: str, checker_class: type):
    SESSION_STATIC_CHECKERS[name.lower()] = checker_class


def register_session_dynamic_checker(name: str, checker_class: type):
    SESSION_DYNAMIC_CHECKERS[name.lower()] = checker_class


# =============================================================================
# 工厂函数
# =============================================================================

def get_session_format_checker(dataset_name: str, **kwargs) -> SessionFormatChecker:
    name = dataset_name.lower()
    if name not in SESSION_FORMAT_CHECKERS:
        available = list(SESSION_FORMAT_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return SESSION_FORMAT_CHECKERS[name](**kwargs)


def get_session_static_checker(dataset_name: str, **kwargs) -> SessionStaticChecker:
    name = dataset_name.lower()
    if name not in SESSION_STATIC_CHECKERS:
        available = list(SESSION_STATIC_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return SESSION_STATIC_CHECKERS[name](**kwargs)


def get_session_dynamic_checker(dataset_name: str, **kwargs) -> SessionDynamicChecker:
    name = dataset_name.lower()
    if name not in SESSION_DYNAMIC_CHECKERS:
        available = list(SESSION_DYNAMIC_CHECKERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return SESSION_DYNAMIC_CHECKERS[name](**kwargs)


# =============================================================================
# 列出可用检查器
# =============================================================================

def list_session_format_checkers() -> List[str]:
    return list(SESSION_FORMAT_CHECKERS.keys())


def list_session_static_checkers() -> List[str]:
    return list(SESSION_STATIC_CHECKERS.keys())


def list_session_dynamic_checkers() -> List[str]:
    return list(SESSION_DYNAMIC_CHECKERS.keys())
