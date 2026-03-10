#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮执行器包 (executor)

按数据集组织的 Session 级别执行器模块：
- executor.weblinx: WebLINX Session 级别检查器
- executor.general: 通用 Session 级别检查器

使用方式：
    from mt_text_gui_agent_eval.executor.weblinx import WebLINXSessionFormatChecker
    from mt_text_gui_agent_eval.executor.general import GeneralSessionFormatChecker

    # 或通过注册表
    from mt_text_gui_agent_eval.executor.mt_executor import get_session_format_checker
    checker = get_session_format_checker('weblinx')
"""

from mt_executor import (
    SessionFormatChecker,
    SessionStaticChecker,
    SessionDynamicChecker,
    register_session_format_checker,
    register_session_static_checker,
    register_session_dynamic_checker,
    get_session_format_checker,
    get_session_static_checker,
    get_session_dynamic_checker,
)


# =============================================================================
# 导入并注册 WebLINX 检查器
# =============================================================================

from .weblinx import (
    WebLINXSessionFormatChecker,
    WebLINXSessionStaticChecker,
)

register_session_format_checker('weblinx', WebLINXSessionFormatChecker)
register_session_static_checker('weblinx', WebLINXSessionStaticChecker)


# =============================================================================
# 导入并注册 General 检查器
# =============================================================================

from .general import GeneralSessionFormatChecker

register_session_format_checker('general', GeneralSessionFormatChecker)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 基类
    'SessionFormatChecker',
    'SessionStaticChecker',
    'SessionDynamicChecker',
    # WebLINX
    'WebLINXSessionFormatChecker',
    'WebLINXSessionStaticChecker',
    # General
    'GeneralSessionFormatChecker',
    # 工厂函数
    'get_session_format_checker',
    'get_session_static_checker',
    'get_session_dynamic_checker',
]
