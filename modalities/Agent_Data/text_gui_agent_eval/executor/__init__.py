#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行器包 (executor)

按数据集组织的执行器模块：
- executor.weblinx: WebLINX 数据集执行器
- executor.mind2web: Mind2Web 数据集执行器
- executor.webshop: WebShop 数据集执行器
- executor.general: 通用执行器（基于 data_types.py 合同，无数据集特有逻辑）

使用方式：
    # 方式1：直接导入类
    from executor.weblinx import WebLINXStaticChecker
    from executor.mind2web import Mind2WebStaticChecker
    from executor.webshop import WebShopStaticChecker
    from executor.general import GeneralFormatChecker, GeneralStaticChecker
    
    # 方式2：通过全局注册表
    from text_gui_executor import get_static_checker, get_format_checker
    checker = get_static_checker('general')
"""

import sys
import os

# 确保父目录在 path 中（用于导入 text_gui_executor）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from text_gui_executor import (
    register_static_checker,
    register_dynamic_checker,
    register_format_checker,
    register_html_locator,
)


# =============================================================================
# 导入并注册 WebLINX 检查器
# =============================================================================

from .weblinx import (
    WebLINXStaticChecker,
    WebLINXFormatChecker,
    WebLINXLocator,
)

register_static_checker('weblinx', WebLINXStaticChecker)
register_format_checker('weblinx', WebLINXFormatChecker)
register_html_locator('weblinx', WebLINXLocator)


# =============================================================================
# 导入并注册 Mind2Web 检查器
# =============================================================================

from .mind2web import (
    Mind2WebStaticChecker,
    Mind2WebDynamicChecker,
    Mind2WebFormatChecker,
    Mind2WebLocator,
)

register_static_checker('mind2web', Mind2WebStaticChecker)
register_dynamic_checker('mind2web', Mind2WebDynamicChecker)
register_format_checker('mind2web', Mind2WebFormatChecker)
register_html_locator('mind2web', Mind2WebLocator)


# =============================================================================
# 导入并注册 WebShop 检查器
# =============================================================================

from .webshop import (
    WebShopStaticChecker,
    WebShopFormatChecker,
    WebShopLocator,
)

register_static_checker('webshop', WebShopStaticChecker)
register_format_checker('webshop', WebShopFormatChecker)
register_html_locator('webshop', WebShopLocator)


# =============================================================================
# 导入并注册 General 检查器
# =============================================================================

from .general import (
    GeneralFormatChecker,
)

register_format_checker('general', GeneralFormatChecker)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # WebLINX
    'WebLINXStaticChecker',
    'WebLINXFormatChecker',
    'WebLINXLocator',
    # Mind2Web
    'Mind2WebStaticChecker',
    'Mind2WebDynamicChecker',
    'Mind2WebFormatChecker',
    'Mind2WebLocator',
    # WebShop
    'WebShopStaticChecker',
    'WebShopFormatChecker',
    'WebShopLocator',
    # General
    'GeneralFormatChecker',
]
