#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop 数据集执行器

包含:
- WebShopStaticChecker: 静态可执行性检查器
- WebShopFormatChecker: 格式检查器
- WebShopLocator: HTML 定位器

WebShop 数据特点：
- 基于 text 或 browser 模式验证
- 执行 action 序列并获取 reward
- 支持 search 和 click 两种 action 类型
"""

# 导入检查器类
from .WebShopStaticChecker import WebShopStaticChecker
from .WebShopFormatChecker import WebShopFormatChecker
from .WebShopLocator import WebShopLocator

# 导入常量
from .constants import DEFAULT_SERVER_URL

# 导入工具函数
from .utils import (
    _remove_price_constraint,
    check_server_running,
    start_server_if_needed,
)

__all__ = [
    # 检查器
    'WebShopStaticChecker',
    'WebShopFormatChecker',
    'WebShopLocator',
    # 常量
    'DEFAULT_SERVER_URL',
    # 工具函数
    '_remove_price_constraint',
    'check_server_running',
    'start_server_if_needed',
]
