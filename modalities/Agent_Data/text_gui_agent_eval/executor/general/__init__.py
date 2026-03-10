#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用执行器（General）

基于 data_types.py 合同 (Record / Action) 的通用检查器，不依赖任何数据集特有逻辑。

提供:
  - GeneralFormatChecker（格式检查）
    纯粹基于 dataclass 字段定义进行检查：
    字段是否存在、类型是否正确、action_idx 是否连续/无重复等。
    适用于任何已通过 Loader 转换为 Record 的数据，或用户按合同自建的数据。

不提供通用版本的组件（及原因）:

  - StaticExecutabilityChecker（静态可执行性）
    验证 Action 能否在静态快照上被定位和执行。
    各数据集的"快照"格式完全不同：
      Mind2Web: MHTML 文件，需 Playwright 渲染，通过 bbox 坐标 / backend_node_id 定位
      WebShop:  纯文本 state，通过 [button] xxx [button_] 模式匹配 available_actions
      WebLINX:  HTML 字符串，通过 data-webtasks-id 属性定位
    无法用统一逻辑覆盖，必须由各数据集专用 Checker 实现。

  - DynamicExecutabilityChecker（动态可执行性）
    在真实网站 / 模拟环境上实际执行操作，验证是否成功。
    依赖数据集特有的运行时环境（Playwright 浏览器、WebShop 模拟器等），
    不存在通用的执行环境。

  - HTMLLocator（HTML 元素定位）
    在 raw_html / cleaned_html 中定位目标元素，用于计算信息保留率。
    各数据集的元素标识方式不同（backend_node_id / data-webtasks-id / 文本模式），
    通用字符串匹配会产生大量假阳性/假阴性，不具备实际评估价值。
"""

from .GeneralFormatChecker import GeneralFormatChecker

__all__ = [
    'GeneralFormatChecker',
]
