#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用执行器（General）

基于 data_types.py 合同 (MathSample) 的通用检查器，不依赖任何数据集特有逻辑。

=== 设计分层 ===

1. 合同层（GeneralFormatChecker — 本模块提供）
   检查数据是否符合 MathSample dataclass 定义：
   字段存在性、类型正确性、值非空等。
   假定所有数据已按 data_types.py 合同整理好，不关心字段内部的文本格式。
   适用于任何已通过 Loader 转换为 MathSample 的数据，或用户按合同自建的数据。

2. 内容质量层（dataset-specific 组件）
   验证 solution 的内容是否正确——即 solution 里声称的答案/代码是否与 ground_truth 一致。
   这需要解析 solution 内部的文本格式（\\boxed{}, <llm-code>, #### 等），
   而 MathSample 合同只定义了 solution 是 str/List[str]，不规定内部格式。
   因此 CodeExtractor、AnswerExtractor、CodeExecutor 属于此层，必须 dataset-specific。

3. 结果比较（已有通用实现）
   code_executor.compare_results() 和 compare_math_answers() 已支持
   字符串/数值/SymPy 符号/LaTeX 等价比较，无需额外 general 版本。
"""

from .GeneralFormatChecker import GeneralFormatChecker

__all__ = [
    'GeneralFormatChecker',
]
