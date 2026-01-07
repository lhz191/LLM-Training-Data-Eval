#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Agent Data Evaluation - 数据类型定义

统一的 API Agent 数据格式，支持：
- 虚拟 API 数据集：xLAM-60k（单轮调用，只有工具定义，无真实 API 响应）
- 真实 API 数据集：ToolBench（多轮对话，有真实 API 调用和响应）

设计原则：
1. 字段定义参照真实数据条目
2. 提供统一接口，不同数据集通过 Loader 转换为统一格式
3. 支持从原始数据中提取 API 调用信息（而非直接存储 conversation）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union


# =============================================================================
# 参数定义
# =============================================================================

@dataclass
class Parameter:
    """
    工具参数定义（统一格式）
    
    统一 ToolBench 和 xLAM-60k 两种格式：
    
    ToolBench 原始格式：
    {
        "type": "object",
        "properties": {
            "username": {"type": "string", "description": "...", "example_value": "nike"}
        },
        "required": ["username"],
        "optional": []
    }
    
    xLAM-60k 原始格式：
    {
        "is_id": {"type": "int", "description": "...", "default": "..."},
        "locale": {"type": "str, optional", "description": "...", "default": ""}
    }
    
    统一为：
    必需字段：
    - name: 参数名
    - type: 参数类型 (str, int, bool, etc.)
    
    可选字段：
    - description: 参数描述
    - default: 默认值
    - optional: 是否可选参数
    - metadata: 其他元信息（如 ToolBench 的 example_value）
    """
    # === 必需字段 ===
    name: str                           # 参数名
    type: str                           # 参数类型 (str, int, bool, float, list, dict, etc.)
    
    # === 可选字段 ===
    description: str = ""               # 参数描述
    default: Any = None                 # 默认值
    optional: bool = False              # 是否可选参数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元信息（如 example_value）


# =============================================================================
# 工具/API 定义
# =============================================================================

@dataclass
class ToolDefinition:
    """
    工具/API 定义
    
    支持 ToolBench 和 xLAM-60k 两种数据集格式。
    参数统一转换为 Parameter 列表。
    """
    name: str                           # 工具名称
    description: str = ""               # 工具描述
    parameters: List[Parameter] = field(default_factory=list)  # 参数列表
    
    def get_required_params(self) -> List[str]:
        """获取必需参数名列表"""
        return [p.name for p in self.parameters if not p.optional]
    
    def get_optional_params(self) -> List[str]:
        """获取可选参数名列表"""
        return [p.name for p in self.parameters if p.optional]
    
    def get_all_param_names(self) -> List[str]:
        """获取所有参数名列表"""
        return [p.name for p in self.parameters]
    
    def get_param(self, param_name: str) -> Optional[Parameter]:
        """根据名称获取参数定义"""
        for p in self.parameters:
            if p.name == param_name:
                return p
        return None


# =============================================================================
# API 调用
# =============================================================================

@dataclass
class APICall:
    """
    单次 API 调用
    
    对应 ToolBench 中的 Action + Action Input：
        Thought: ...
        Action: userinfo_for_instagram_cheapest
        Action Input: {"username": "nike"}
    
    也对应 xLAM-60k 的 answers 字段中的每个调用。
    
    必需字段：
    - name: 调用的 API/工具名称
    
    可选字段：
    - arguments: 调用参数
    - response: API 响应（原始字符串，格式为 {"error": "...", "response": "..."}）
    - metadata: 其他元信息（如 ToolBench 的 thought）
    
    注意：response 存储原始字符串，判断是否失败时用关键词搜索（timeout, exception 等）。
    """
    # === 必需字段 ===
    name: str                           # 调用的 API/工具名称
    
    # === 可选字段 ===
    arguments: Dict[str, Any] = field(default_factory=dict)  # 调用参数
    response: Optional[str] = None      # API 响应（原始字符串）
    metadata: Dict[str, Any] = field(default_factory=dict)   # 其他元信息（如 thought）


# =============================================================================
# 统一数据样本
# =============================================================================

@dataclass
class APIAgentSample:
    """
    API Agent 数据的统一输入格式
    
    支持两种数据集格式：
    
    1. xLAM-60k（单轮）:
       - query: 用户查询
       - tools: 可用工具列表（从 tools 字段解析）
       - api_calls: 单次或多次并行调用（从 answers 字段解析）
       - final_answer: 无（xLAM 没有显式的最终答案）
    
    2. ToolBench（多轮对话）:
       - query: 从 id 字段提取（Step N: 后面的内容）
       - tools: 从 system prompt 解析（Specifically, you have access to the following APIs: [...]）
       - api_calls: 从每个 assistant 轮次解析（Action + Action Input）
       - final_answer: 从 Action: Finish 的 final_answer 参数提取
    
    注意：不直接存储 conversation，而是解析后存储结构化字段。
    """
    
    # === 必需字段 ===
    query: str                          # 用户查询/指令
    tools: List[ToolDefinition]         # 可用工具列表（API 定义）
    api_calls: List[APICall]            # API 调用列表（GT 或模型预测）
    
    # === 可选字段 ===
    final_answer: Optional[str] = None  # 最终答案（ToolBench 从 Finish 提取，xLAM 无）
    
    # === 元信息 ===
    sample_id: Optional[str] = None     # 样本唯一标识
    source_dataset: Optional[str] = None  # 来源数据集名称 (xlam_60k, toolbench, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元信息（如 ToolBench 的 step_number）
    
    def __repr__(self):
        q_preview = self.query[:50] + "..." if len(self.query) > 50 else self.query
        return f"APIAgentSample(query='{q_preview}', tools={len(self.tools)}, calls={len(self.api_calls)})"
    
    def get_tool_names(self) -> List[str]:
        """获取所有可用工具名称"""
        return [t.name for t in self.tools]
    
    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """根据名称获取工具定义"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


# # =============================================================================
# # 格式检查结果
# # =============================================================================

# @dataclass 
# class FormatCheckResult:
#     """
#     格式检查结果
    
#     对应 evaluate_toolbench_basic.py 中的格式检查逻辑：
#     - API 定义格式检查（_validate_api_definition）
#     - Action 格式检查（Thought/Action/Action Input）
#     - Action Input JSON 格式检查
#     - Finish 函数语义检查
#     """
#     is_valid: bool                      # 格式是否有效
#     errors: List[str] = field(default_factory=list)    # 错误列表
#     warnings: List[str] = field(default_factory=list)  # 警告列表（如 Thought 超过5句）
    
#     # 详细检查项结果
#     api_definition_valid: bool = True   # API 定义格式是否有效
#     action_format_valid: bool = True    # Action 格式是否有效
#     action_input_valid: bool = True     # Action Input JSON 是否有效
#     finish_valid: bool = True           # Finish 函数是否正确（如果存在）
    
#     # 元信息
#     sample_id: Optional[str] = None


# @dataclass
# class ExecutabilityResult:
#     """
#     可执行性检查结果
    
#     静态检查：
#     - 调用的 API 是否在 tools 列表中存在
#     - 必需参数是否都提供了
#     - 参数类型是否正确
    
#     动态检查（真实 API 数据集）：
#     - 实际调用 API 是否成功
#     - 响应是否有效
#     """
#     is_executable: bool                 # 是否可执行
    
#     # 静态检查结果
#     tool_exists: bool = True            # 调用的工具是否存在于 tools 列表
#     params_valid: bool = True           # 参数是否有效
#     missing_required_params: List[str] = field(default_factory=list)  # 缺失的必需参数
#     unknown_params: List[str] = field(default_factory=list)           # 未知的参数（不在定义中）
#     type_errors: List[str] = field(default_factory=list)              # 类型错误
    
#     # 动态检查结果（真实 API 调用）
#     api_called: bool = False            # 是否实际调用了 API
#     api_success: bool = False           # API 调用是否成功
#     api_response: Optional[Any] = None  # API 响应
#     api_error: Optional[str] = None     # API 错误
#     response_truncated: bool = False    # 响应是否被截断
    
#     # 元信息
#     sample_id: Optional[str] = None
#     api_call_idx: Optional[int] = None  # 第几个 API 调用


# @dataclass
# class SampleEvalResult:
#     """单条数据的完整评估结果"""
    
#     sample_id: str
    
#     # 格式检查
#     format_result: Optional[FormatCheckResult] = None
    
#     # 可执行性检查（每个 API 调用一个结果）
#     executability_results: List[ExecutabilityResult] = field(default_factory=list)
    
#     # 汇总
#     format_valid: bool = True           # 格式是否有效
#     all_executable: bool = True         # 所有调用是否都可执行
    
#     # 原始样本引用（可选）
#     sample: Optional[APIAgentSample] = None


# @dataclass
# class DatasetReport:
#     """数据集评估报告"""
    
#     # === 基本信息 ===
#     dataset_name: str
#     total_samples: int
#     timestamp: str = ""
#     elapsed_seconds: float = 0.0
    
#     # === 格式正确性 ===
#     format_valid_count: int = 0
#     format_valid_rate: float = 0.0
#     format_error_details: Dict[str, int] = field(default_factory=dict)  # 错误类型统计
    
#     # === 可执行性 ===
#     executable_count: int = 0
#     executable_rate: float = 0.0
    
#     # 静态可执行性细分
#     tool_not_found_count: int = 0       # 工具不存在的样本数
#     param_error_count: int = 0          # 参数错误的样本数
    
#     # 动态可执行性（真实 API）
#     api_call_count: int = 0             # 总 API 调用次数
#     api_success_count: int = 0          # 成功的 API 调用次数
#     api_success_rate: float = 0.0
#     response_truncated_count: int = 0   # 响应被截断的次数
    
#     # === 多样性（后续扩展）===
#     unique_tools: int = 0               # 唯一工具数
#     tool_entropy: float = 0.0           # 工具使用熵
    
#     # === 错误样本 ===
#     error_samples: List[SampleEvalResult] = field(default_factory=list)
    
#     def summary(self) -> str:
#         """生成摘要报告"""
#         lines = [
#             "=" * 60,
#             f"API Agent Dataset Evaluation: {self.dataset_name}",
#             "=" * 60,
#             f"Total Samples:      {self.total_samples:,}",
#             "",
#             "--- Format Correctness ---",
#             f"  Valid:            {self.format_valid_count:,} ({self.format_valid_rate:.2%})",
#             "",
#             "--- Executability ---",
#             f"  Executable:       {self.executable_count:,} ({self.executable_rate:.2%})",
#             f"  Tool Not Found:   {self.tool_not_found_count:,}",
#             f"  Param Errors:     {self.param_error_count:,}",
#             "",
#             "--- API Calls ---",
#             f"  Total Calls:      {self.api_call_count:,}",
#             f"  Success:          {self.api_success_count:,} ({self.api_success_rate:.2%})",
#             f"  Truncated:        {self.response_truncated_count:,}",
#             "",
#             "--- Diversity ---",
#             f"  Unique Tools:     {self.unique_tools:,}",
#             f"  Tool Entropy:     {self.tool_entropy:.2f} bits",
#             "=" * 60,
#         ]
#         return "\n".join(lines)
