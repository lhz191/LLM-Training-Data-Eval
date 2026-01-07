#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Reasoning Data Evaluation - 数据类型定义

统一的数学推理数据格式，支持：
- 纯 CoT 数据集：MetaMathQA, GSM8K-Aug, NuminaMath-CoT
- Code/TIR 数据集：OpenMathInstruct-1, NuminaMath-TIR
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union

@dataclass
class MathSample:
    """数学推理数据的统一输入格式"""
    
    # === 必需字段 ===
    question: str           # 数学问题
    solution: Union[str, List[str]]  # 解答过程 (CoT/代码)
                                      # - str: 单个解答（如 OpenMathInstruct）
                                      # - List[str]: 多个程序（如 LILA，需全部验证通过）
    
    # === GT (用户提供或从原始数据集加载) ===
    ground_truth: Union[str, List[Any], None] = None  # 标准答案
                                                       # - str: 单个答案
                                                       # - List[Any]: 多个答案（如 x=75, y=25）
    
    # === 元信息 ===
    source_dataset: Optional[str] = None    # 来源数据集名称 (如 "gsm8k", "math")
    question_type: Optional[str] = None     # 题目类型 (如 "MATH_AnsAug", "GSM_Rephrased")
    sample_id: Optional[str] = None         # 样本唯一标识
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元信息
    
    def __repr__(self):
        q_preview = self.question[:50] + "..." if len(self.question) > 50 else self.question
        gt_str = self.ground_truth if self.ground_truth else "(no GT)"
        return f"MathSample(question='{q_preview}', ground_truth='{gt_str}')"


# =============================================================================
# 以下类暂未使用，先注释掉
# =============================================================================

# @dataclass
# class EvalResult:
#     """单条数据的评估结果"""
#     
#     is_correct: bool                        # 答案是否正确
#     predicted_answer: Optional[str]         # 提取/执行得到的预测答案
#     ground_truth: str                       # 标准答案
#     match_type: Optional[str] = None        # 匹配方式: "exact", "numeric", "symbolic", "failed"
#     error_message: Optional[str] = None     # 如果出错，错误信息
#     
#     # 原始样本引用
#     sample: Optional[MathSample] = None


# @dataclass
# class DatasetReport:
#     """数据集评估报告"""
#     
#     # === 基本统计 ===
#     total_samples: int              # 总样本数
#     correct_count: int              # 正确数
#     accuracy: float                 # 准确率
#     
#     # === 细分统计 ===
#     by_type: Optional[Dict[str, Dict]] = None    # 按题目类型分组: {type: {total, correct, accuracy}}
#     by_source: Optional[Dict[str, Dict]] = None  # 按来源数据集分组
#     
#     # === 错误分析 ===
#     error_count: int = 0
#     error_samples: Optional[List[EvalResult]] = None  # 错误样本 (可选保存)
#     
#     def summary(self) -> str:
#         """生成摘要报告"""
#         lines = [
#             "=" * 60,
#             "Math Reasoning Dataset Evaluation Report",
#             "=" * 60,
#             f"Total Samples:  {self.total_samples:,}",
#             f"Correct:        {self.correct_count:,}",
#             f"Accuracy:       {self.accuracy:.2%}",
#         ]
#         
#         if self.by_type:
#             lines.append("\n--- By Question Type ---")
#             for qtype, stats in sorted(self.by_type.items()):
#                 lines.append(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
#         
#         if self.by_source:
#             lines.append("\n--- By Source Dataset ---")
#             for source, stats in sorted(self.by_source.items()):
#                 lines.append(f"  {source}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
#         
#         lines.append("=" * 60)
#         return "\n".join(lines)

