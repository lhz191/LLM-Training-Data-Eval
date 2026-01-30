#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training Data Evaluation - 统一基类

所有模态的检查器、提取器、执行器的根基类定义。
通过继承这些基类，各模态的实现可以被统一管理和调用。

模态 (Modality):
- api: API Agent (ToolBench, xLAM)
- gui: GUI Agent (Mind2Web, WebShop, WebLINX)
- math: Math/Symbolic (LILA, OpenMath)
- image: Image-Text (COCO, etc.)
- video: Video-Text

检查器类型 (Checker Type):
- format_check: 格式检查
- static_executability: 静态可执行性检查
- dynamic_executability: 动态可执行性检查
- code_extractor: 代码提取
- code_executor: 代码执行
- result_comparator: 结果比较
- html_locator: HTML 元素定位
- answer_extractor: 答案提取
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


# =============================================================================
# 根基类
# =============================================================================

class BaseChecker(ABC):
    """
    所有检查器的根基类
    
    提供统一的身份标识，使得不同模态的检查器可以被统一管理。
    """
    
    @property
    @abstractmethod
    def modality(self) -> str:
        """
        所属模态
        
        Returns:
            'api' | 'gui' | 'math' | 'image' | 'video'
        """
        pass
    
    @property
    @abstractmethod
    def checker_type(self) -> str:
        """
        检查器类型
        
        Returns:
            'format_check' | 'static_executability' | 'dynamic_executability' | 
            'code_extractor' | 'code_executor' | 'result_comparator' | 
            'html_locator' | 'answer_extractor'
        """
        pass


# =============================================================================
# 格式检查器
# =============================================================================

class BaseFormatChecker(BaseChecker):
    """
    格式检查器统一基类
    
    检查数据样本的格式是否正确。
    所有模态都应该有格式检查器。
    """
    
    checker_type = 'format_check'
    
    @abstractmethod
    def check(self, data) -> Tuple[List[str], List[str]]:
        """
        检查数据格式
        
        Args:
            data: 数据样本（具体类型由各模态定义）
            
        Returns:
            (errors, warnings) 元组
            - errors: 错误列表（严重问题）
            - warnings: 警告列表（轻微问题）
        """
        pass


# =============================================================================
# 可执行性检查器
# =============================================================================

class BaseExecutabilityChecker(BaseChecker):
    """
    静态可执行性检查器统一基类
    
    检查数据样本是否可以被静态执行/验证。
    适用于: API Agent, GUI Agent
    """
    
    checker_type = 'static_executability'
    
    @abstractmethod
    def check(self, data) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查静态可执行性
        
        Args:
            data: 数据样本
            
        Returns:
            (errors, warnings, stats) 元组
            - errors: 错误列表
            - warnings: 警告列表
            - stats: 统计信息（如成功率等）
        """
        pass


class BaseDynamicChecker(BaseChecker):
    """
    动态可执行性检查器统一基类
    
    在真实环境中执行并验证。
    适用于: API Agent (真实 API 调用), GUI Agent (真实网页)
    """
    
    checker_type = 'dynamic_executability'
    
    @abstractmethod
    def check(self, data) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查动态可执行性
        
        Args:
            data: 数据样本
            
        Returns:
            (errors, warnings, stats) 元组
        """
        pass


# =============================================================================
# 代码相关（主要用于 Math 模态，但预留扩展性）
# =============================================================================

class BaseCodeExtractor(BaseChecker):
    """
    代码提取器统一基类
    
    从文本中提取可执行代码。
    目前主要用于: Math (从 solution 中提取 Python 代码)
    未来可能用于: 代码生成任务
    """
    
    checker_type = 'code_extractor'
    
    @abstractmethod
    def extract(self, text: str) -> Optional[str]:
        """
        从文本中提取代码
        
        Args:
            text: 包含代码的文本
            
        Returns:
            提取的代码，如果没有则返回 None
        """
        pass
    
    def extract_output(self, text: str) -> Optional[str]:
        """
        从文本中提取预期输出（可选）
        
        Args:
            text: 包含预期输出的文本
            
        Returns:
            预期输出，如果没有则返回 None
        """
        return None


class BaseCodeExecutor(BaseChecker):
    """
    代码执行器统一基类
    
    安全执行代码并获取结果。
    目前主要用于: Math
    """
    
    checker_type = 'code_executor'
    
    @abstractmethod
    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        执行代码
        
        Args:
            code: 要执行的代码
            
        Returns:
            (result, error) 元组
            - result: 执行结果
            - error: 错误信息，如果成功则为 None
        """
        pass


class BaseResultComparator(BaseChecker):
    """
    结果比较器统一基类
    
    比较执行结果与预期答案。
    目前主要用于: Math
    """
    
    checker_type = 'result_comparator'
    
    @abstractmethod
    def compare(self, result: Any, expected: Any) -> bool:
        """
        比较结果
        
        Args:
            result: 实际结果
            expected: 预期结果
            
        Returns:
            是否匹配
        """
        pass


# =============================================================================
# 元素定位器（主要用于 GUI 模态）
# =============================================================================

class BaseHTMLLocator(BaseChecker):
    """
    HTML 元素定位器统一基类
    
    在 HTML 中定位目标元素。
    目前主要用于: GUI Agent (Mind2Web, WebLINX)
    """
    
    checker_type = 'html_locator'
    
    @abstractmethod
    def can_locate(self, action, html: str) -> Tuple[bool, str]:
        """
        检查是否能在 HTML 中定位目标元素
        
        Args:
            action: 动作对象（包含目标元素信息）
            html: HTML 内容
            
        Returns:
            (success, reason) 元组
            - success: 是否成功定位
            - reason: 原因说明
        """
        pass


# =============================================================================
# 答案提取器（主要用于 Math 模态）
# =============================================================================

class BaseAnswerExtractor(BaseChecker):
    """
    答案提取器统一基类
    
    从解答文本中提取最终答案。
    目前主要用于: Math (提取 \\boxed{} 中的答案)
    """
    
    checker_type = 'answer_extractor'
    
    @abstractmethod
    def extract(self, solution: str) -> Optional[str]:
        """
        从解答中提取答案
        
        Args:
            solution: 解答文本
            
        Returns:
            提取的答案，如果无法提取返回 None
        """
        pass
