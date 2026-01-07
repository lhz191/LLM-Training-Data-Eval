#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码提取器和执行器 - 基础模块

支持从不同格式的 solution 中提取代码并安全执行。
设计为可扩展架构，支持不同数据集的定制化。

本文件包含：
- 基类定义（AnswerExtractor, CodeExtractor, CodeExecutor, ResultComparator）
- 通用比较函数（to_sympy, sympy_equal, compare_results, compare_math_answers）
- 工厂函数和注册表

具体实现：
- OpenMath 相关: openmath_executor.py
- LILA 相关: lila_executor.py

使用方式:
    # 获取提取器和执行器
    extractor = get_extractor('openmathinstruct-1')
    executor = get_executor('openmath')
    
    # 提取代码和预期输出
    code = extractor.extract(solution)
    expected = extractor.extract_output(solution)
    
    # 执行代码
    result, error = executor.execute(code)
    
    # 比较结果
    is_match = compare_results(result, expected)
"""
import re
import sys
import io
from typing import Optional, Tuple, Any, List
from abc import ABC, abstractmethod


# =============================================================================
# 基类
# =============================================================================

class FormatChecker(ABC):
    """
    格式检查器基类
    
    检查数学数据集样本的格式正确性。
    """
    
    @abstractmethod
    def check(self, sample) -> Tuple[List[str], List[str]]:
        """
        检查样本的格式正确性
        
        Args:
            sample: MathSample 样本
            
        Returns:
            (errors, warnings) 元组
            - errors: 错误列表（严重问题，影响数据可用性）
            - warnings: 警告列表（轻微问题，不影响数据可用性）
        """
        pass


class AnswerExtractor(ABC):
    """
    答案提取器基类
    
    从 solution 中提取最终答案，不同数据集格式不同：
    - OpenMathInstruct-1: \boxed{} 格式
    - GSM8K: #### 格式
    - 有些数据集答案直接在独立字段，不需要提取
    """
    
    @abstractmethod
    def extract(self, solution: str) -> Optional[str]:
        """
        从 solution 中提取答案
        
        Args:
            solution: 解答文本
            
        Returns:
            提取的答案，如果无法提取返回 None
        """
        pass


class CodeExtractor(ABC):
    """
    代码提取器基类
    
    不同数据集有不同的代码格式，需要实现对应的提取器。
    """
    
    @abstractmethod
    def extract(self, solution: str) -> Optional[str]:
        """
        从 solution 中提取代码
        
        Args:
            solution: 包含代码的解答文本
            
        Returns:
            提取出的代码字符串，如果没有代码则返回 None
        """
        pass
    
    def extract_output(self, solution: str) -> Optional[str]:
        """
        从 solution 中提取预期输出（如果有的话）
        
        Args:
            solution: 包含代码的解答文本
            
        Returns:
            预期输出字符串，如果没有则返回 None
        """
        return None


class CodeExecutor(ABC):
    """
    代码执行器基类
    
    不同数据集对"输出"的定义不同，需要实现对应的执行器。
    """
    
    @abstractmethod
    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        执行代码
        
        Args:
            code: 要执行的代码
            
        Returns:
            (result, error_message)
            - result: 执行结果
            - error_message: 如果出错，错误信息；否则为 None
        """
        pass


class ResultComparator:
    """
    结果比较器基类
    
    不同数据集可能有不同的比较逻辑
    """
    
    def compare(self, result: Any, expected: Any) -> bool:
        """
        比较执行结果与预期答案
        
        Args:
            result: 程序执行结果
            expected: 预期答案（可能是单个值或列表）
            
        Returns:
            是否匹配
        """
        return compare_results(result, expected)


# =============================================================================
# 异常和辅助函数
# =============================================================================

class TimeoutError(Exception):
    """代码执行超时异常"""
    pass


def _timeout_handler(signum, frame):
    """信号处理器：超时时抛出异常"""
    raise TimeoutError("Code execution timed out")


# =============================================================================
# 工厂函数和注册表
# =============================================================================

# 答案提取器注册表
ANSWER_EXTRACTORS = {}

# 代码提取器注册表
CODE_EXTRACTORS = {}

# 执行器注册表
EXECUTORS = {}

# 保持向后兼容
EXTRACTORS = CODE_EXTRACTORS


def get_answer_extractor(extractor_type: str) -> AnswerExtractor:
    """
    获取答案提取器
    
    Args:
        extractor_type: 提取器类型
            - 'boxed' / 'openmath' / 'math': \\boxed{} 格式
            - 'hash' / 'gsm8k': #### 格式
            - 'direct' / 'gsm8kaug': 不提取，使用 ground_truth 字段
        
    Returns:
        对应的 AnswerExtractor 实例
    """
    name_lower = extractor_type.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    if name_lower not in ANSWER_EXTRACTORS:
        available = list(ANSWER_EXTRACTORS.keys())
        raise ValueError(f"Unknown answer extractor: {extractor_type}. Available: {available}")
    
    return ANSWER_EXTRACTORS[name_lower]()


def get_code_extractor(dataset_name: str) -> CodeExtractor:
    """
    根据数据集名称获取对应的代码提取器
    
    Args:
        dataset_name: 数据集名称，如 'openmathinstruct-1'
        
    Returns:
        对应的 CodeExtractor 实例
    """
    name_lower = dataset_name.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    if name_lower not in CODE_EXTRACTORS:
        available = list(CODE_EXTRACTORS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return CODE_EXTRACTORS[name_lower]()


# 保持向后兼容
def get_extractor(dataset_name: str) -> CodeExtractor:
    """向后兼容的别名"""
    return get_code_extractor(dataset_name)


def get_executor(executor_type: str = 'openmath') -> CodeExecutor:
    """
    获取代码执行器
    
    Args:
        executor_type: 执行器类型，如 'openmath'
            
    Returns:
        对应的 CodeExecutor 实例
    """
    type_lower = executor_type.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    if type_lower not in EXECUTORS:
        available = list(EXECUTORS.keys())
        raise ValueError(f"Unknown executor: {executor_type}. Available: {available}")
    
    return EXECUTORS[type_lower]()


def register_answer_extractor(name: str, extractor_class: type):
    """注册自定义答案提取器"""
    ANSWER_EXTRACTORS[name.lower()] = extractor_class


def register_code_extractor(name: str, extractor_class: type):
    """注册自定义代码提取器"""
    CODE_EXTRACTORS[name.lower()] = extractor_class


def register_executor(name: str, executor_class: type):
    """注册自定义执行器"""
    EXECUTORS[name.lower()] = executor_class


# 保持向后兼容
register_extractor = register_code_extractor


# =============================================================================
# 通用比较函数
# =============================================================================

def compare_math_answers(answer1: str, answer2: str) -> bool:
    """
    比较两个数学答案是否等价
    
    使用 NVIDIA NeMo-Skills 的 math_grader 库进行比较
    
    策略（按优先级）：
    1. 额外标准化（移除百分号、句末句号）
    2. latex2sympy2_extended.normalize_latex 标准化
    3. 字符串相等比较（移除空格后）
    4. math_verify 符号验证（处理 LaTeX、数值、表达式等价性）
    
    支持：
    - LaTeX 格式差异（空格、百分号、括号）
    - 数值等价性（0.5 vs 1/2）
    - 比例格式（5:6 vs 5/6）
    - 符号表达式等价性
    - MCQ 选项（A, B, C, D 等）
    """
    from latex2sympy2_extended import NormalizationConfig, normalize_latex
    from math_verify import LatexExtractionConfig, parse, verify
    
    if answer1 is None or answer2 is None:
        return False
    
    s1 = str(answer1).strip()
    s2 = str(answer2).strip()
    
    # 1. 直接相等
    if s1 == s2:
        return True
    
    # 2. 额外标准化（参考 NeMo-Skills 的 _additional_normalization）
    # 移除百分号（支持空格）
    percentage_pattern = r"^(\d+\.?\d*)\s*(?:\\%|%)$"
    match1 = re.fullmatch(percentage_pattern, s1)
    if match1:
        s1 = match1.group(1)
    match2 = re.fullmatch(percentage_pattern, s2)
    if match2:
        s2 = match2.group(1)
    
    # 移除句末的句号和反斜杠
    s1 = s1.rstrip(".\\")
    s2 = s2.rstrip(".\\")
    
    # 3. 使用 latex2sympy2_extended 标准化
    try:
        s1_normalized = normalize_latex(s1, NormalizationConfig)
        s2_normalized = normalize_latex(s2, NormalizationConfig)
        
        # 移除空格后比较
        if s1_normalized.replace(" ", "") == s2_normalized.replace(" ", ""):
            return True
    except:
        # 如果标准化失败，继续尝试其他方法
        s1_normalized = s1
        s2_normalized = s2
    
    # 4. 检查是否是 MCQ 选项
    mcq_options = "ABCDEFGHIJ"
    from math_verify import StringExtractionConfig
    
    is_mcq = re.fullmatch("|".join(mcq_options), s1_normalized.strip())
    if is_mcq:
        try:
            parsed_gt = parse(s1, [StringExtractionConfig(strings=tuple(mcq_options))], parsing_timeout=30)
            parsed_pred = parse(s2, [StringExtractionConfig(strings=tuple(mcq_options))], parsing_timeout=30)
            if verify(parsed_gt, parsed_pred):
                return True
        except:
            pass
    
    # 5. 使用 math_verify 进行符号验证
    # math_verify.parse 期望输入在 latex 环境中，例如 $...$
    # parsing_timeout 从默认 5 秒改为 30 秒，避免复杂表达式超时
    latex_env_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    
    current_s1 = s1_normalized
    current_s2 = s2_normalized
    
    if not re.search(latex_env_pattern, current_s1, re.DOTALL):
        current_s1 = f"${current_s1}$"
    if not re.search(latex_env_pattern, current_s2, re.DOTALL):
        current_s2 = f"${current_s2}$"
    
    try:
        parsed_1 = parse(current_s1, [LatexExtractionConfig()], parsing_timeout=30)
        parsed_2 = parse(current_s2, [LatexExtractionConfig()], parsing_timeout=30)
        if verify(parsed_1, parsed_2):
            return True
    except:
        pass
    
    return False


def normalize_math_symbols(s: str) -> str:
    """
    将 Unicode 数学符号标准化为 ASCII 等价物
    
    处理 SymPy pprint 等工具输出的 Unicode 符号差异
    """
    # 先处理 √ 符号，需要加括号：√x → sqrt(x), √3 → sqrt(3)
    # 匹配 √ 后面跟着的数字或字母序列
    s = re.sub(r'√(\d+)', r'sqrt(\1)', s)           # √3 → sqrt(3)
    s = re.sub(r'√([a-zA-Z_][a-zA-Z0-9_]*)', r'sqrt(\1)', s)  # √x → sqrt(x)
    s = re.sub(r'√\(', r'sqrt(', s)                  # √( → sqrt(
    s = re.sub(r'√', r'sqrt', s)                     # 其他情况
    
    # Unicode 数学符号 → ASCII 等价物
    replacements = {
        '⋅': '*',      # DOT OPERATOR → asterisk
        '·': '*',      # MIDDLE DOT → asterisk
        '×': '*',      # MULTIPLICATION SIGN → asterisk
        '−': '-',      # MINUS SIGN → hyphen-minus
        '↦': '->',     # RIGHTWARDS ARROW FROM BAR → arrow
        '→': '->',     # RIGHTWARDS ARROW → arrow
        '≤': '<=',     # LESS-THAN OR EQUAL TO
        '≥': '>=',     # GREATER-THAN OR EQUAL TO
        '≠': '!=',     # NOT EQUAL TO
        '∞': 'oo',     # INFINITY (SymPy notation)
        'π': 'pi',     # GREEK SMALL LETTER PI
        '²': '**2',    # SUPERSCRIPT TWO
        '³': '**3',    # SUPERSCRIPT THREE
        '⁴': '**4',    # SUPERSCRIPT FOUR
    }
    
    for unicode_sym, ascii_sym in replacements.items():
        s = s.replace(unicode_sym, ascii_sym)
    
    return s


def normalize_output(s: str) -> str:
    """
    标准化输出字符串，处理格式差异
    """
    s = str(s).strip()
    
    # Unicode 数学符号标准化
    s = normalize_math_symbols(s)
    
    # 移除多余的空格和换行
    s = re.sub(r'\s+', ' ', s)
    
    # 移除外层引号（如果有）
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    
    return s


# ============================================================================
# 辅助函数：用于 compare_results()（比较代码执行结果）
# ============================================================================

def to_sympy(obj: Any):
    """
    统一转换任意对象为 SymPy 表达式
    
    用于 compare_results() 比较代码执行结果
    
    支持：
    - 数值 (int, float, complex)
    - 字符串 (普通数学表达式或 LaTeX)
    - numpy array → SymPy Matrix
    - SymPy 对象 → 直接返回
    - list/tuple → SymPy Matrix
    
    Returns:
        SymPy 表达式，或 None（无法转换）
    """
    from sympy import sympify, Matrix
    
    if obj is None:
        return None
    
    # 已经是 SymPy 对象
    if hasattr(obj, 'is_number') or hasattr(obj, 'is_Matrix'):
        return obj
    
    # numpy array → Matrix
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return Matrix(obj.tolist())
    except ImportError:
        pass
    
    # list/tuple → Matrix (如果是数值列表)
    if isinstance(obj, (list, tuple)):
        try:
            return Matrix(obj)
        except:
            pass
    
    # 数值
    if isinstance(obj, (int, float, complex)):
        return sympify(obj)
    
    # 字符串
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        
        # 移除 LaTeX 数学模式包装 $...$
        was_latex = s.startswith('$') and s.endswith('$')
        if was_latex:
            s = s[1:-1].strip()
        
        # 尝试 LaTeX 解析（使用 latex2sympy2_extended，兼容性更好）
        # 条件：包含 LaTeX 命令（\）、花括号（{}）、或者原本就是 $...$ 包装的
        if '\\' in s or '{' in s or was_latex:
            try:
                from latex2sympy2_extended import latex2sympy
                return latex2sympy(s)
            except:
                pass
            
            # 尝试处理 Mathematica 风格的表达式: \left\{\left\{x\to EXPR\right\}\right\}
            # 提取 x\to 后面的表达式部分
            if r'\to' in s or r'\\to' in s:
                try:
                    from latex2sympy2_extended import latex2sympy
                    # 匹配 x\\to EXPR 或 x\to EXPR，提取 EXPR 部分（到 \\right\\} 或 \right\} 为止）
                    match = re.search(r'[a-z]\\\\?to\s*(.+?)(?:\\\\right\\\\}|\\right\\}|$)', s, re.IGNORECASE)
                    if match:
                        expr_str = match.group(1).strip()
                        if expr_str:
                            return latex2sympy(expr_str)
                except:
                    pass
        
        # 普通表达式（非 LaTeX）
        try:
            return sympify(s)
        except:
            pass
    
    return None


def sympy_equal(a, b, tol: float = 1e-6) -> bool:
    """
    比较两个 SymPy 表达式是否等价
    
    用于 compare_results() 比较代码执行结果
    
    策略：
    1. 直接相等
    2. 符号简化后相等
    3. 数值计算后在容差内相等
    """
    from sympy import simplify, N, Abs, oo
    
    if a is None or b is None:
        return False
    
    # 直接相等
    if a == b:
        return True
    
    # 尝试简化比较
    try:
        diff = simplify(a - b)
        if diff == 0:
            return True
    except:
        pass
    
    # 数值比较
    try:
        # 处理矩阵
        if hasattr(a, 'is_Matrix') and a.is_Matrix:
            if hasattr(b, 'is_Matrix') and b.is_Matrix:
                if a.shape != b.shape:
                    return False
                # 逐元素比较
                diff_matrix = a - b
                for elem in diff_matrix:
                    val = complex(N(Abs(elem)))
                    if abs(val) > tol:
                        return False
                return True
            return False
        
        # 标量数值比较
        v1 = complex(N(a))
        v2 = complex(N(b))
        if abs(v1 - v2) < tol:
            return True
    except:
        pass
    
    return False


def compare_results(result: Any, expected: Any, tol: float = 1e-6) -> bool:
    """
    比较执行结果和预期答案
    
    核心思路：统一转换为 SymPy 表达式进行比较
    
    比较流程：
    1. 快速路径：字符串直接相等
    2. 统一路径：转为 SymPy → 符号/数值比较
    
    Args:
        result: 执行结果（任意类型）
        expected: 预期输出（任意类型）
        tol: 数值比较容差（默认 1e-6）
        
    Returns:
        是否匹配
    """
    if result is None:
        return False
    
    # 快速路径：字符串/repr 直接相等
    try:
        r_str = str(result).strip()
        e_str = str(expected).strip()
        
        # str 直接比较
        if r_str == e_str:
            return True
        
        # 标准化后比较（处理空格差异）
        r_norm = normalize_output(r_str)
        e_norm = normalize_output(e_str)
        if r_norm == e_norm:
            return True
        
        # repr 比较（处理 Decimal('66.450') vs "Decimal('66.450')" 等情况）
        r_repr = repr(result).strip()
        if r_repr == e_str or r_repr == e_norm:
            return True
        
        # expected 可能带外层引号，去掉后比较
        if e_str.startswith(("'", '"')) and e_str.endswith(("'", '"')):
            e_unquoted = e_str[1:-1]
            if r_str == e_unquoted or r_repr == e_unquoted:
                return True
    except:
        pass
    
    # 容器类型比较（set/list/dict/tuple - 通过 eval expected 后直接比较对象）
    # 这处理了 Python 标准容器的顺序差异（如 set 元素顺序、dict 键顺序）
    try:
        e_str = str(expected).strip()
        if e_str.startswith(("'", '"')) and e_str.endswith(("'", '"')):
            e_str = e_str[1:-1]
        expected_obj = eval(e_str)
        # NumPy 数组比较会返回布尔数组，需要特殊处理
        import numpy as np
        if isinstance(result, np.ndarray) or isinstance(expected_obj, np.ndarray):
            if np.array_equal(result, expected_obj):
                return True
        elif result == expected_obj:
            return True
        # 无序列表比较（处理 sympy.solve 等返回顺序不确定的情况）
        # 数学问题中列表通常表示"所有满足条件的结果"，顺序不重要
        elif isinstance(result, (list, tuple)) and isinstance(expected_obj, (list, tuple)):
            try:
                if sorted(str(x) for x in result) == sorted(str(x) for x in expected_obj):
                    return True
            except:
                pass
    except:
        pass
    
    # SymPy dict 比较（如 {b: 2, m: 1} vs {m: 1, b: 2}，键是符号不能直接 eval）
    # 通过解析键值对字符串，排序后比较
    try:
        r_str = str(result).strip()
        e_str = str(expected).strip()
        if r_str.startswith('{') and ':' in r_str and e_str.startswith('{') and ':' in e_str:
            def parse_sympy_dict(s):
                """解析 {k1: v1, k2: v2} 格式为排序后的 (key, value) 列表"""
                content = s.strip()[1:-1].strip()
                if not content:
                    return []
                pairs = []
                for part in content.split(','):
                    if ':' in part:
                        k, v = part.split(':', 1)
                        pairs.append((k.strip(), v.strip()))
                return sorted(pairs)
            
            r_pairs = parse_sympy_dict(r_str)
            e_pairs = parse_sympy_dict(e_str)
            if r_pairs and e_pairs and r_pairs == e_pairs:
                return True
    except:
        pass
    
    # 统一路径：转为 SymPy 比较（处理数学等价性、SymPy 对象等）
    try:
        r_sym = to_sympy(result)
        e_sym = to_sympy(expected)
        if sympy_equal(r_sym, e_sym, tol=tol):
            return True
    except:
        pass
        
        return False


def get_comparator(dataset_type: str) -> ResultComparator:
    """
    根据数据集类型获取对应的比较器
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        ResultComparator 实例
    """
    dataset_type = dataset_type.lower()
    
    if 'lila' in dataset_type:
        from lila_executor import LILAResultComparator
        return LILAResultComparator()
    elif 'openmath' in dataset_type or 'math' in dataset_type:
        from openmath_executor import OpenMathResultComparator
        return OpenMathResultComparator()
    else:
        return ResultComparator()


# =============================================================================
# 自动导入子模块以注册提取器和执行器
# =============================================================================

def _auto_import_submodules():
    """自动导入子模块，触发它们的注册逻辑"""
    try:
        import openmath_executor
    except ImportError:
        pass
    
    try:
        import lila_executor
    except ImportError:
        pass


# 在模块加载时自动导入
_auto_import_submodules()


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("代码执行器测试")
    print("=" * 60)
    
    # 测试 OpenMathInstruct-1 格式
    test_solution = """Let's solve this problem using Python code.
<llm-code>
score_first_player = 30
score_second_player = 2 * score_first_player
score_first_player + score_second_player
</llm-code>
<llm-code-output>
90
</llm-code-output>
Thus Tom and Phillip scored \\boxed{90} points."""

    print("\n【测试 1: OpenMathInstruct-1 格式】")
    print("-" * 40)
    
    extractor = get_extractor('openmathinstruct-1')
    executor = get_executor('openmath')
    
    code = extractor.extract(test_solution)
    expected = extractor.extract_output(test_solution)
    
    print(f"提取的代码:\n{code}")
    print(f"\n预期输出: {expected}")
    
    result, error = executor.execute(code)
    print(f"\n实际结果: {result}")
    print(f"执行错误: {error}")
    
    match = compare_results(result, expected)
    print(f"\n结果匹配: {match}")
