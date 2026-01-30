#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LILA 结果比较器
"""

import os
import sys
import re
from typing import Optional, Tuple, Any, List

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import ResultComparator, compare_results, to_sympy

class LILAResultComparator(ResultComparator):
    """
    LILA 数据集结果比较器
    
    核心策略：统一提取数值，然后比较
    1. 使用 SymPy 将各种格式（分数、LaTeX、表达式）转换为浮点数
    2. 检查 expected 中的每个数值是否在 result 中能找到匹配（允许误差）
    
    支持格式：
    - 普通数值: '3.25', '-2.14'
    - SymPy 分数/表达式: '15/7', 'sqrt(2)', '{x: 15/7, y: 11/4}'
    - LaTeX: '$\\frac{3}{4}$', '$\\sqrt{2}$'
    - numpy 数组: '[[ 0.  -2.   0.]]'
    - Python dict: "{'Perimeter': 2.35, 'Area': 0.3}"
    """
    
    def compare(self, result: Any, expected: Any) -> bool:
        """
        统一比较逻辑：提取数值后比较
        """
        if result is None:
            return False
        
        if expected is None:
            return True
        
        # 统一为列表
        expected_list = expected if isinstance(expected, list) else [expected]
        
        if not expected_list:
            return True
        
        r_str = str(result).strip()
        e_str = expected_list[0] if len(expected_list) == 1 else str(expected_list)
        
        # 0. 直接字符串相等
        if r_str == e_str:
            return True
        
        # 0.5. 布尔值标准化比较
        # 处理 LaTeX 格式的布尔值: \text{True}, \text{False}, \text{true}, \text{false}
        bool_match = self._compare_boolean(r_str, e_str)
        if bool_match is not None:
            return bool_match
        
        # 1. 尝试使用符号比较（处理 LaTeX 等价性、数值比较，容差 0.01）
        # 对于 LILA 数据集，\log 表示自然对数，需要替换为 \ln
        if self._lila_compare_results(r_str, e_str, tol=0.01):
            return True
        
        # 2. 检查是否为多项式/方程格式（GT 包含变量 x, y, z）
        if self._is_polynomial_or_equation(e_str):
            poly_match = self._compare_polynomial_or_equation(r_str, e_str)
            if poly_match is not None:
                return poly_match
        
        # 3. 从容器类型中提取数值后比较
        # 适用场景：result 或 expected 是字典、列表或元组
        # 例如：result='18', expected="{'number': '18', ...}" 需要提取数值比较
        #
        # 判断逻辑：result 或 expected 任一是容器类型就走数值提取
        def is_container(s):
            return (s.startswith('{') and s.endswith('}')) or \
                   (s.startswith('[') and s.endswith(']')) or \
                   (s.startswith('(') and s.endswith(')'))
        
        if is_container(r_str) or is_container(e_str):
            result_nums = self._extract_all_numbers(r_str)
            expected_nums = self._extract_all_numbers(e_str)
            
            if expected_nums:
                # 检查 expected 中的每个数值是否在 result 中能找到匹配
                return self._all_expected_in_result(expected_nums, result_nums)
        
        return False
    
    def _compare_boolean(self, result: str, expected: str) -> Optional[bool]:
        """
        比较布尔值，处理 LaTeX 格式
        
        支持格式:
        - Python: True, False, true, false
        - LaTeX: \text{True}, \text{False}, \text{true}, \text{false}
        - LaTeX: \\text{True}, \\text{False} (escaped)
        
        Returns:
            True 如果匹配，False 如果不匹配，None 如果不是布尔值比较
        """
        # 标准化布尔值
        def normalize_bool(s: str) -> Optional[str]:
            s = s.strip()
            # 移除 LaTeX 数学模式 $...$ 包装
            s = re.sub(r'^\$|\$$', '', s).strip()
            # 移除 LaTeX \text{} 包装
            # 处理 \text{True}, \text{False} 等
            match = re.match(r'\\?\\text\{(\w+)\}', s)
            if match:
                s = match.group(1)
            
            # 统一为小写比较
            s_lower = s.lower()
            if s_lower in ('true', '1', 'yes'):
                return 'true'
            elif s_lower in ('false', '0', 'no'):
                return 'false'
            return None
        
        r_bool = normalize_bool(result)
        e_bool = normalize_bool(expected)
        
        # 只有当两者都是布尔值时才比较
        if r_bool is not None and e_bool is not None:
            return r_bool == e_bool
        
        return None
    
    def _lila_compare_results(self, result: str, expected: str, tol: float = 0.01) -> bool:
        """
        LILA 专用的结果比较函数
        
        与通用 compare_results 的区别：
        - LILA 数据集中 \\log 表示自然对数（ln），需要替换为 \\ln 后再比较
        - 处理 Mathematica 风格的 x\\to EXPR 格式
        
        Args:
            result: 程序执行结果字符串
            expected: 期望答案字符串（可能包含 LaTeX）
            tol: 数值比较容差
            
        Returns:
            是否匹配
        """
        # LILA 数据集中 \log 表示自然对数，替换为 \ln
        expected_fixed = expected
        if r'\log' in expected or r'\\log' in expected:
            expected_fixed = re.sub(r'\\log\s*(\^|\()', r'\\ln\1', expected)
            expected_fixed = re.sub(r'\\log\b', r'\\ln', expected_fixed)
            expected_fixed = re.sub(r'\\\\log\s*(\^|\()', r'\\\\ln\1', expected_fixed)
            expected_fixed = re.sub(r'\\\\log\b', r'\\\\ln', expected_fixed)
        
        # 先尝试标准比较
        if compare_results(result, expected_fixed, tol=tol):
            return True
        
        # 处理 Mathematica 风格: result 是列表如 [-2.15]，expected 是 x\to EXPR
        # 用 to_sympy 解析 expected 得到数值，然后和 result 中的数值比较
        if r'\to' in expected or r'\\to' in expected:
            try:
                from sympy import N
                # 解析 expected 得到表达式
                e_sym = to_sympy(expected_fixed)
                if e_sym is not None:
                    e_val = float(N(e_sym))
                    # 从 result 中提取数值
                    result_nums = self._extract_all_numbers(result)
                    if result_nums:
                        # 检查 result 中是否有数值和 e_val 接近
                        for r_val in result_nums:
                            if abs(r_val - e_val) < tol or (e_val != 0 and abs((r_val - e_val) / e_val) < tol):
                                return True
            except:
                pass
        
        return False
    
    def _is_polynomial_or_equation(self, s: str) -> bool:
        """检查是否为多项式或方程格式（包含变量 x, y, z 和系数）"""
        # 移除 LaTeX 包装
        s_clean = re.sub(r'^\$|\$$', '', s.strip())
        
        # 检查是否包含变量（x, y, z）后面跟着指数、运算符或结尾
        # 排除 \sqrt, \frac 等命令中的字母
        var_pattern = r'(?<!\\)[xyz](?:\^|[+-]|=|\s|$)'
        return bool(re.search(var_pattern, s_clean))
    
    def _compare_polynomial_or_equation(self, result: str, expected: str) -> Optional[bool]:
        """
        比较多项式或方程
        
        多项式格式：
        - GT: $x^2+6 x-12$ (符号形式)
        - Result: [1, 6, -12] (系数数组，从高次到低次)
        
        方程格式：
        - GT: $9 x-7 y-6 z-14=0$ (方程形式)
        - Result: 9 -7 -6 -14 (系数，空格分隔)
        
        Returns:
            True 如果匹配，False 如果不匹配，None 如果无法处理
        """
        from sympy import symbols, Poly, sympify, N, Eq, Rational, expand
        
        # 移除 LaTeX 包装
        e_clean = re.sub(r'^\$|\$$', '', expected.strip())
        
        # 检查是否是方程（包含 =0）
        is_equation = '=0' in e_clean.replace(' ', '')
        
        # 从 result 提取系数
        result_coeffs = self._extract_coefficients_from_result(result)
        if not result_coeffs:
            return None
        
        # 从 GT 提取系数
        expected_coeffs = self._extract_coefficients_from_gt(e_clean, is_equation)
        if not expected_coeffs:
            return None
            
        # 比较系数
        return self._compare_coefficients(result_coeffs, expected_coeffs)
    
    def _extract_coefficients_from_result(self, result: str) -> Optional[List[float]]:
        """从执行结果中提取系数列表"""
        import ast
        
        result = result.strip()
        
        # 1. 尝试解析为 Python 列表/数组
        # 处理 numpy 数组格式: [  1.   6. -12.]
        if result.startswith('[') and result.endswith(']'):
            # 规范化空格
            normalized = re.sub(r'\s+', ' ', result)
            # 替换空格为逗号（在数字之间）
            normalized = re.sub(r'(\d)\s+(-?\d)', r'\1, \2', normalized)
            normalized = re.sub(r'(\d\.)\s+(-?\d)', r'\1, \2', normalized)
            normalized = re.sub(r'(\.)\s+(-?\d)', r'\1, \2', normalized)
            try:
                coeffs = ast.literal_eval(normalized)
                if isinstance(coeffs, list):
                    return [float(c) for c in coeffs]
            except:
                pass
            
            # 回退：用正则提取所有数字
            nums = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', result)
            if nums:
                return [float(n) for n in nums]
        
        # 2. 空格分隔的数字: "9 -7 -6 -14"
        if re.match(r'^-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+$', result):
            nums = result.split()
            return [float(n) for n in nums]
        
        return None
    
    def _extract_coefficients_from_gt(self, gt: str, is_equation: bool) -> Optional[List[float]]:
        """
        从 GT（符号表达式）中提取系数
        
        支持格式：
        - 多项式: x^2+6 x-12
        - 方程: 9 x-7 y-6 z-14=0
        - 平面方程（单变量）: x+4=0 -> [1, 0, 0, 4]
        """
        from sympy import symbols, Poly, sympify, N, expand
        
        try:
            # 移除 =0（考虑末尾可能有 $ 符号）
            expr_str = re.sub(r'\s*=\s*0\s*\$?\s*$', '', gt)
            
            # 替换 LaTeX 语法为 SymPy 语法
            expr_str = self._latex_to_sympy_expr(expr_str)
            
            # 确定变量
            x, y, z = symbols('x y z')
            vars_in_expr = []
            if 'x' in expr_str:
                vars_in_expr.append(x)
            if 'y' in expr_str:
                vars_in_expr.append(y)
            if 'z' in expr_str:
                vars_in_expr.append(z)
            
            if not vars_in_expr:
                return None
            
            # 解析表达式
            expr = sympify(expr_str)
            
            if len(vars_in_expr) == 1 and not is_equation:
                # 单变量多项式（非方程）：提取系数（从高次到低次）
                var = vars_in_expr[0]
                poly = Poly(expr, var)
                coeffs = poly.all_coeffs()
                return [float(N(c)) for c in coeffs]
            else:
                # 方程或多变量：提取 x, y, z 系数和常数项
                # 对于平面方程 x+4=0，也按 [coeff_x, coeff_y, coeff_z, const] 格式
                # 例如 x+4=0 -> [1, 0, 0, 4]
                # 例如 9x - 7y - 6z - 14 -> [9, -7, -6, -14]
                expr = expand(expr)
                
                # 始终按 x, y, z 顺序提取系数
                all_vars = [x, y, z]
                coeffs = []
                for var in all_vars:
                    coeff = expr.coeff(var)
                    coeffs.append(float(N(coeff)))
                
                # 常数项
                const = expr
                for var in all_vars:
                    const = const.subs(var, 0)
                coeffs.append(float(N(const)))
                return coeffs
                
        except Exception as e:
            return None
    
    def _latex_to_sympy_expr(self, latex: str) -> str:
        """
        将 LaTeX 表达式转换为 SymPy 可解析的字符串
        
        使用 latex2sympy2_extended 库，能处理：
        - 复杂嵌套的 \\frac, \\sqrt
        - 隐式乘法（如 2x, 2(y+z)）
        - 方程格式（如 x + 4 = 0）
        """
        from latex2sympy2_extended import latex2sympy
        
        # 移除 $ 符号
        s = re.sub(r'\$', '', latex.strip())
        
        expr = latex2sympy(s)
        return str(expr)
    
    
    def _compare_coefficients(self, result_coeffs: List[float], expected_coeffs: List[float]) -> bool:
        """
        比较两组系数是否等价
        
        支持：
        1. 精确匹配
        2. 比例缩放（如 [8, 8, 4, 4] vs [2, 2, 1, 1]）
        3. 符号翻转（如 [-1, -6, 12] vs [1, 6, -12]）
        """
        if len(result_coeffs) != len(expected_coeffs):
            return False
        
        # 1. 精确匹配（允许小误差）
        if all(self._numbers_equal(r, e) for r, e in zip(result_coeffs, expected_coeffs)):
            return True
        
        # 2. 检查比例关系
        # 找到第一个非零系数来计算比例
        ratio = None
        for r, e in zip(result_coeffs, expected_coeffs):
            if abs(e) > 1e-9:  # expected 非零
                if abs(r) < 1e-9:  # result 为零但 expected 非零
                    return False
                ratio = r / e
                break
        
        if ratio is not None:
            # 检查所有系数是否满足相同比例
            for r, e in zip(result_coeffs, expected_coeffs):
                expected_r = e * ratio
                if not self._numbers_equal(r, expected_r):
                    return False
            return True
        
        # 3. 检查符号翻转
        if all(self._numbers_equal(r, -e) for r, e in zip(result_coeffs, expected_coeffs)):
            return True
        
        return False
    
    def _extract_all_numbers(self, s: str) -> List[float]:
        """
        从任意格式的字符串中提取所有数值（转为 float）
        
        使用 SymPy 进行统一解析，支持分数、表达式等
        """
        from sympy import sympify, N
        from sympy.core.numbers import Number
        import ast
        
        s = str(s).strip()
        numbers = []
        
        # 0. 处理 SymPy 元组格式: (Matrix([...]), (0, 1))
        # 这是 rref(), lu() 等函数的返回值，只需要第一个元素（Matrix 部分）
        if s.startswith('(Matrix('):
            # 提取 Matrix 部分
            matrix_match = re.search(r'\(Matrix\(\[(.*?)\]\)', s, re.DOTALL)
            if matrix_match:
                # 只处理 Matrix 内容，忽略后面的 pivot 索引
                matrix_content = matrix_match.group(1)
                # 提取所有分数和数字
                return self._extract_from_sympy_matrix(matrix_content)
        
        # 1. 尝试直接解析为单个 SymPy 表达式（处理 '15/7', 'sqrt(2)' 等）
        # 注意：只对看起来像单个表达式的字符串使用 sympify
        # 避免把 "9 -7 -6 -14" 这种空格分隔的多个数字当成数学表达式
        if not re.search(r'\d\s+-?\d', s):  # 不包含 "数字 空格 数字" 模式
            try:
                expr = sympify(s)
                if isinstance(expr, Number):
                    val = float(N(expr))
                    return [val]
            except:
                pass
        
        # 2. 尝试解析为 Python 对象（dict, list, tuple）
        try:
            obj = ast.literal_eval(s)
            nums = self._extract_from_python_obj(obj)
            if nums:
                return nums
        except:
            pass
        
        # 3. 尝试解析 SymPy dict: {x: 15/7, y: 11/4}（key 没有引号）
        if s.startswith('{') and ':' in s and "'" not in s.split(':')[0]:
            nums = self._extract_from_sympy_dict(s)
            if nums:
                return nums
        
        # 4. 尝试从 LaTeX 中提取数值
        if '$' in s or '\\' in s:
            nums = self._extract_from_latex(s)
            if nums:
                return nums
        
        # 5. 最后使用正则提取数值
        return self._extract_with_regex(s)
    
    def _extract_from_sympy_matrix(self, matrix_content: str) -> List[float]:
        """从 SymPy Matrix 内容中提取数值"""
        from sympy import sympify, N
        
        numbers = []
        
        # 按行分割: [1, 0, 9/19, -23/19, -10/19], [0, 1, -45/19, -55/38, 5/38]
        # 移除外层方括号，按 ], [ 分割
        rows = re.split(r'\],\s*\[', matrix_content)
        
        for row in rows:
            # 清理方括号
            row = row.strip().strip('[]')
            # 按逗号分割
            elements = row.split(',')
            
            for elem in elements:
                elem = elem.strip()
                if not elem:
                    continue
                try:
                    # 使用 sympify 解析分数如 9/19, -23/19
                    val = float(N(sympify(elem)))
                    if not (val != val or abs(val) == float('inf')):
                        numbers.append(val)
                except:
                    pass
        
        return numbers
    
    def _extract_from_python_obj(self, obj) -> List[float]:
        """从 Python 对象中递归提取数值"""
        numbers = []
        if isinstance(obj, (int, float)):
            if not (obj != obj or abs(obj) == float('inf')):  # 排除 nan 和 inf
                numbers.append(float(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                numbers.extend(self._extract_from_python_obj(item))
        elif isinstance(obj, dict):
            for v in obj.values():
                numbers.extend(self._extract_from_python_obj(v))
        elif isinstance(obj, str):
            try:
                numbers.append(float(obj))
            except:
                pass
        return numbers
    
    def _extract_from_sympy_dict(self, s: str) -> List[float]:
        """从 SymPy dict 字符串中提取数值: {x: 15/7, y: 11/4}"""
        from sympy import sympify, N
        
        numbers = []
        content = s.strip()[1:-1]  # 移除 {}
        
        # 按逗号分割（处理嵌套）
        parts = []
        depth = 0
        current = ''
        for c in content:
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            if c == ',' and depth == 0:
                parts.append(current.strip())
                current = ''
            else:
                current += c
        if current.strip():
            parts.append(current.strip())
        
        # 提取每个 key: value 的 value
        for part in parts:
            if ':' in part:
                _, value = part.split(':', 1)
                value = value.strip()
                try:
                    expr = sympify(value)
                    val = float(N(expr))
                    if not (val != val or abs(val) == float('inf')):
                        numbers.append(val)
                except:
                    pass
        return numbers
    
    def _extract_from_latex(self, s: str) -> List[float]:
        """从 LaTeX 字符串中提取数值，使用 math_verify 库（与 OpenMath 一致）"""
        from math_verify import LatexExtractionConfig, parse
        from sympy import N
        
        numbers = []
        s_clean = re.sub(r'^\$|\$$', '', s.strip())
        
        # 1. 检查是否是 LaTeX 矩阵/向量格式，如果是则分别解析每个元素
        if r'\begin{array}' in s_clean or r'\begin{matrix}' in s_clean or r'\begin{pmatrix}' in s_clean:
            # 提取矩阵内容
            matrix_content = re.search(r'\\begin\{(?:array|matrix|pmatrix)\}(?:\{[^}]*\})?(.*?)\\end\{(?:array|matrix|pmatrix)\}', s_clean, re.DOTALL)
            if matrix_content:
                content = matrix_content.group(1)
                # 按 \\ 分割行
                rows = re.split(r'\\\\', content)
                for row in rows:
                    row = row.strip()
                    if not row:
                        continue
                    # 按 & 分割列（如果有）
                    cells = row.split('&') if '&' in row else [row]
                    for cell in cells:
                        cell = cell.strip()
                        if not cell:
                            continue
                        # 解析单个单元格
                        cell_nums = self._parse_latex_expr(cell)
                        numbers.extend(cell_nums)
                if numbers:
                    return numbers
        
        # 2. 检查是否是 LaTeX 集合格式 {a, b, c}
        if r'\{' in s_clean and r'\}' in s_clean:
            # 提取集合内容
            set_match = re.search(r'\\{(.+?)\\}', s_clean, re.DOTALL)
            if set_match:
                content = set_match.group(1)
                # 按逗号分割
                elements = content.split(',')
                for elem in elements:
                    elem = elem.strip()
                    if not elem:
                        continue
                    elem_nums = self._parse_latex_expr(elem)
                    numbers.extend(elem_nums)
                if numbers:
                    return numbers
        
        # 3. 尝试直接解析为单个表达式
        nums = self._parse_latex_expr(s_clean)
        if nums:
            return nums
        
        # 4. 最后使用正则提取普通数值
        return self._extract_with_regex(s_clean)
    
    def _parse_latex_expr(self, expr: str) -> List[float]:
        """解析单个 LaTeX 表达式为数值"""
        from math_verify import LatexExtractionConfig, parse
        from sympy import N
        
        numbers = []
        expr = expr.strip()
        if not expr:
            return numbers
        
        # LILA 数据集中 \log 表示自然对数，但 math_verify 会解析成常用对数
        # 所以先把 \log 替换成 \ln
        expr_fixed = re.sub(r'\\log\s*(\^|\()', r'\\ln\1', expr)
        expr_fixed = re.sub(r'\\log\b', r'\\ln', expr_fixed)
        
        # 确保有 $ 符号
        expr_latex = f"${expr_fixed}$" if not re.search(r'\$.*\$', expr_fixed) else expr_fixed
        
        # 使用 math_verify 解析
        try:
            parsed = parse(expr_latex, [LatexExtractionConfig()], parsing_timeout=5)
            if parsed:
                # math_verify.parse 返回 [数值, 原始字符串]，只取第一个元素（数值）
                p = parsed[0] if parsed else None
                if p is not None and not isinstance(p, str):
                    try:
                        val = complex(N(p))
                        if val.imag == 0 and not (val.real != val.real):  # 排除 nan
                            numbers.append(val.real)
                    except:
                        pass
            if numbers:
                return numbers
        except:
            pass
        
        # 回退：手动解析常见格式
        # 处理 -\frac{...}{...} 格式（带负号）
        neg_frac = re.match(r'^-\\frac\{(.+)\}\{(.+)\}$', expr)
        if neg_frac:
            try:
                num_part = neg_frac.group(1)
                den_part = neg_frac.group(2)
                num_val = self._eval_latex_simple(num_part)
                den_val = self._eval_latex_simple(den_part)
                if num_val is not None and den_val is not None and den_val != 0:
                    numbers.append(-num_val / den_val)
                    return numbers
            except:
                pass
        
        # 处理 \frac{...}{...} 格式
        frac = re.match(r'^\\frac\{(.+)\}\{(.+)\}$', expr)
        if frac:
            try:
                num_part = frac.group(1)
                den_part = frac.group(2)
                num_val = self._eval_latex_simple(num_part)
                den_val = self._eval_latex_simple(den_part)
                if num_val is not None and den_val is not None and den_val != 0:
                    numbers.append(num_val / den_val)
                    return numbers
            except:
                pass
        
        return numbers
    
    def _eval_latex_simple(self, expr: str) -> Optional[float]:
        """简单求值 LaTeX 表达式（支持数字、sqrt、frac 的组合）"""
        from sympy import sqrt, Rational, N, sympify
        
        expr = expr.strip()
        
        # 纯数字
        try:
            return float(expr)
        except:
            pass
        
        # 替换 LaTeX 命令为 SymPy 格式
        s = expr
        # \sqrt{\frac{a}{b}} -> sqrt(Rational(a,b))
        s = re.sub(r'\\sqrt\{\\frac\{(\d+)\}\{(\d+)\}\}', r'sqrt(Rational(\1,\2))', s)
        # \sqrt{n} -> sqrt(n)
        s = re.sub(r'\\sqrt\{(\d+)\}', r'sqrt(\1)', s)
        # \frac{a}{b} -> Rational(a,b)
        s = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'Rational(\1,\2)', s)
        # 处理乘法（空格或相邻）
        s = re.sub(r'(\d)\s+sqrt', r'\1*sqrt', s)
        s = re.sub(r'(\))\s*(\d)', r'\1*\2', s)
        s = re.sub(r'(\d)\s*(\()', r'\1*\2', s)
        
        try:
            result = eval(s, {'sqrt': sqrt, 'Rational': Rational})
            return float(N(result))
        except:
            pass
        
        # 尝试 sympify
        try:
            result = sympify(s)
            return float(N(result))
        except:
            pass
        
        return None
    
    def _extract_with_regex(self, s: str) -> List[float]:
        """使用正则表达式提取数值"""
        from sympy import sympify, N
        
        numbers = []
        
        # 先尝试提取分数 (15/7)
        frac_pattern = r'-?\d+/\d+'
        for match in re.finditer(frac_pattern, s):
            try:
                val = float(N(sympify(match.group())))
                numbers.append(val)
            except:
                pass
        
        if numbers:
            return numbers
        
        # 提取普通数值（包括科学计数法）
        num_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        for match in re.finditer(num_pattern, s):
            try:
                val = float(match.group())
                if not (val != val or abs(val) == float('inf')):
                    numbers.append(val)
            except:
                pass
        
        return numbers
    
    def _all_expected_in_result(self, expected_nums: List[float], result_nums: List[float]) -> bool:
        """检查 expected 中的每个数值是否在 result 中能找到匹配"""
        if not expected_nums:
            return True
        if not result_nums:
            return False
        
        for exp in expected_nums:
            found = False
            for res in result_nums:
                if self._numbers_equal(exp, res):
                    found = True
                    break
            if not found:
                return False
        return True
    
    def _numbers_equal(self, a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.01) -> bool:
        """
        比较两个数值是否相等（允许 1% 相对误差或 0.01 绝对误差）
        """
        if a == b:
            return True
        
        # nan 和 inf 特殊处理
        if a != a or b != b:  # nan
            return False
        if abs(a) == float('inf') or abs(b) == float('inf'):
            return a == b
        
        diff = abs(a - b)
        max_val = max(abs(a), abs(b))
        
        if diff <= abs_tol:
            return True
        if max_val > 0 and diff / max_val <= rel_tol:
            return True
        
        return False

