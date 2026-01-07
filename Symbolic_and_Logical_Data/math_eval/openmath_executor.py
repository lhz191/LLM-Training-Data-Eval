#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMathInstruct-1 代码执行器

包含 OpenMath 相关的提取器、执行器和比较器。
"""
import re
import sys
import io
from typing import Optional, Tuple, Any

from code_executor import (
    AnswerExtractor,
    CodeExtractor,
    CodeExecutor,
    ResultComparator,
    TimeoutError,
    _timeout_handler,
    compare_math_answers,
    compare_results,
    register_answer_extractor,
    register_code_extractor,
    register_executor,
)


# =============================================================================
# 答案提取器
# =============================================================================

class BoxedAnswerExtractor(AnswerExtractor):
    """
    提取 \\boxed{} 格式的答案
    
    适用于: OpenMathInstruct-1, NuminaMath, MATH 等
    
    策略：
    1. 找最后一个 \\boxed（rfind）
    2. 递归提取最内层的 \\boxed 内容
    3. 清理首尾的 $ 符号和多余空格
    """
    
    def _extract_single_boxed(self, text: str) -> Optional[str]:
        """从文本中提取最后一个 \\boxed{} 的内容"""
        # 找最后一个 \boxed
        idx = text.rfind("\\boxed")
        if idx < 0:
            return None
        
        # 找到 { 开始的位置
        i = idx + len("\\boxed")
        while i < len(text) and text[i] != '{':
            i += 1
        if i >= len(text):
            return None
        
        # 括号匹配
        start = i + 1
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        
        if depth == 0:
            return text[start:i-1]
        return None
    
    def extract(self, solution: str) -> Optional[str]:
        """
        提取最内层的 \\boxed{} 答案
        
        处理嵌套情况如：
        - \\boxed{\\boxed{36\\pi}} → 36\\pi
        - \\boxed{-3+26=\\boxed{23}} → 23
        """
        result = self._extract_single_boxed(solution)
        if result is None:
            return None
        
        # 递归提取最内层
        while '\\boxed{' in result:
            inner = self._extract_single_boxed(result)
            if inner is None:
                break
            result = inner
        
        # 清理
        result = result.strip()
        
        # 移除首尾的 $ 符号
        if result.startswith('$') and result.endswith('$'):
            result = result[1:-1].strip()
        
        return result


class DirectAnswerExtractor(AnswerExtractor):
    """
    直接答案提取器 - 不做任何提取，返回 None
    
    适用于: 答案直接存储在 ground_truth 字段的数据集（如 GSM8K-Aug, LILA）
    在 validity.py 中，如果 extract 返回 None，会直接使用 ground_truth
    """
    
    def extract(self, solution: str) -> Optional[str]:
        """不提取答案，返回 None"""
        return None


# =============================================================================
# 代码提取器
# =============================================================================

class OpenMathCodeExtractor(CodeExtractor):
    """
    OpenMathInstruct-1 代码提取器
    
    格式:
        <llm-code>
        Python 代码
        </llm-code>
        <llm-code-output>
        输出结果
        </llm-code-output>
    
    多代码块说明：
        OpenMath 的 solution 可能包含多个 <llm-code> 块（论文允许最多3个）。
        这些代码块通常是前后依赖的（后续代码块使用前面定义的变量）。
        
        - extract(): 只提取第一个代码块，用于 validity 指标验证
        - extract_all_code(): 提取所有代码块，用于 reasoning_validity 让 LLM 判断完整逻辑
    """
    
    def extract(self, solution: str) -> Optional[str]:
        """提取第一个 <llm-code>...</llm-code> 中的代码"""
        pattern = r'<llm-code>(.*?)</llm-code>'
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def extract_all_code(self, solution: str) -> Optional[str]:
        """
        提取所有 <llm-code>...</llm-code> 中的代码，合并返回
        
        用于 reasoning_validity 指标，让 LLM 看到完整的代码逻辑。
        多个代码块之间用分隔符连接。
        """
        pattern = r'<llm-code>(.*?)</llm-code>'
        matches = re.findall(pattern, solution, re.DOTALL)
        if matches:
            if len(matches) == 1:
                return matches[0].strip()
            else:
                # 多个代码块，用分隔符连接
                return '\n\n# --- 代码块分隔 ---\n\n'.join(m.strip() for m in matches)
        return None
    
    def extract_output(self, solution: str) -> Optional[str]:
        """提取第一个 <llm-code-output>...</llm-code-output> 中的预期输出"""
        pattern = r'<llm-code-output>(.*?)</llm-code-output>'
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


# =============================================================================
# 代码执行器
# =============================================================================

class OpenMathExecutor(CodeExecutor):
    """
    OpenMathInstruct-1 执行器（IPython 版本）
    
    使用 IPython 的 TerminalInteractiveShell 执行代码，这是最接近数据集生成环境的方式。
    
    执行策略：
    1. 每次执行前重置命名空间（shell.reset），保证样本独立
    2. 配置 SymPy 打印格式（use_unicode=True, pretty_print=False）
    3. 执行代码，捕获 stdout/stderr
    4. 后处理输出（移除 Out[X]: 前缀和 ANSI 颜色码）
    
    多进程兼容:
    - 检测进程 fork，在新进程中自动重建 shell
    - 支持在 multiprocessing.Pool worker 中使用
    - 不创建子进程，避免 daemon 进程限制
    
    超时保护:
    - 使用 signal.alarm 实现超时（仅限 Unix）
    - 默认超时 60 秒
    """
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._shell = None
        self._pid = None  # 记录创建 shell 的 PID，用于检测 fork
    
    def _ensure_shell(self):
        """确保 IPython shell 已创建，且属于当前进程"""
        import os
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        
        current_pid = os.getpid()
        
        # 如果是 fork 后的子进程，需要重新创建 shell
        if self._pid != current_pid or self._shell is None:
            # 清理旧实例（如果有的话）
            if self._shell is not None:
                try:
                    TerminalInteractiveShell.clear_instance()
                except:
                    pass
            
            # 创建新的 IPython shell
            self._shell = TerminalInteractiveShell.instance()
            self._pid = current_pid
    
    def _reset_namespace(self):
        """重置 shell 命名空间，避免变量污染和内存累积"""
        if self._shell is not None:
            # 使用 reset 清除用户变量
            try:
                self._shell.reset(new_session=False)
            except:
                pass
            
            # 强制垃圾回收，释放内存
            import gc
            gc.collect()
    
    def _postprocess_output(self, output: str) -> str:
        """
        后处理 IPython 输出，移除 Out[X]: 前缀和 ANSI 颜色码
        参考 NeMo-Skills 官方实现
        """
        # 移除 ANSI 颜色码
        ansi_escape = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
        output = ansi_escape.sub("", output)
        
        lines = []
        for line in output.split("\n"):
            # 跳过 IPython 文件行号
            if line.strip().startswith("File <ipython-"):
                continue
            # 移除 Out[X]: 前缀
            line = re.sub(r"^Out\[\d+\]:\s*", "", line)
            lines.append(line)
        
        # 移除开头的空行（displayhook 可能引入）
        while lines and lines[0] == "":
            lines.pop(0)
        
        output = "\n".join(lines).rstrip()
        return output
    
    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        执行代码并返回结果
        
        Args:
            code: 要执行的 Python 代码
            
        Returns:
            (result, error) 元组:
            - result: 执行结果（stdout 输出，已后处理）
            - error: 错误信息（如果有）
        """
        import signal
        from contextlib import redirect_stdout, redirect_stderr
        
        result = None
        error = None
        
        # 确保 shell 已创建（处理 fork 后的进程切换）
        self._ensure_shell()
        
        # 重置命名空间，避免变量污染
        self._reset_namespace()
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # 设置超时处理器
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # 配置 SymPy 打印格式（每次执行前设置，因为用户代码可能修改它）
                # - use_unicode=True: pprint() 输出 Unicode 格式
                # - pretty_print=False: displayhook 使用 str 格式
                self._shell.run_cell(
                    'try:\n'
                    '    from sympy import init_printing\n'
                    '    init_printing(use_unicode=True, pretty_print=False)\n'
                    'except: pass',
                    silent=True
                )
                
                # 执行用户代码
                exec_result = self._shell.run_cell(code)
            
            # 取消超时
            signal.alarm(0)
            
            # 检查是否有错误
            if exec_result.error_before_exec or exec_result.error_in_exec:
                error_info = exec_result.error_in_exec or exec_result.error_before_exec
                error = f"{type(error_info).__name__}: {str(error_info)}"
                result = None
            else:
                # 后处理输出
                raw_stdout = stdout_buf.getvalue()
                result = self._postprocess_output(raw_stdout)
                if not result:
                    result = None
                    
        except TimeoutError:
            error = f"TimeoutError: Code execution exceeded {self.timeout} seconds"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        return result, error


class OpenMathExecutorFast(CodeExecutor):
    """
    OpenMathInstruct-1 快速执行器（exec 版本）
    
    相比 OpenMathExecutor (IPython 版本)：
    - 速度快 ~10-50x
    - 准确率略低（不支持某些 IPython 特性，如重复的 from __future__ import）
    
    执行策略：
    1. 使用 AST 分析代码，找到最后一个顶层语句
    2. 如果是表达式语句（ast.Expr），包装成赋值语句来捕获其值
    3. 执行代码，捕获 stdout 输出
    4. 组合 stdout 输出和表达式值作为结果
    
    输出规则（模拟 IPython 行为）:
    - 所有 print() 的输出
    - 如果最后一个顶层语句是表达式，追加其值
    
    超时保护:
    - 使用 signal.alarm 实现超时（仅限 Unix）
    - 默认超时 60 秒
    """
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        执行代码并返回结果
        
        Args:
            code: 要执行的 Python 代码
            
        Returns:
            (result, error) 元组:
            - result: 执行结果（stdout 输出 + 最后表达式的值）
            - error: 错误信息（如果有）
        """
        import signal
        
        # 设置递归限制（IPython 默认是 3000，我们也用 3000 保持一致）
        old_recursion_limit = sys.getrecursionlimit()
        if old_recursion_limit < 3000:
            sys.setrecursionlimit(3000)
        
        exec_globals = {'__builtins__': __builtins__, '__name__': '__main__'}
        
        # 定义 display 函数（IPython 兼容）
        def display(*args, **kwargs):
            for arg in args:
                print(arg)
        exec_globals['display'] = display
        
        # 预配置 SymPy 输出格式（在 exec_globals 中，不修改用户代码）
        # 这避免了 from __future__ 冲突问题
        try:
            from sympy import init_printing
            init_printing(use_unicode=True)
        except:
            pass
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        result = None
        error = None
        
        # 设置超时处理器
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            # 使用 AST 分析代码，找到最后一个顶层语句
            # 如果是表达式语句（ast.Expr），将其包装成赋值语句来捕获值
            # 这是 IPython 自动显示最后表达式值的行为
            import ast
            
            _last_expr_result_var = '__last_expr_result__'
            modified_code = code
            
            try:
                tree = ast.parse(code)
                
                if tree.body:
                    last_stmt = tree.body[-1]
                    
                    # 检查最后一个语句是否是表达式语句
                    if isinstance(last_stmt, ast.Expr):
                        # 提取表达式的源代码
                        lines = code.split('\n')
                        start_line = last_stmt.lineno - 1  # 0-indexed
                        end_line = last_stmt.end_lineno - 1
                        start_col = last_stmt.col_offset
                        end_col = last_stmt.end_col_offset
                        
                        # 提取表达式代码
                        if start_line == end_line:
                            expr_code = lines[start_line][start_col:end_col]
                        else:
                            expr_lines = lines[start_line:end_line + 1]
                            expr_lines[0] = expr_lines[0][start_col:]
                            expr_lines[-1] = expr_lines[-1][:end_col]
                            expr_code = '\n'.join(expr_lines)
                        
                        # 构建修改后的代码：
                        # 1. 保留表达式之前的所有代码
                        # 2. 将表达式替换为赋值语句
                        before_lines = lines[:start_line]
                        after_lines = lines[end_line + 1:]
                        
                        # 保留表达式开始行之前的部分（如果有的话）
                        prefix = lines[start_line][:start_col] if start_col > 0 else ''
                        # 保留表达式结束行之后的部分（如果有的话）
                        suffix = lines[end_line][end_col:] if end_col < len(lines[end_line]) else ''
                        
                        # 生成赋值语句
                        assign_stmt = f'{_last_expr_result_var} = {expr_code}'
                        
                        # 组合修改后的代码
                        modified_lines = before_lines + [prefix + assign_stmt + suffix] + after_lines
                        modified_code = '\n'.join(modified_lines)
                        
            except SyntaxError:
                # 代码本身有语法错误，不做修改，让 exec 报错
                pass
            
            # 执行修改后的代码
            exec(modified_code, exec_globals)
            
            # 取消超时
            signal.alarm(0)
            
            # 获取 print 输出
            stdout_output = sys.stdout.getvalue().strip()
            
            # 获取最后一行表达式的值
            last_line_value = exec_globals.get(_last_expr_result_var, None)
            
            # 组合输出
            if stdout_output and last_line_value is not None:
                result = f"{stdout_output}\n{last_line_value}"
            elif stdout_output:
                result = stdout_output
            elif last_line_value is not None:
                result = last_line_value
            else:
                result = None
                
        except TimeoutError:
            error = f"TimeoutError: Code execution exceeded {self.timeout} seconds"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            
        finally:
            signal.alarm(0)  # 确保取消超时
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
            sys.stdout = old_stdout
            # 恢复递归限制
            if old_recursion_limit < 3000:
                sys.setrecursionlimit(old_recursion_limit)
        
        return result, error


# 保持向后兼容
PythonExecutor = OpenMathExecutor


# =============================================================================
# 结果比较器
# =============================================================================

class OpenMathResultComparator(ResultComparator):
    """
    OpenMath 数据集结果比较器
    
    - result: 单个执行结果
    - expected: 单个预期值（来自 llm-code-output 或 ground_truth）
    """
    
    def compare(self, result: Any, expected: Any) -> bool:
        """
        OpenMath: 直接比较 result 和 expected
        """
        if expected is None:
            return True
        
        # 如果 expected 是列表，取第一个（OpenMath 通常只有一个答案）
        if isinstance(expected, list):
            if len(expected) == 0:
                return True
            expected = expected[0]
        
        return compare_results(result, expected)


# =============================================================================
# 格式检查器
# =============================================================================

from code_executor import FormatChecker
from typing import List, Tuple as TypingTuple


class OpenMathFormatChecker(FormatChecker):
    """
    OpenMathInstruct-1 数据集格式检查器
    
    基于论文 Section 2.1 和 Section 2.3 的要求：
    
    必需字段检查：
    1. question: 问题文本（不能为空）
    2. solution: 解答文本（不能为空）
    3. ground_truth: 标准答案（不能为空）
    
    Solution 格式检查（论文 Section 2.3 Post-processing）：
    1. <llm-code> 和 </llm-code> 标签必须配对
    2. <llm-code-output> 和 </llm-code-output> 标签必须配对
    3. 不能有多个 \\boxed{} 块（论文明确要求移除，嵌套的算一个）
    """
    
    def _count_top_level_boxed(self, text: str) -> int:
        """
        统计顶层（非嵌套）的 \\boxed{} 数量
        
        例如：
        - "\\boxed{30}" -> 1
        - "\\boxed{\\boxed{30}}" -> 1 (嵌套算一个)
        - "\\boxed{30} and \\boxed{40}" -> 2 (两个独立的)
        """
        count = 0
        i = 0
        boxed_marker = '\\boxed{'
        
        while i < len(text):
            # 找下一个 \boxed{
            pos = text.find(boxed_marker, i)
            if pos == -1:
                break
            
            # 找到一个顶层 boxed，计数+1
            count += 1
            
            # 跳过这个 boxed 的内容（包括嵌套的）
            # 从 { 开始匹配括号
            brace_start = pos + len(boxed_marker) - 1  # 指向 {
            depth = 1
            j = brace_start + 1
            
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            
            # 跳到这个 boxed 结束后继续搜索
            i = j
        
        return count
    
    def check(self, sample) -> TypingTuple[List[str], List[str]]:
        """检查 OpenMath 样本的格式正确性"""
        errors = []
        warnings = []
        
        # === 1. 必需字段检查 ===
        if not sample.question:
            errors.append("Missing or empty 'question' field")
        
        if not sample.solution:
            errors.append("Missing or empty 'solution' field")
        
        if sample.ground_truth is None:
            errors.append("Missing 'ground_truth' field")
        elif isinstance(sample.ground_truth, str) and not sample.ground_truth.strip():
            errors.append("Empty 'ground_truth' field")
        elif isinstance(sample.ground_truth, list) and len(sample.ground_truth) == 0:
            errors.append("Empty 'ground_truth' list")
        
        # === 2. solution 格式检查 ===
        if sample.solution:
            solution = sample.solution
            if isinstance(solution, list):
                solution = "\n".join(solution)
            
            # 检查代码标签配对（论文 Section 2.3：移除有 <llm-code> 但没有 </llm-code> 的）
            code_open = solution.count('<llm-code>')
            code_close = solution.count('</llm-code>')
            if code_open != code_close:
                errors.append(f"Mismatched <llm-code> tags: {code_open} open, {code_close} close")
            
            # 检查代码输出标签配对
            output_open = solution.count('<llm-code-output>')
            output_close = solution.count('</llm-code-output>')
            if output_open != output_close:
                errors.append(f"Mismatched <llm-code-output> tags: {output_open} open, {output_close} close")
            
            # 检查多个 \boxed{}（论文 Section 2.3：移除有多个 \boxed{} 的）
            # 注意：嵌套的 \boxed{} 算一个，只统计顶层的
            boxed_count = self._count_top_level_boxed(solution)
            if boxed_count > 1:
                warnings.append(f"Multiple \\boxed{{}} blocks found: {boxed_count}")
        
        return errors, warnings



# =============================================================================
# 注册到全局注册表
# =============================================================================

# 注册答案提取器
register_answer_extractor('boxed', BoxedAnswerExtractor)
register_answer_extractor('openmath', BoxedAnswerExtractor)
register_answer_extractor('openmathinstruct1', BoxedAnswerExtractor)
register_answer_extractor('direct', DirectAnswerExtractor)

# 注册代码提取器
register_code_extractor('openmath', OpenMathCodeExtractor)
register_code_extractor('openmathinstruct1', OpenMathCodeExtractor)

# 注册执行器
register_executor('openmath', OpenMathExecutor)
register_executor('openmathinstruct1', OpenMathExecutor)
register_executor('openmathfast', OpenMathExecutorFast)
register_executor('openmathinstruct1fast', OpenMathExecutorFast)

