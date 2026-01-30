#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMath 代码执行器
"""

import os
import sys
import re
import io
import traceback
import multiprocessing
import signal
from typing import Optional, Tuple, Any, List

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import CodeExecutor, TimeoutError

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
