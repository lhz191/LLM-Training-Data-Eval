#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LILA 代码执行器
"""

import os
import sys
import io
from typing import Optional, Tuple, Any

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import CodeExecutor

class LILACodeExecutor(CodeExecutor):
    """
    LILA 数据集代码执行器
    
    执行 LILA 的 Python 程序，捕获 print 输出
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        执行 LILA 代码，返回 print 输出
        """
        exec_globals = {
            '__builtins__': __builtins__,
        }
        
        # 捕获 stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            exec(code, exec_globals)
            output = sys.stdout.getvalue().strip()
            return output if output else None, None
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout

