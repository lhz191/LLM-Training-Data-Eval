#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop 工具函数
"""

import os
import sys
import re
import time
import subprocess
from typing import Optional

from .constants import DEFAULT_SERVER_URL


def _remove_price_constraint(text: str) -> str:
    """移除 instruction 中的价格约束部分，用于匹配"""
    # 移除 ", and price lower than X.XX dollars" 部分
    text = re.sub(r',?\s*and price lower than \d+\.?\d* dollars', '', text, flags=re.IGNORECASE)
    return text.strip().lower()


def check_server_running(server_url: str = DEFAULT_SERVER_URL, timeout: int = 10) -> bool:
    """检查 Flask 服务器是否在运行"""
    import requests
    try:
        response = requests.get(server_url, timeout=timeout, allow_redirects=False)
        # 服务器可能返回 200 或 302 重定向
        return response.status_code in [200, 302]
    except:
        return False


def start_server_if_needed(server_url: str = DEFAULT_SERVER_URL, wait_timeout: int = 300) -> bool:
    """如果服务器未运行，自动启动它
    
    Args:
        server_url: 服务器地址
        wait_timeout: 等待服务器启动的最大时间（秒）
    
    Returns:
        服务器是否成功运行
    """
    # 先检查是否已运行
    if check_server_running(server_url, timeout=5):
        return True
    
    print("🚀 Flask 服务器未运行，正在自动启动...")
    
    # 获取 webshop 目录路径（从 executor/webshop/ 回到 text_gui_agent_eval/webshop/）
    executor_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    webshop_dir = os.path.join(executor_dir, 'webshop')
    
    # 启动服务器进程
    env = os.environ.copy()
    env['PYTHONPATH'] = webshop_dir
    
    log_file = os.path.join(webshop_dir, 'flask_server.log')
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', 'web_agent_site/app.py'],
            cwd=webshop_dir,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True  # 独立进程组，不受父进程影响
        )
    
    print(f"   服务器 PID: {process.pid}")
    print(f"   日志文件: {log_file}")
    print(f"   等待服务器启动（可能需要 1-2 分钟加载数据）...")
    
    # 等待服务器启动
    start_time = time.time()
    check_interval = 5  # 每 5 秒检查一次
    
    while time.time() - start_time < wait_timeout:
        elapsed = int(time.time() - start_time)
        if check_server_running(server_url, timeout=10):
            print(f"   ✅ 服务器启动成功！耗时 {elapsed} 秒")
            return True
        
        # 检查进程是否还在运行
        if process.poll() is not None:
            print(f"   ❌ 服务器进程意外退出，退出码: {process.returncode}")
            print(f"   查看日志: cat {log_file}")
            return False
        
        print(f"   等待中... ({elapsed}s / {wait_timeout}s)")
        time.sleep(check_interval)
    
    print(f"   ❌ 服务器启动超时 ({wait_timeout}秒)")
    return False
