#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop 静态可执行性检查器

完全照搬 verify_il_traj.py 的逻辑：
- 在 WebShop 仿真环境中执行 action 序列
- 支持 Text 环境（无需服务器）和 Browser 环境（需要 Flask 服务器）
- 验证 action 是否在环境实时返回的 available_actions 中

注意：
- Text 环境需要 WebShop 的 web_agent_site 模块
- Browser 环境需要先启动 Flask 服务器: python -m web_agent_site.app --port 3000

=============================================================================
关于 Reward 和 Ground Truth (GT) 的说明
=============================================================================

【GT 来源】
GT 定义在 WebShop 的 human_goals.json 中，由 get_human_goals() 函数生成。
每个 goal 包含：
  - instruction_text: 用户的购物需求描述
  - asin: 目标商品的 ASIN（这是"正确答案"）
  - attributes: 要求的商品属性
  - goal_options: 要求的商品选项（如颜色、尺寸）
  - price_upper: 价格上限

【Reward 计算】
reward = (属性匹配数 + 选项匹配数 + 价格满足) / (属性数 + 选项数 + 1) × 类型匹配度

【为什么训练数据的 reward 可能不是 1.0？】
WebShop 训练数据是通过 Amazon Mechanical Turk 众包收集的：
1. 先定义好 goals（包含 instruction 和目标 ASIN）
2. MTurk 工人根据 instruction 执行购物任务
3. 工人的购物轨迹被记录为训练数据

但工人不一定选择预定义的"正确"商品，他们可能：
- 选择了更便宜的商品
- 选择了搜索结果中更靠前的商品
- 或者就是选错了

因此，reward < 1.0 是正常现象，反映了人类标注的真实情况。
=============================================================================
"""

import os
import sys
import re
import time
from typing import List, Dict, Any, Tuple, Optional

# 添加 WebShop 路径（复制到 text_gui_agent_eval/webshop 下）
WEBSHOP_PATH = os.path.join(os.path.dirname(__file__), 'webshop')
if WEBSHOP_PATH not in sys.path:
    sys.path.insert(0, WEBSHOP_PATH)

from text_gui_executor import (
    StaticExecutabilityChecker, 
    FormatChecker,
    HTMLLocator,
    register_static_checker,
    register_format_checker,
    register_html_locator,
)
from data_types import Record, Action

# 导入 WebShop 官方的 reward 计算函数
from web_agent_site.engine.goal import (
    get_type_reward,
    get_attribute_reward, 
    get_option_reward,
    get_reward,
)
from web_agent_site.engine.normalize import normalize_color


# =============================================================================
# 常量配置
# =============================================================================

DEFAULT_SERVER_URL = 'http://127.0.0.1:3000'


# =============================================================================
# 辅助函数
# =============================================================================

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
    import subprocess  # 仅在此函数使用
    
    # 先检查是否已运行
    if check_server_running(server_url, timeout=5):
        return True
    
    print("🚀 Flask 服务器未运行，正在自动启动...")
    
    # 获取 webshop 目录路径
    webshop_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webshop')
    
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


# =============================================================================
# WebShop 静态可执行性检查器
# =============================================================================

class WebShopStaticChecker(StaticExecutabilityChecker):
    """
    WebShop 静态可执行性检查器
    
    在 WebShop 仿真环境中执行 action 序列，验证是否可执行。
    
    支持两种模式：
    1. Text 模式（默认）：直接使用 WebAgentTextEnv，无需服务器
    2. Browser 模式：使用 WebAgentSiteEnv，需要先启动 Flask 服务器
    """
    
    def __init__(
        self,
        use_browser: bool = False,
        render: bool = False,
        server_url: str = DEFAULT_SERVER_URL,
        timeout: int = 30000,
    ):
        """
        初始化 WebShop 静态检查器
        
        Args:
            use_browser: 是否使用浏览器环境（需要先启动 Flask 服务器）
            render: 是否显示浏览器窗口（仅 Browser 模式）
            server_url: Flask 服务器地址（仅 Browser 模式）
            timeout: 超时时间（毫秒）
        """
        self.use_browser = use_browser
        self.render = render
        self.server_url = server_url
        self.timeout = timeout
        
        self._env = None
        self._env_type = "Browser" if use_browser else "Text"
        self._goals = None  # 缓存 goals（用于 Browser 环境的 goal 匹配）
    
    def _load_goals(self):
        """加载 goals（用于 Browser 环境的 goal 匹配）
        
        注意：必须使用与 Flask 服务器相同的 random seed，确保价格和顺序一致
        """
        if self._goals is not None:
            return self._goals
        
        import random
        from web_agent_site.engine.engine import load_products
        from web_agent_site.engine.goal import get_goals
        from web_agent_site.utils import DEFAULT_FILE_PATH
        
        all_products, _, product_prices, _ = load_products(
            filepath=DEFAULT_FILE_PATH,
            num_products=None,
            human_goals=1,  # 使用 human goals
        )
        
        # ★ 关键：在 get_goals() 之前设置 seed，确保价格生成一致
        random.seed(233)
        goals = get_goals(all_products, product_prices, human_goals=1)
        
        # shuffle 也用同一个 seed（此时 seed 状态已被 get_goals 内部的 random 调用改变）
        # 重新设置 seed 确保 shuffle 顺序一致
        random.seed(233)
        random.shuffle(goals)
        
        self._goals = goals
        return goals
    
    def _create_text_env(self):
        """创建 Text 环境"""
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        env = WebAgentTextEnv(
            observation_mode='text_rich',
            human_goals=1,
        )
        return env
    
    def _create_browser_env(self):
        """创建 Browser 环境"""
        from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
        env = WebAgentSiteEnv(
            observation_mode='text',
            render=self.render,
            server_url=self.server_url,
        )
        return env
    
    def _ensure_env(self) -> bool:
        """确保环境已创建"""
        if self._env is not None:
            return True
        
        try:
            if self.use_browser:
                # 自动启动服务器（如果未运行）
                if not start_server_if_needed(self.server_url):
                    print(f"❌ Flask 服务器启动失败: {self.server_url}")
                    return False
                self._env = self._create_browser_env()
            else:
                self._env = self._create_text_env()
            return True
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            return False
    
    def _close_env(self):
        """关闭环境"""
        if self._env is not None:
            try:
                if self.use_browser:
                    self._env.close()
            except:
                pass
            self._env = None
    
    def check(self, record: Record) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        在 WebShop 仿真环境中验证 Record 的 action 序列
        
        完全照搬 verify_il_traj.py 中 verify_trajectory 的逻辑
        """
        errors = []
        warnings = []
        
        actions = record.actions
        instruction = record.instruction
        
        print("=" * 80)
        print(f"验证 Record: {record.sample_id} (环境: {self._env_type})")
        print("=" * 80)
        
        # [1] 轨迹信息（由 loader 提供）
        print(f"\n[1] 轨迹信息...")
        print(f"    轨迹长度: {len(actions)} 步")
        
        # [2] Instruction
        print(f"\n[2] Instruction...")
        if instruction:
            print(f"    Instruction: {instruction[:80]}...")
        else:
            print(f"    警告: 无法提取 instruction")
        
        # [3] 创建环境
        print(f"\n[3] 创建 WebShop {self._env_type} 环境...")
        if not self._ensure_env():
            errors.append("Failed to create environment")
            return errors, warnings, {
                'total_actions': len(actions),
                'success_count': 0,
                'fail_count': len(actions),
                'success_rate': 0.0,
                'action_results': [],
            }
        
        env = self._env
        matched_goal_idx = None
        
        if self.use_browser:
            print(f"    已连接到服务器: {self.server_url}")
            # Browser 环境: 加载本地 goals 用于匹配
            goals = self._load_goals()
            print(f"    加载了 {len(goals)} 个 goals (本地缓存)")
        else:
            goals = env.server.goals
            print(f"    加载了 {len(goals)} 个 goals")
        
        # [4] 找到匹配的 goal (Text 和 Browser 环境都执行)
        print(f"\n[4] 匹配 goal...")
        gt_info = None  # Ground Truth 信息
        
        if instruction:
            # 调试信息
            instruction_core = _remove_price_constraint(instruction)
            print(f"\n    [调试] Record instruction (数据集):")
            print(f"    \"{instruction}\"")
            print(f"\n    [调试] 去除价格后 (用于匹配):")
            print(f"    \"{instruction_core}\"")
            
            # 在 goals 列表中查找匹配
            for idx, goal in enumerate(goals):
                goal_instruction = goal['instruction_text']
                goal_core = _remove_price_constraint(goal_instruction)
                if instruction_core == goal_core:
                    matched_goal_idx = idx
                    # 保存 GT 信息 (包含 reward 计算所需的所有字段)
                    gt_info = {
                        'goal_idx': idx,
                        'asin': goal.get('asin', ''),
                        'name': goal.get('name', ''),
                        'query': goal.get('query', ''),  # r_type 计算所需
                        'product_category': goal.get('product_category', ''),  # r_type 计算所需
                        'attributes': goal.get('attributes', []),
                        'goal_options': goal.get('goal_options', []),
                        'price_upper': goal.get('price_upper', 0),
                    }
                    break
            
            if matched_goal_idx is None:
                print(f"\n    ❌ 错误: 找不到匹配的 goal，终止验证")
                
                # 模糊匹配找最相似的
                from difflib import SequenceMatcher
                similar_goals = []
                for idx, goal in enumerate(goals):
                    goal_text = goal['instruction_text']
                    goal_core = _remove_price_constraint(goal_text)
                    sim = SequenceMatcher(None, instruction_core, goal_core).ratio()
                    similar_goals.append((idx, sim, goal_text))
                similar_goals.sort(key=lambda x: -x[1])
                
                print(f"\n    [诊断] 最相似的 goals:")
                for idx, sim, goal_text in similar_goals[:3]:
                    print(f"    [{idx}] 相似度 {sim:.2%}: \"{goal_text}\"")
                
                errors.append("No matching goal found in human_goals.json")
                return errors, warnings, {
                    'total_actions': len(actions),
                    'success_count': 0,
                    'fail_count': len(actions),
                    'success_rate': 0.0,
                    'task_completed': 0,
                    'final_reward': 0.0,
                    'task_success': 0,
                    'action_results': [],
                }
            else:
                env_goal_instruction = goals[matched_goal_idx]['instruction_text']
                env_goal_core = _remove_price_constraint(env_goal_instruction)
                print(f"\n    [调试] 环境 goal[{matched_goal_idx}] instruction (随机价格):")
                print(f"    \"{env_goal_instruction}\"")
                print(f"\n    [调试] 环境 goal 去除价格后:")
                print(f"    \"{env_goal_core}\"")
                print(f"\n    ✅ 匹配成功 (去除价格后一致)")
        
        # [5] 初始化环境
        print(f"\n[5] 初始化环境...")
        try:
            if self.use_browser:
                # Browser 环境: 使用 custom_<idx>_<price> 格式来指定 goal 并设置自定义价格
                if matched_goal_idx is not None:
                    # 从数据集 instruction 中提取价格 (re 已在文件顶部导入)
                    price_match = re.search(r'price lower than ([\d.]+) dollars', instruction)
                    if price_match:
                        dataset_price = price_match.group(1)
                        session_id = f"custom_{matched_goal_idx}_{dataset_price}"
                        print(f"    使用 session: {session_id} (使用数据集价格: ${dataset_price})")
                    else:
                        session_id = f"fixed_{matched_goal_idx}"
                        print(f"    使用 session: {session_id}")
                    obs, _ = env.reset(session=session_id)
                else:
                    obs, _ = env.reset()
                # 获取当前页面的 instruction
                browser_instruction = env.get_instruction_text()
                print(f"    Browser instruction: {browser_instruction[:80]}...")
            else:
                # Text 环境: 使用 goal_idx，并强制使用数据集中的 instruction
                # 这样可以确保价格约束和数据集一致
                env.server.assigned_instruction_text = instruction
                obs, _ = env.reset(session=matched_goal_idx)
                
                # 验证环境实际使用的 instruction
                actual_instruction = env.get_instruction_text()
                # 环境返回的可能有 "Instruction: " 前缀，去掉再比较
                actual_clean = actual_instruction.replace("Instruction: ", "").replace("Instruction:", "").strip()
                print(f"\n    [调试] 强制设置的 instruction (数据集的):")
                print(f"    \"{instruction}\"")
                print(f"\n    [调试] 环境实际使用的 instruction:")
                print(f"    \"{actual_clean}\"")
                if instruction.strip() == actual_clean:
                    print(f"\n    ✅ 价格约束设置成功")
                else:
                    print(f"\n    ⚠️ 价格约束可能不一致")
            
            
        except Exception as e:
            print(f"    错误: 初始化环境失败 - {e}")
            import traceback
            traceback.print_exc()
            errors.append(f"Failed to reset environment: {e}")
            return errors, warnings, {
                'total_actions': len(actions),
                'success_count': 0,
                'fail_count': len(actions),
                'success_rate': 0.0,
                'action_results': [],
            }
        
        # [6] 逐步执行动作
        print(f"\n[6] 执行动作序列...")
        print("-" * 80)
        
        results = []
        success_count = 0
        fail_count = 0
        
        try:
            for step_idx, action in enumerate(actions):
                action_str = action.action_repr  # 原始动作字符串
                
                print(f"\n  【Step {step_idx}】")
                print(f"    动作: {action_str}")
                
                # 获取环境当前的可用动作
                available = env.get_available_actions()
                
                # 检查动作是否可执行
                action_lower = action_str.lower()
                can_execute = False
                click_target = None
                reason = ""
                
                if action_lower.startswith('search['):
                    # search 动作总是可以执行（如果有搜索栏）
                    if available.get('has_search_bar', False):
                        can_execute = True
                        reason = "search_bar_available"
                    else:
                        can_execute = False
                        reason = "search_bar_not_available"
                        print(f"    ❌ 搜索栏不可用")
                        
                elif action_lower.startswith('click['):
                    # 提取点击目标
                    match = re.match(r'click\[(.+)\]', action_lower)
                    if match:
                        click_target = match.group(1)
                        # 检查是否在可点击列表中（大小写不敏感）
                        clickables = [c.lower() for c in available.get('clickables', [])]
                        can_execute = click_target in clickables
                        if not can_execute:
                            reason = "click_target_not_available"
                            print(f"    ❌ 点击目标不在可用列表中: {click_target}")
                            print(f"    可用点击项: {available.get('clickables', [])[:5]}...")
                        else:
                            reason = "click_target_available"
                    else:
                        can_execute = False
                        reason = "invalid_click_format"
                else:
                    can_execute = False
                    reason = "unknown_action_type"
                
                # Browser 模式下高亮要操作的元素
                if self.use_browser and self.render:
                    try:
                        if click_target:
                            element = env.highlight_action(click_target)
                            if element:
                                print(f"    🎯 已高亮目标元素")
                        elif action_lower.startswith('search['):
                            env.highlight_search_bar()
                            print(f"    🔍 已高亮搜索框")
                    except Exception:
                        pass
                
                # 执行动作
                executed = False
                reward = 0
                done = False
                
                if can_execute:
                    try:
                        obs, reward, done, info = env.step(action_str)
                        executed = True
                        success_count += 1
                        print(f"    ✅ 执行成功, reward={reward}, done={done}")
                        
                        if done:
                            print(f"\n    🏁 任务完成! 最终 reward: {reward}")
                    except Exception as e:
                        fail_count += 1
                        reason = f"execution_error: {e}"
                        print(f"    ❌ 执行异常: {e}")
                else:
                    fail_count += 1
                    errors.append(f"Step {step_idx}: action '{action_str}' not in available_actions")
                    print(f"    ❌ 动作不可执行")
                
                results.append({
                    'step': step_idx,
                    'action': action_str,
                    'can_execute': can_execute,
                    'executed': executed,
                    'reason': reason,
                    'reward': reward,
                    'done': done,
                })
                
                if done:
                    break
                
                # Browser 模式下加点延迟，方便观察
                if self.use_browser and self.render:
                    time.sleep(0.5)
        
        finally:
            # 确保关闭 Browser 环境
            if self.use_browser and self._env:
                try:
                    self._env.close()
                    print("\n    浏览器已关闭")
                except:
                    pass
                self._env = None
        
        # [7] 检查最终 reward 并提取实际购买的商品信息
        final_reward = 0.0
        task_completed = False
        actual_purchase = None  # 实际购买的商品信息
        
        # 从 actions 中提取实际购买的 ASIN（click[asin] 格式，ASIN 通常是大写字母+数字）
        purchased_asin = None
        purchased_options = []
        for action in actions:
            action_str = action.action_repr.lower()
            # 匹配 click[b0xxxxxxxx] 格式的 ASIN（10 位字母数字）
            asin_match = re.match(r'click\[([a-z0-9]{10})\]', action_str)
            if asin_match:
                purchased_asin = asin_match.group(1).upper()
            # 匹配选项（不是 ASIN，不是 buy now 等按钮）
            elif action_str.startswith('click[') and not any(x in action_str for x in ['buy now', 'back', 'prev', 'next', 'description', 'features', 'review']):
                option_match = re.match(r'click\[(.+)\]', action_str)
                if option_match:
                    opt = option_match.group(1)
                    if len(opt) != 10:  # 排除 ASIN
                        purchased_options.append(opt)
        
        actual_purchase = {
            'asin': purchased_asin,
            'selected_options': purchased_options,  # 用户选择的选项
        }
        
        # 尝试获取实际购买商品的详细信息（用于 reward 分析）
        # 这些字段与 WebShop 官方 reward 计算所需的字段对应
        purchased_product = None
        if purchased_asin and env:
            try:
                # Text 环境: 从 server.product_item_dict 获取
                if hasattr(env, 'server') and hasattr(env.server, 'product_item_dict'):
                    purchased_product = env.server.product_item_dict.get(purchased_asin, {})
                    if purchased_product:
                        # 基本信息
                        actual_purchase['name'] = purchased_product.get('name', purchased_product.get('Title', ''))
                        actual_purchase['category'] = purchased_product.get('category', '')
                        # r_type 计算所需字段
                        actual_purchase['query'] = purchased_product.get('query', '')
                        actual_purchase['product_category'] = purchased_product.get('product_category', '')
                        # 属性匹配所需字段
                        actual_purchase['attributes'] = purchased_product.get('Attributes', [])
                        actual_purchase['title'] = purchased_product.get('Title', actual_purchase['name'])
                        actual_purchase['bullet_points'] = purchased_product.get('BulletPoints', [])
                        actual_purchase['description'] = purchased_product.get('Description', '')
                    # 获取价格
                    if hasattr(env.server, 'product_prices'):
                        actual_purchase['price'] = env.server.product_prices.get(purchased_asin)
            except Exception:
                pass  # 获取失败时不影响主流程
        
        if results:
            last_result = results[-1]
            if last_result.get('done', False):
                final_reward = last_result.get('reward', 0.0)
                task_completed = True
                
                # 构建详细的错误/警告信息
                if final_reward < 1.0 and gt_info:
                    comparison = f"GT ASIN: {gt_info['asin']}, 实际购买 ASIN: {purchased_asin or 'unknown'}"
                    if gt_info['asin'] != purchased_asin:
                        comparison += " (ASIN 不匹配!)"
                    
                    if final_reward <= 0:
                        errors.append(f"Task completed but reward={final_reward}. {comparison}")
                    else:
                        warnings.append(f"Task completed with partial reward={final_reward}. {comparison}")
        
        # [8] 输出统计
        total = len(actions)
        success_rate = success_count / total if total > 0 else 0.0
        
        print("\n" + "=" * 80)
        print("验证结果:")
        print(f"  环境类型: {self._env_type}")
        print(f"  总步数: {total}")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  成功率: {success_rate * 100:.1f}%")
        print(f"  任务完成: {'是' if task_completed else '否'}")
        print(f"  最终 reward: {final_reward}")
        if task_completed and final_reward == 1.0:
            print(f"  ✅ 买到了正确的商品!")
        elif task_completed and 0 < final_reward < 1.0:
            print(f"  ⚠️ 任务完成但部分满足要求 (reward={final_reward})")
        elif task_completed and final_reward <= 0:
            print(f"  ❌ 任务完成但未买到正确商品")
        print("=" * 80)
        
        # WebShop 特有的返回格式
        stats = {
            'total_actions': total,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': success_rate,
            'task_completed': 1 if task_completed else 0,
            'final_reward': final_reward,
            'task_success': 1 if (task_completed and final_reward == 1.0) else 0,  # reward=1 才算成功
            'task_partial': 1 if (task_completed and 0 < final_reward < 1.0) else 0,  # 部分成功
            'action_results': results,
        }
        
        # 构建清晰的对比结构 (使用 WebShop 官方 reward 计算函数)
        # 直接调用官方函数：get_type_reward, get_attribute_reward, get_option_reward
        # 确保与环境返回的 reward 完全一致
        
        if task_completed and gt_info:
            from thefuzz import fuzz
            
            # 准备官方函数所需的数据结构
            gt_attrs = gt_info.get('attributes', [])
            gt_options = gt_info.get('goal_options', [])
            gt_price_upper = gt_info.get('price_upper', float('inf'))
            actual_price = actual_purchase.get('price')
            selected_opts = actual_purchase.get('selected_options', [])
            
            # 构造 purchased_product (官方函数所需格式)
            purchased_product = {
                'query': actual_purchase.get('query', ''),
                'product_category': actual_purchase.get('product_category', ''),
                'name': actual_purchase.get('name', ''),
                'Title': actual_purchase.get('title', actual_purchase.get('name', '')),
                'Attributes': actual_purchase.get('attributes', []),
                'BulletPoints': actual_purchase.get('bullet_points', []),
                'Description': actual_purchase.get('description', ''),
            }
            
            # 构造 goal (官方函数所需格式)
            goal = {
                'query': gt_info.get('query', ''),
                'product_category': gt_info.get('product_category', ''),
                'name': gt_info.get('name', ''),
                'attributes': gt_attrs,
                'goal_options': gt_options,
                'price_upper': gt_price_upper,
            }
            
            # 构造 options dict (官方函数所需格式)
            # 用户选择的选项转换为 dict 格式
            options_dict = {f'option_{i}': opt for i, opt in enumerate(selected_opts)}
            
            # ============ 使用官方函数计算 ============
            # 1. 类型匹配 (r_type)
            r_type_dict = get_type_reward(purchased_product, goal)
            r_type = r_type_dict['r_type']
            query_match = r_type_dict['query_match']
            category_match = r_type_dict['category_match']
            title_score = r_type_dict['title_score']
            
            # 2. 属性匹配
            r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)
            
            # 3. 选项匹配
            goal_options_for_check = goal['goal_options'].items() if isinstance(goal['goal_options'], dict) else goal['goal_options']
            r_option, num_option_matches = get_option_reward(
                list(options_dict.values()),
                goal_options_for_check
            )
            
            # 4. 价格检查
            r_price = (actual_price <= gt_price_upper) if (gt_price_upper > 0 and actual_price is not None) else None
            r_price_int = 1 if r_price else 0
            
            # 5. 计算理论 reward (与官方完全一致)
            denominator = len(gt_attrs) + len(gt_options) + 1
            if denominator > 0:
                theoretical_reward = r_type * (num_attr_matches + num_option_matches + (1 if r_price else 0)) / denominator
            else:
                theoretical_reward = 0.0
            
            # ============ 为日志构建详细的检查结果 ============
            # 属性检查详情 (复用官方逻辑但保留详细信息)
            attr_check_results = []
            product_attrs = purchased_product['Attributes']
            for g_attr in gt_attrs:
                found = False
                matched_with = None
                match_method = None
                
                # 方法1: 在 Attributes 列表中模糊匹配 (fuzz > 85)
                for p_attr in product_attrs:
                    score = fuzz.token_set_ratio(p_attr, g_attr)
                    if score > 85:
                        found = True
                        matched_with = p_attr
                        match_method = f"属性列表匹配 (fuzz={score}%)"
                        break
                
                # 方法2: 在 Title/BulletPoints/Description 中查找
                if not found:
                    if g_attr in purchased_product['Title'].lower():
                        found = True
                        match_method = "在商品标题(Title)中找到"
                    elif g_attr in ' '.join(purchased_product['BulletPoints']).lower():
                        found = True
                        match_method = "在商品特性(BulletPoints)中找到"
                    elif g_attr in purchased_product['Description'].lower():
                        found = True
                        match_method = "在商品描述(Description)中找到"
                
                attr_check_results.append({
                    'required': g_attr,
                    'found': found,
                    'matched_with': matched_with,
                    'match_method': match_method
                })
            
            # 选项检查详情
            opt_check_results = []
            normalized_selected = [normalize_color(o) for o in selected_opts]
            normalized_goal_opts = [normalize_color(o) if isinstance(o, str) else normalize_color(str(o)) for o in gt_options]
            
            for i, g_opt in enumerate(gt_options):
                g_opt_str = g_opt if isinstance(g_opt, str) else str(g_opt)
                g_opt_normalized = normalized_goal_opts[i]
                found = False
                matched_with = None
                best_score = 0
                
                for j, s_opt in enumerate(selected_opts):
                    s_opt_normalized = normalized_selected[j]
                    score = fuzz.token_set_ratio(s_opt_normalized, g_opt_normalized)
                    if score > 85:
                        found = True
                        matched_with = s_opt
                        best_score = score
                        break
                    if score > best_score:
                        best_score = score
                
                opt_check_results.append({
                    'required': g_opt_str,
                    'selected': found,
                    'matched_with': matched_with,
                    'best_score': best_score if best_score > 0 else None
                })
            
            # 构建对比结构
            comparison = {
                # 1. 商品类型匹配 (r_type) - 使用官方函数结果
                'type_match': {
                    'goal_query': goal['query'],
                    'actual_query': purchased_product['query'],
                    'query_match': query_match,
                    'goal_category': goal['product_category'],
                    'actual_category': purchased_product['product_category'],
                    'category_match': category_match,
                    'goal_name': goal['name'],
                    'actual_name': purchased_product['name'],
                    'title_score': round(title_score, 4),  # 官方使用 spaCy 计算的相似度
                    'r_type': r_type
                },
                # 2. 属性匹配 - 使用官方函数结果
                'attributes': {
                    'required': gt_attrs,
                    'product_has': product_attrs,
                    'check_results': attr_check_results,
                    'num_matches': num_attr_matches,
                    'total': len(gt_attrs),
                    'r_att': round(r_att, 4) if r_att is not None else None
                },
                # 3. 选项匹配 - 使用官方函数结果
                'options': {
                    'required': gt_options,
                    'selected': selected_opts,
                    'check_results': opt_check_results,
                    'num_matches': num_option_matches,
                    'total': len(gt_options),
                    'r_option': round(r_option, 4) if r_option is not None else None
                },
                # 4. 价格检查
                'price': {
                    'limit': gt_price_upper if gt_price_upper < float('inf') else None,
                    'actual': actual_price,
                    'within_budget': r_price,
                    'r_price': r_price_int
                },
                # 5. 汇总 - 完整的 reward 计算公式
                'summary': {
                    'formula': f"r_type × (num_attr + num_opt + r_price) / (len_attr + len_opt + 1)",
                    'r_type': r_type,
                    'num_attr_matches': num_attr_matches,
                    'num_option_matches': num_option_matches,
                    'r_price': r_price_int,
                    'denominator': denominator,
                    'calculation': f"{r_type} × ({num_attr_matches} + {num_option_matches} + {r_price_int}) / {denominator}",
                    'theoretical_reward': round(theoretical_reward, 4),
                    'actual_reward': final_reward
                }
            }
            
            # 生成不匹配原因列表
            mismatch_reasons = []
            
            if r_type < 1.0:
                if r_type == 0.0:
                    mismatch_reasons.append(f"商品类型完全不匹配 (r_type=0): query不同, category无交集, 标题相似度=0")
                elif r_type == 0.1:
                    mismatch_reasons.append(f"商品类型基本不匹配 (r_type=0.1): 标题相似度={title_score:.1%} < 10%")
                elif r_type == 0.5:
                    mismatch_reasons.append(f"商品类型部分不匹配 (r_type=0.5): query不同, category交集<2, 标题相似度={title_score:.1%}")
            
            for check in attr_check_results:
                if not check['found']:
                    mismatch_reasons.append(f"属性未满足: 要求 '{check['required']}', 商品不具备此属性")
            
            for check in opt_check_results:
                if not check['selected']:
                    if check['best_score']:
                        mismatch_reasons.append(f"选项未选: 要求 '{check['required']}', 最接近的选项相似度仅 {check['best_score']}%")
                    else:
                        mismatch_reasons.append(f"选项未选: 要求 '{check['required']}', 用户未选择任何相关选项")
            
            if r_price == False:
                mismatch_reasons.append(f"价格超限: 要求 ≤${gt_price_upper:.2f}, 实际 ${actual_price:.2f}")
            
            stats['comparison'] = comparison
            if mismatch_reasons:
                stats['mismatch_reasons'] = mismatch_reasons
            else:
                stats['match_status'] = "✅ 完美匹配: 商品类型、属性、选项、价格全部符合要求"
        else:
            # 未完成任务时，只保存基本信息
            if gt_info:
                stats['ground_truth'] = gt_info
            if actual_purchase:
                stats['actual_purchase'] = actual_purchase
        
        return errors, warnings, stats
    
    def __del__(self):
        """析构时关闭环境"""
        self._close_env()


# =============================================================================
# 格式检查器
# =============================================================================

class WebShopFormatChecker(FormatChecker):
    """
    WebShop 数据格式检查器
    
    检查项：
    1. Record 级别
       - instruction 是否存在且非空
       - actions 是否存在且非空
       
    2. Action 级别
       - action 格式是否正确（search[xxx] 或 click[xxx]）
       - action_type 是否为 search 或 click
       - cleaned_html (state) 是否存在
       - 对于 click 操作，target 是否在 candidates (available_actions) 中
    """
    
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """检查 WebShop Record 的数据格式"""
        errors = []
        warnings = []
        
        # === 1. Record 级别检查 ===
        
        # instruction
        if not record.instruction or not record.instruction.strip():
            errors.append("Record has empty 'instruction'")
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, action_warnings = self._check_action(action, i)
            errors.extend(action_errors)
            warnings.extend(action_warnings)
        
        return errors, warnings
    
    def _check_action(self, action: Action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []
        prefix = f"Action[{idx}]"
        
        # action_type
        action_type = action.action_type
        if action_type not in ('search', 'click'):
            errors.append(f"{prefix}: invalid action_type '{action_type}', expected 'search' or 'click'")
        
        # action_repr 格式检查
        action_repr = action.action_repr
        if action_repr:
            if action_type == 'search':
                if not action_repr.startswith('search[') or not action_repr.endswith(']'):
                    errors.append(f"{prefix}: invalid search format: {action_repr}")
            elif action_type == 'click':
                if not action_repr.startswith('click[') or not action_repr.endswith(']'):
                    errors.append(f"{prefix}: invalid click format: {action_repr}")
        else:
            errors.append(f"{prefix}: missing 'action_repr'")
        
        # cleaned_html (state)
        state = action.cleaned_html or ''
        if not state:
            errors.append(f"{prefix}: empty 'cleaned_html' (state)")
        
        # 对于 search 操作，检查是否有搜索内容
        if action_type == 'search':
            if not action.action_value or not action.action_value.strip():
                errors.append(f"{prefix}: search must have action_value (search content)")
            
            # 检查 [button] Search [button_] 是否在 state 中
            if state and '[button] search [button_]' not in state.lower():
                errors.append(f"{prefix}: search button not found in state")
        
        # 对于 click 操作，检查 target 是否在 candidates 中
        elif action_type == 'click':
            target = action.target_element
            candidates = action.candidates
            
            if target and candidates:
                # WebShop 的 candidates 是 available_actions 列表
                # target_element 是 action_translate（商品名版本）
                if target not in candidates:
                    # 尝试匹配原始 action_repr
                    if action_repr not in candidates:
                        errors.append(f"{prefix}: target not in available_actions")
            
            # 检查 action_repr 是否在 state 中可定位
            # 格式: [button] xxx [button_] 或 [clicked button] xxx [clicked button_]
            if state and action_repr:
                click_target = action_repr[6:-1].lower() if action_repr.startswith('click[') else ''
                if click_target:
                    state_lower = state.lower()
                    pattern1 = f'[button] {click_target} [button_]'
                    pattern2 = f'[clicked button] {click_target} [clicked button_]'
                    if pattern1 not in state_lower and pattern2 not in state_lower:
                        errors.append(f"{prefix}: target not found in state")
        
        return errors, warnings


# =============================================================================
# 注册检查器
# =============================================================================

# =============================================================================
# HTML 定位器
# =============================================================================

class WebShopLocator(HTMLLocator):
    """
    WebShop HTML 定位器
    
    定位方式：通过 [button] xxx [button_] 模式
    
    WebShop 的 state/cleaned_html 是文本格式，按钮格式如：
    - [button] B09LM1Y6QZ [button_]  (ASIN)
    - [button] Features [button_]    (文本)
    - [clicked button] xxx [clicked button_]  (已点击)
    
    定位率预期：100%（所有 target 都在 state 中）
    """
    
    def can_locate(self, action: Action, html: str) -> Tuple[bool, str]:
        """
        检查是否能在 state 中定位到 target
        
        Args:
            action: Action 对象
            html: state 字符串（可以是 raw_html 或 cleaned_html）
            
        Returns:
            (success, reason)
        """
        if not html:
            return False, "empty_html"
        
        action_type = action.action_type
        action_repr = action.action_repr
        
        if not action_repr:
            return False, "no_action_repr"
        
        html_lower = html.lower()
        
        if action_type == 'search':
            # search: 检查 [button] search [button_]
            if '[button] search [button_]' in html_lower:
                return True, "found"
            else:
                return False, "search_button_not_found"
        
        elif action_type == 'click':
            # click: 提取 target，检查 [button] xxx [button_] 或 [clicked button] xxx [clicked button_]
            if action_repr.startswith('click[') and action_repr.endswith(']'):
                target = action_repr[6:-1].lower()
                pattern1 = f'[button] {target} [button_]'
                pattern2 = f'[clicked button] {target} [clicked button_]'
                
                if pattern1 in html_lower or pattern2 in html_lower:
                    return True, "found"
                else:
                    return False, "target_not_found"
            else:
                return False, "invalid_action_format"
        
        else:
            return False, f"unknown_action_type_{action_type}"


# =============================================================================
# 注册检查器和定位器
# =============================================================================

register_static_checker('webshop', WebShopStaticChecker)
register_format_checker('webshop', WebShopFormatChecker)
register_html_locator('webshop', WebShopLocator)


# =============================================================================
# 命令行测试
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WebShop 静态可执行性检查')
    parser.add_argument('--data-path', type=str, 
                        default='/mnt/petrelfs/liuhaoze/main/Agent_Data/webshop/baseline_models/data/il_trajs_finalized_images.jsonl',
                        help='WebShop 数据文件路径')
    parser.add_argument('--browser', action='store_true',
                        help='使用浏览器环境（需要先启动 Flask 服务器）')
    parser.add_argument('--render', action='store_true',
                        help='显示浏览器窗口（仅 Browser 模式）')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL,
                        help=f'Flask 服务器地址（默认: {DEFAULT_SERVER_URL}）')
    parser.add_argument('--batch', type=int, default=3,
                        help='测试的记录数量')
    
    args = parser.parse_args()
    
    # 导入 loader
    from loaders import WebShopLoader
    
    print("=" * 60)
    print("WebShop 静态可执行性检查")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"环境模式: {'Browser' if args.browser else 'Text'}")
    print(f"测试数量: {args.batch}")
    print()
    
    # 加载数据
    loader = WebShopLoader(args.data_path)
    
    # 创建检查器
    checker = WebShopStaticChecker(
        use_browser=args.browser,
        render=args.render,
        server_url=args.server_url,
    )
    
    # 测试
    total_success = 0
    total_actions = 0
    
    try:
        for i, record in enumerate(loader.iterate()):
            if i >= args.batch:
                break
            
            print(f"\n[{i+1}/{args.batch}] {record.sample_id}")
            errors, warnings, stats = checker.check(record)
            
            total_success += stats['success_count']
            total_actions += stats['total_actions']
    finally:
        checker._close_env()
    
    # 汇总
    print("\n" + "=" * 60)
    print("总体汇总")
    print("=" * 60)
    print(f"测试记录数: {min(args.batch, i+1)}")
    print(f"总动作数: {total_actions}")
    print(f"成功动作数: {total_success}")
    print(f"总体成功率: {total_success / total_actions * 100:.1f}%" if total_actions > 0 else "N/A")


if __name__ == '__main__':
    main()
