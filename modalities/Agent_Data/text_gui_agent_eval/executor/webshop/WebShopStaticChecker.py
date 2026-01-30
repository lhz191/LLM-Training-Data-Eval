#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop 静态可执行性检查器

在 WebShop 仿真环境中执行 action 序列，验证是否可执行。

支持两种模式：
1. Text 模式（默认）：直接使用 WebAgentTextEnv，无需服务器
2. Browser 模式：通过 Flask 服务器进行验证

关于 Reward 和 Ground Truth (GT) 的说明：
- GT 定义在 WebShop 的 human_goals.json 中
- reward = (属性匹配数 + 选项匹配数 + 价格满足) / (属性数 + 选项数 + 1) × 类型匹配度
- reward < 1.0 是正常现象，反映了人类标注的真实情况
"""

import os
import sys
import re
import time
import random
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 添加 WebShop 路径
webshop_path = os.path.join(parent_dir, 'webshop')
if webshop_path not in sys.path:
    sys.path.insert(0, webshop_path)

from text_gui_executor import StaticExecutabilityChecker
from data_types import Record, Action

from .constants import DEFAULT_SERVER_URL
from .utils import (
    _remove_price_constraint,
    check_server_running,
    start_server_if_needed,
)

# 延迟导入 WebShop 官方模块（需要 spacy 等依赖）
HAS_WEBSHOP = False
try:
    from web_agent_site.engine.goal import (
        get_type_reward,
        get_attribute_reward,
        get_option_reward,
        get_reward,
    )
    from web_agent_site.engine.normalize import normalize_color
    HAS_WEBSHOP = True
except ImportError as e:
    print(f"⚠️ WebShop 模块导入失败: {e}")
    print("   WebShopStaticChecker 需要安装: pip install spacy && python -m spacy download en_core_web_sm")
    # 提供占位函数
    def get_type_reward(*args, **kwargs): return 0
    def get_attribute_reward(*args, **kwargs): return 0, 0
    def get_option_reward(*args, **kwargs): return 0, 0
    def get_reward(*args, **kwargs): return 0
    def normalize_color(s): return s

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

