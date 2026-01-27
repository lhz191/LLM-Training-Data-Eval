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

from text_gui_executor import StaticExecutabilityChecker, register_static_checker
from data_types import Record, Action


# =============================================================================
# 常量配置
# =============================================================================

DEFAULT_SERVER_URL = 'http://127.0.0.1:3000'


# =============================================================================
# 辅助函数
# =============================================================================

def find_goal_idx(env, instruction: str) -> Optional[int]:
    """在环境的 goals 中找到匹配的 goal 索引（精确匹配）"""
    if not hasattr(env, 'server') or not hasattr(env.server, 'goals'):
        return None
    
    goals = env.server.goals
    instruction_lower = instruction.strip().lower()
    
    for i, goal in enumerate(goals):
        if goal['instruction_text'].strip().lower() == instruction_lower:
            return i
    
    return None


def check_server_running(server_url: str = DEFAULT_SERVER_URL) -> bool:
    """检查 Flask 服务器是否在运行"""
    import requests
    try:
        response = requests.get(server_url, timeout=5)
        return response.status_code == 200
    except:
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
                if not check_server_running(self.server_url):
                    print(f"❌ Flask 服务器未运行: {self.server_url}")
                    print(f"   请先启动服务器: python -m web_agent_site.app --port 3000")
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
        else:
            print(f"    加载了 {len(env.server.goals)} 个 goals")
            
            # [4] 找到匹配的 goal (仅 Text 环境)
            print(f"\n[4] 匹配 goal...")
            if instruction:
                matched_goal_idx = find_goal_idx(env, instruction)
                if matched_goal_idx is None:
                    print(f"    警告: 找不到匹配的 goal，使用随机 goal")
                    warnings.append("No matching goal found, using random goal")
                else:
                    print(f"    找到匹配的 goal, 索引: {matched_goal_idx}")
                    print(f"    Goal instruction: {env.server.goals[matched_goal_idx]['instruction_text'][:80]}...")
        
        # [5] 初始化环境
        print(f"\n[5] 初始化环境...")
        try:
            if self.use_browser:
                # Browser 环境: 使用 fixed_<idx> 格式来指定 goal
                if matched_goal_idx is not None:
                    session_id = f"fixed_{matched_goal_idx}"
                    print(f"    使用 session: {session_id}")
                    obs, _ = env.reset(session=session_id)
                else:
                    obs, _ = env.reset()
                # 获取当前页面的 instruction
                browser_instruction = env.get_instruction_text()
                print(f"    Browser instruction: {browser_instruction[:80]}...")
            else:
                # Text 环境: 使用 goal_idx
                obs, _ = env.reset(session=matched_goal_idx)
            
            obs_preview = obs[:200] if isinstance(obs, str) else str(obs)[:200]
            print(f"    初始观察 (前200字符):\n    {obs_preview}...")
            
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
                    # 尝试强制执行看看会发生什么
                    try:
                        obs, reward, done, info = env.step(action_str)
                        executed = True
                        print(f"    ⚠️ 动作不在可用列表但执行了, reward={reward}, done={done}")
                        warnings.append(f"Step {step_idx}: action not in available but executed")
                        
                        if done:
                            print(f"\n    🏁 任务完成! 最终 reward: {reward}")
                    except Exception as e:
                        print(f"    ❌ 执行失败: {e}")
                
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
        
        # [7] 输出统计
        total = len(actions)
        success_rate = success_count / total if total > 0 else 0.0
        
        print("\n" + "=" * 80)
        print("验证结果:")
        print(f"  环境类型: {self._env_type}")
        print(f"  总步数: {total}")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  成功率: {success_rate * 100:.1f}%")
        print("=" * 80)
        
        return errors, warnings, {
            'total_actions': total,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': success_rate,
            'action_results': results,
        }
    
    def __del__(self):
        """析构时关闭环境"""
        self._close_env()


# =============================================================================
# 注册检查器
# =============================================================================

register_static_checker('webshop', WebShopStaticChecker)


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
