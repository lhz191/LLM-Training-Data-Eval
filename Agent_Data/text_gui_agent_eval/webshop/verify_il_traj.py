#!/usr/bin/env python3
"""
WebShop 训练数据验证工具
验证 il_trajs_finalized_images.jsonl 中的动作序列是否能在环境中真实执行

支持两种环境:
1. Text 环境 (默认): 直接读取 JSON 数据，无需启动服务器
2. Browser 环境: 需要先启动 Flask 服务器，使用 Selenium 操作浏览器

用法: 
  python verify_il_traj.py <轨迹索引>                    # Text 环境
  python verify_il_traj.py <轨迹索引> --browser          # Browser 环境 (headless)
  python verify_il_traj.py <轨迹索引> --browser --render # Browser 环境 (显示浏览器)
  python verify_il_traj.py <轨迹索引> --goal-idx 5       # 指定 goal 索引
  python verify_il_traj.py <轨迹索引> --compare-states   # 比较记录的 states 与环境 obs

注意: 使用 Browser 环境前需要先启动服务器:
  python -m web_agent_site.app --port 3000
"""

import argparse
import json
import sys
import os
import re
import time
from difflib import SequenceMatcher

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 配置
IL_TRAJ_PATH = './baseline_models/data/il_trajs_finalized_images.jsonl'
DEFAULT_SERVER_URL = 'http://127.0.0.1:3000'


def similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def extract_instruction(state):
    """从 state 文本中提取 instruction"""
    # 格式: "Amazon Shopping Game\nInstruction: \nxxx\n[button]..."
    # 或: "Instruction:\nxxx\n[button]..."
    lines = state.strip().split('\n')
    instruction = None
    capture = False
    for line in lines:
        if 'Instruction:' in line:
            capture = True
            # 如果 Instruction: 后面有内容
            after = line.split('Instruction:')[-1].strip()
            if after:
                instruction = after
                break
            continue
        if capture:
            if line.startswith('[button]') or line.startswith('[clicked'):
                break
            if line.strip():
                instruction = line.strip()
                break
    return instruction


def load_traj(traj_idx, traj_path=None):
    """加载指定索引的轨迹
    
    Args:
        traj_idx: 轨迹索引
        traj_path: 轨迹文件路径 (默认使用 IL_TRAJ_PATH)
    """
    path = traj_path if traj_path else IL_TRAJ_PATH
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == traj_idx:
                return json.loads(line)
    return None


def find_goal_idx(env, instruction, threshold=0.8):
    """在环境的 goals 中找到匹配的 goal 索引
    
    Args:
        env: 环境实例 (Text 环境需要有 server.goals)
        instruction: 要匹配的 instruction 文本
        threshold: 相似度阈值 (默认 0.8)
    
    Returns:
        匹配的 goal 索引，如果没找到返回 None
    """
    if not hasattr(env, 'server') or not hasattr(env.server, 'goals'):
        return None
    
    goals = env.server.goals
    instruction = instruction.strip().lower()
    
    # 1. 精确匹配
    for i, goal in enumerate(goals):
        if goal['instruction_text'].strip().lower() == instruction:
            return i
    
    # 2. 包含匹配
    for i, goal in enumerate(goals):
        goal_text = goal['instruction_text'].strip().lower()
        if instruction in goal_text or goal_text in instruction:
            return i
    
    # 3. 相似度匹配
    best_idx = None
    best_score = 0
    for i, goal in enumerate(goals):
        goal_text = goal['instruction_text'].strip().lower()
        score = similarity(instruction, goal_text)
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_score >= threshold:
        return best_idx
    
    return None


def normalize_action(action):
    """标准化动作格式"""
    # 去除空格，统一小写
    action = action.lower().strip()
    return action


def create_text_env():
    """创建 Text 环境"""
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(
        observation_mode='text_rich',
        human_goals=1,  # 使用人类标注的 goals
    )
    return env


def create_browser_env(render=False, server_url=DEFAULT_SERVER_URL):
    """创建 Browser 环境
    
    Args:
        render: 是否显示浏览器窗口
        server_url: Flask 服务器地址
    
    Returns:
        WebAgentSiteEnv 实例
    """
    from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
    
    env = WebAgentSiteEnv(
        observation_mode='text',  # 使用 text 模式便于比较
        render=render,
        server_url=server_url,
    )
    return env


def check_server_running(server_url=DEFAULT_SERVER_URL):
    """检查 Flask 服务器是否在运行"""
    import requests
    try:
        response = requests.get(server_url, timeout=5)
        return response.status_code == 200
    except:
        return False


def verify_trajectory(traj_idx, use_browser=False, render=False, server_url=DEFAULT_SERVER_URL, 
                      goal_idx=None, compare_states=False, traj_path=None, interactive=False):
    """验证单条轨迹
    
    Args:
        traj_idx: 轨迹索引
        use_browser: 是否使用浏览器环境
        render: 是否显示浏览器窗口 (仅浏览器模式)
        server_url: Flask 服务器地址 (仅浏览器模式)
        goal_idx: 手动指定的 goal 索引 (可选)
        compare_states: 是否比较记录的 states 与环境返回的 obs
        traj_path: 轨迹文件路径 (可选)
        interactive: 是否启用交互模式，每步等待用户按 Enter (仅浏览器模式)
    """
    env_type = "Browser" if use_browser else "Text"
    
    print("=" * 80)
    print(f"验证轨迹索引: {traj_idx} (环境: {env_type})")
    print("=" * 80)
    
    # 1. 加载轨迹
    print("\n[1] 加载轨迹数据...")
    traj = load_traj(traj_idx, traj_path)
    if traj is None:
        print(f"错误: 找不到索引 {traj_idx} 的轨迹")
        return False
    
    actions = traj['actions']
    states = traj['states']
    print(f"    轨迹长度: {len(actions)} 步")
    
    # 2. 提取 instruction
    print("\n[2] 提取 instruction...")
    instruction = extract_instruction(states[0])
    if instruction:
        print(f"    Instruction: {instruction[:80]}...")
    else:
        print("    警告: 无法提取 instruction")
    
    # 3. 创建环境
    print(f"\n[3] 创建 WebShop {env_type} 环境...")
    env = None
    matched_goal_idx = goal_idx  # 使用用户指定的，或者后面自动匹配
    
    try:
        if use_browser:
            # 检查服务器是否运行
            if not check_server_running(server_url):
                print(f"    错误: Flask 服务器未运行!")
                print(f"    请先启动服务器: python -m web_agent_site.app --port 3000")
                return False
            
            env = create_browser_env(render=render, server_url=server_url)
            print(f"    已连接到服务器: {server_url}")
        else:
            env = create_text_env()
            print(f"    加载了 {len(env.server.goals)} 个 goals")
            
            # 4. 找到匹配的 goal (仅 Text 环境，如果没有手动指定)
            print("\n[4] 匹配 goal...")
            if matched_goal_idx is not None:
                print(f"    使用手动指定的 goal 索引: {matched_goal_idx}")
                if matched_goal_idx < len(env.server.goals):
                    print(f"    Goal instruction: {env.server.goals[matched_goal_idx]['instruction_text'][:80]}...")
            else:
                matched_goal_idx = find_goal_idx(env, instruction)
                if matched_goal_idx is None:
                    print(f"    警告: 找不到匹配的 goal，使用随机 goal")
                    # 不指定 session，让环境随机选择
                else:
                    print(f"    找到匹配的 goal, 索引: {matched_goal_idx}")
                    print(f"    Goal instruction: {env.server.goals[matched_goal_idx]['instruction_text'][:80]}...")
    except Exception as e:
        print(f"    错误: 创建环境失败 - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 初始化环境
    print("\n[5] 初始化环境...")
    try:
        if use_browser:
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
        
        # 比较初始状态
        if compare_states and len(states) > 0:
            print(f"\n    [比较] 记录的初始状态 (前200字符):")
            print(f"    {states[0][:200]}...")
            sim = similarity(obs, states[0])
            print(f"    相似度: {sim:.2%}")
            
    except Exception as e:
        print(f"    错误: 初始化环境失败 - {e}")
        import traceback
        traceback.print_exc()
        if env:
            try:
                env.close()
            except:
                pass
        return False
    
    # 6. 逐步执行动作
    print("\n[6] 执行动作序列...")
    print("-" * 80)
    
    success_count = 0
    fail_count = 0
    
    try:
        for step_idx, action in enumerate(actions):
            print(f"\n  【Step {step_idx}】")
            print(f"    动作: {action}")
            
            # 获取可用动作
            available = env.get_available_actions()
            
            # 检查动作是否可执行
            action_lower = action.lower()
            click_target = None
            
            if action_lower.startswith('search['):
                # search 动作总是可以执行（如果有搜索栏）
                if available['has_search_bar']:
                    can_execute = True
                else:
                    can_execute = False
                    print(f"    ❌ 搜索栏不可用")
            elif action_lower.startswith('click['):
                # 提取点击目标
                match = re.match(r'click\[(.+)\]', action_lower)
                if match:
                    click_target = match.group(1)
                    # 检查是否在可点击列表中（大小写不敏感）
                    can_execute = click_target in [c.lower() for c in available['clickables']]
                    if not can_execute:
                        print(f"    ❌ 点击目标不在可用列表中: {click_target}")
                        print(f"    可用点击项: {available['clickables'][:5]}...")
                else:
                    can_execute = False
            else:
                can_execute = False
            
            # Browser 模式下高亮要操作的元素
            if use_browser and render:
                try:
                    if click_target:
                        # 高亮点击目标
                        element = env.highlight_action(click_target)
                        if element:
                            print(f"    🎯 已高亮目标元素")
                    elif action_lower.startswith('search['):
                        # 高亮搜索框
                        env.highlight_search_bar()
                        print(f"    🔍 已高亮搜索框")
                except Exception:
                    pass
            
            # 交互模式：等待用户按 Enter 继续
            if use_browser and interactive:
                input("    按 Enter 执行此动作...")
            
            # 执行动作
            if can_execute:
                try:
                    obs, reward, done, info = env.step(action)
                    success_count += 1
                    print(f"    ✅ 执行成功, reward={reward}, done={done}")
                    
                    # 比较状态
                    if compare_states and step_idx + 1 < len(states):
                        recorded_state = states[step_idx + 1]
                        sim = similarity(obs, recorded_state)
                        print(f"    [比较] 状态相似度: {sim:.2%}")
                    
                    if done:
                        print(f"\n    🏁 任务完成! 最终 reward: {reward}")
                        break
                except Exception as e:
                    fail_count += 1
                    print(f"    ❌ 执行异常: {e}")
            else:
                fail_count += 1
                # 尝试强制执行看看会发生什么
                try:
                    obs, reward, done, info = env.step(action)
                    print(f"    ⚠️ 动作不在可用列表但执行了, reward={reward}, done={done}")
                    
                    # 比较状态
                    if compare_states and step_idx + 1 < len(states):
                        recorded_state = states[step_idx + 1]
                        sim = similarity(obs, recorded_state)
                        print(f"    [比较] 状态相似度: {sim:.2%}")
                    
                    if done:
                        print(f"\n    🏁 任务完成! 最终 reward: {reward}")
                        break
                except Exception as e:
                    print(f"    ❌ 执行失败: {e}")
            
            # Browser 模式下可以加一点延迟，方便观察
            if use_browser and render and not interactive:
                time.sleep(0.5)
    
    finally:
        # 确保关闭环境
        if use_browser and env:
            try:
                env.close()
                print("\n    浏览器已关闭")
            except:
                pass
    
    # 7. 输出统计
    print("\n" + "=" * 80)
    print("验证结果:")
    print(f"  环境类型: {env_type}")
    print(f"  总步数: {len(actions)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  成功率: {success_count / len(actions) * 100:.1f}%")
    print("=" * 80)
    
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='WebShop 训练数据验证工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python verify_il_traj.py 0                      # Text 环境验证
  python verify_il_traj.py 0 --browser            # Browser 环境验证 (headless)
  python verify_il_traj.py 0 --browser --render   # Browser 环境验证 (显示浏览器+交互模式)
  python verify_il_traj.py 0 --browser -i         # Browser 环境验证 (headless+交互模式)
  python verify_il_traj.py 0 --goal-idx 5         # 指定 goal 索引
  python verify_il_traj.py 0 --compare-states     # 比较状态相似度

功能:
  --render    显示浏览器窗口，高亮要点击的元素（红色边框）
  -i          交互模式：每步等待用户按 Enter 继续
  
注意: 使用 Browser 环境前需要先启动服务器:
  python -m web_agent_site.app
        """
    )
    
    parser.add_argument('traj_idx', type=int, help='轨迹索引')
    parser.add_argument('--browser', action='store_true', 
                        help='使用浏览器环境 (需要先启动 Flask 服务器)')
    parser.add_argument('--render', action='store_true',
                        help='显示浏览器窗口 (仅浏览器模式有效)')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL,
                        help=f'Flask 服务器地址 (默认: {DEFAULT_SERVER_URL})')
    parser.add_argument('--traj-path', type=str, default=None,
                        help=f'轨迹文件路径 (默认: {IL_TRAJ_PATH})')
    parser.add_argument('--goal-idx', type=int, default=None,
                        help='手动指定 goal 索引 (覆盖自动匹配)')
    parser.add_argument('--compare-states', action='store_true',
                        help='比较记录的 states 与环境返回的 observation')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互模式：每步等待用户按 Enter 继续 (仅浏览器模式)')
    
    args = parser.parse_args()
    
    # 更新轨迹文件路径（如果指定了的话）
    traj_path = args.traj_path if args.traj_path else IL_TRAJ_PATH
    
    # 切换到脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 验证轨迹
    # 如果使用 --render 且没有明确禁用，默认启用交互模式
    interactive = args.interactive or (args.render and args.browser)
    
    success = verify_trajectory(
        traj_idx=args.traj_idx,
        use_browser=args.browser,
        render=args.render,
        server_url=args.server_url,
        goal_idx=args.goal_idx,
        compare_states=args.compare_states,
        traj_path=traj_path,
        interactive=interactive
    )
    
    if success:
        print("\n✅ 验证通过!")
    else:
        print("\n❌ 验证失败!")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
