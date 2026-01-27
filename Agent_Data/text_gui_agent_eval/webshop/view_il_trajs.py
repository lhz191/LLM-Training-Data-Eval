#!/usr/bin/env python3
"""
WebShop IL训练数据查看工具
查看 il_trajs_finalized_images.jsonl 的原始内容

用法: python view_il_trajs.py <轨迹索引>
"""

import ijson
import sys
import os

FILE_PATH = './baseline_models/data/il_trajs_finalized_images.jsonl'
OUTPUT_PATH = './output_il_traj.txt'


def main():
    if len(sys.argv) < 2:
        print("用法: python view_il_trajs.py <轨迹索引>")
        print("示例: python view_il_trajs.py 0")
        sys.exit(1)
    
    traj_index = int(sys.argv[1])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, FILE_PATH)
    
    print(f"读取文件: {file_path}")
    print(f"查看第 {traj_index} 条轨迹")
    
    # 使用ijson逐行读取，找到目标轨迹
    traj = None
    current_index = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if current_index == traj_index:
                # 使用ijson解析这一行
                traj = list(ijson.items(line.encode('utf-8'), ''))[0]
                break
            current_index += 1
    
    if traj is None:
        print(f"错误: 轨迹索引 {traj_index} 超出范围")
        sys.exit(1)
    
    num_steps = len(traj['actions'])
    
    output = []
    
    # 标题
    output.append("=" * 100)
    output.append(f"轨迹文件: il_trajs_finalized_images.jsonl")
    output.append(f"轨迹索引: {traj_index}")
    output.append(f"步数: {num_steps}")
    output.append("=" * 100)
    
    # 显示每一步
    for step in range(num_steps):
        output.append("")
        output.append("─" * 100)
        output.append(f"【Step {step}】")
        output.append("─" * 100)
        
        # actions
        output.append("")
        output.append("actions:")
        output.append(f'  "{traj["actions"][step]}"')
        
        # actions_translate
        output.append("")
        output.append("actions_translate:")
        output.append(f'  "{traj["actions_translate"][step]}"')
        
        # action_idxs
        output.append("")
        output.append("action_idxs:")
        output.append(f'  {traj["action_idxs"][step]}')
        
        # states
        output.append("")
        output.append("states:")
        state_text = traj["states"][step]
        for line in state_text.split('\n'):
            output.append(f'  {line}')
        
        # available_actions
        output.append("")
        output.append("available_actions:")
        avail = traj["available_actions"][step]
        if len(avail) == 0:
            output.append("  []")
        else:
            output.append("  [")
            for i, act in enumerate(avail):
                output.append(f'    [{i}] "{act}"')
            output.append("  ]")
        
        # images
        output.append("")
        output.append("images:")
        img = traj["images"][step]
        if img == 0:
            output.append("  0")
        elif isinstance(img, list):
            output.append(f"  [")
            for i in range(0, len(img), 10):
                chunk = img[i:i+10]
                output.append(f"    {chunk}")
            output.append(f"  ]")
            output.append(f"  (共 {len(img)} 维)")
        else:
            output.append(f"  {img}")
    
    output.append("")
    output.append("=" * 100)
    
    # 写入文件
    result = '\n'.join(output)
    output_path = os.path.join(script_dir, OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"已保存到: {output_path}")


if __name__ == '__main__':
    main()
