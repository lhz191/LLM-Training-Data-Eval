#!/usr/bin/env python3
"""
WebShop 原始轨迹数据查看工具
查看 all_trajs/*.jsonl 的原始内容

用法: python view_trajectory.py <轨迹文件名或索引>
示例: 
  python view_trajectory.py 20220427_1_12.jsonl
  python view_trajectory.py 0
"""

import json
import sys
import os
import glob

# 数据目录
DATA_DIR = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/webshop/all_trajs'
OUTPUT_PATH = './output_traj.txt'


def format_dict(d, indent=2):
    """格式化字典显示"""
    prefix = " " * indent
    lines = ["{"]
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}  {k}: {format_dict(v, indent + 2)}")
        elif isinstance(v, list):
            if len(v) == 0:
                lines.append(f"{prefix}  {k}: []")
            elif len(v) <= 5:
                lines.append(f"{prefix}  {k}: {v}")
            else:
                lines.append(f"{prefix}  {k}: [")
                for i, item in enumerate(v):
                    lines.append(f"{prefix}    [{i}] {item}")
                lines.append(f"{prefix}  ]")
        else:
            lines.append(f"{prefix}  {k}: {repr(v)}")
    lines.append(f"{prefix[:-2]}}}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python view_trajectory.py <轨迹文件名或索引>")
        print("示例: ")
        print("  python view_trajectory.py 20220427_1_12.jsonl")
        print("  python view_trajectory.py 0")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # 获取所有轨迹文件
    traj_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.jsonl')))
    print(f"共找到 {len(traj_files)} 个轨迹文件")
    
    # 确定要读取的文件
    if arg.endswith('.jsonl'):
        file_path = os.path.join(DATA_DIR, arg)
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    else:
        try:
            idx = int(arg)
            if idx < 0 or idx >= len(traj_files):
                print(f"错误: 索引 {idx} 超出范围 (0-{len(traj_files)-1})")
                sys.exit(1)
            file_path = traj_files[idx]
        except ValueError:
            print(f"错误: 无效参数: {arg}")
            sys.exit(1)
    
    file_name = os.path.basename(file_path)
    print(f"读取文件: {file_path}")
    
    # 读取轨迹
    steps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    
    print(f"步数: {len(steps)}")
    
    output = []
    
    # 标题
    output.append("=" * 100)
    output.append(f"轨迹文件: {file_name}")
    output.append(f"步骤数: {len(steps)}")
    output.append("=" * 100)
    
    # 显示每一步
    for step_idx, step in enumerate(steps):
        output.append("")
        output.append("─" * 100)
        output.append(f"【Step {step_idx}】")
        output.append("─" * 100)
        
        # 按顺序显示所有字段
        for key in step.keys():
            output.append("")
            output.append(f"{key}:")
            
            value = step[key]
            
            if isinstance(value, dict):
                # 字典格式化
                output.append(f"  {{")
                for k, v in value.items():
                    if isinstance(v, list):
                        if len(v) == 0:
                            output.append(f"    {k}: []")
                        else:
                            output.append(f"    {k}: [")
                            for i, item in enumerate(v):
                                output.append(f"      [{i}] {repr(item)}")
                            output.append(f"    ]")
                    else:
                        output.append(f"    {k}: {repr(v)}")
                output.append(f"  }}")
            elif isinstance(value, list):
                if len(value) == 0:
                    output.append(f"  []")
                else:
                    output.append(f"  [")
                    for i, item in enumerate(value):
                        output.append(f"    [{i}] {repr(item)}")
                    output.append(f"  ]")
    else:
                output.append(f"  {repr(value)}")
    
    output.append("")
    output.append("=" * 100)
    
    # 写入文件
    result = '\n'.join(output)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"已保存到: {output_path}")


if __name__ == '__main__':
    main()
