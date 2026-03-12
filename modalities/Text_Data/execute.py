import torch
import torch.nn as nn
import argparse
from configs.basic_cfg import get_cfg

# Plain Text metrics
from metrics.GrammaticalityRate import ComputeGrammaticalityRate


import os
import sys
import json
"""
srun -p TDS --job-name=Sample --ntasks-per-node=1 --gres=gpu:1 \
    python execute.py -f configs/test.yaml
"""

sys.path.append(os.path.join(os.path.dirname(__file__), "metrics"))


def load_jsonl(file_path):
    """
    加载 jsonl 文件，每一行是一个 JSON 对象

    Args:
        file_path (str): jsonl 文件路径

    Returns:
        list: 由每一行 JSON 组成的列表（list[dict]）
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: 第 {line_num} 行\n{line}") from e
    return data






def main(args):
    config = get_cfg(args.config_file)

    data = load_jsonl(config.jsonl_path)
    
    outputs = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.mode == 'plain text':
        if config.metrics.grammatical_acceptablity:
            print("Measuring Grammatical Acc...")
            outputs['GrammaticalityRate'] = ComputeGrammaticalityRate(data, device)
        
        
    elif config.mode == 'dialogue':
        if config.metrics.
    
    output_path = os.path.join(config.output_dir, "res.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--config-file',type=str,default="")
    args = parser.parse_args()
    main(args)
    
    
    
    
