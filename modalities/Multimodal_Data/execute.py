import torch
import torch.nn as nn
import argparse
from data_utils.ucf101 import UCF101
from data_utils.internvid import InternVid
import numpy as np
from metrics.frame_diver import get_flow_farn
# from metrics.holistic_fidelity import get_holistic_fidelity
from model.internvid.temporal_accuracy import get_temporal_accuarcy
from data_utils import video_transforms

import torchvision.transforms as transforms
from configs.basic_cfg import get_cfg
from metrics.semantic_diver import get_semantic_diversity
from metrics.object_consistency import get_object_consistency
from metrics.frame_diver import get_frame_diversity
from metrics.T2VSafetyBench import T2VSafetyBench
from metrics.holistic_fidelity import run_VBench
from metrics.video_cmc import get_cmc

#image
from metrics.well_formed_rate import get_well_formed_rate
from metrics.win_rate import get_win_rate
from metrics.validate_cpa import get_validate_cpa
from metrics.safety_asr_rr import get_safety_asr_rr
from metrics.inception_score import get_inception_score
from metrics.image_prompt_fidelity import get_prompt_fidelity
from metrics.subject_fidelity import get_subject_fidelity

#audio
from metrics.clap_score import compute_clap_score
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
    
    if config.mode == 'video':
        # video metrics

        if config.metrics.frame_diversity:
            print("Measuring Frame Diversity...")
            outputs['frame_diversity'] = float(get_frame_diversity(data))
        
        if config.metrics.object_consistency:
            print("Measuring Object Consistency...")
            outputs['object_consistency'] = float(get_object_consistency(data, device))
        
        if config.metrics.semantic_diversity:
            print("Measuring Semantic Diversity...")
            outputs['semantic_diversity'] = float(get_semantic_diversity(data, device))
        
        if config.metrics.safety_bench:
            print("Measuring Safety using T2VBench...")
            outputs["Safety"] = T2VSafetyBench(config, data)
            
        if config.metrics.cross_modal_consistency:
            print("Measuring Cross Modal Consistency...")
            outputs["cmc"] = get_cmc(data, device)
            
        if config.metrics.vbench:
            print("Measuring via VBench...")
            outputs["VBench"] = run_VBench(config, data)
    
    elif config.mode == 'image':
        #image metrics
        if config.metrics.well_formed_rate:
            print("Measuring Well-Formed Rate...")
            outputs["well_formed_rate"] = get_well_formed_rate(data,config)
        
        if config.metrics.win_rate:
                print("Measuring Win Rate...")
                outputs["win_rate"] = get_win_rate(data)

        if config.metrics.validate_cpa:
            print("Measuring Validate CPA...")
            outputs["validate_cpa"] = get_validate_cpa(data, config)
        
        if config.metrics.safety_asr_rr:
            print("Measuring Safety ASR/RR...")
            outputs["safety_asr_rr"] = get_safety_asr_rr(data, config.safety_asr_rr)

        if config.metrics.inception_score:
            print("Measuring Inception Score...")
            outputs["inception_score"] = get_inception_score(data, config.inception_score)
        
        if config.metrics.prompt_fidelity:
            print("Measuring Prompt Fidelity...")
            outputs["prompt_fidelity"] = get_prompt_fidelity(data, device)

        if getattr(config.metrics, 'subject_fidelity', False):
            print("Measuring Subject Fidelity (CLIP-I)...")
            outputs["subject_fidelity"] = get_subject_fidelity(data, device)

    else:
        if config.metrics.clap_score:
            print("Measuring Clap Score...")
            outputs['clap_score'] = compute_clap_score(data, device)
        if config.metrics.fad:
            print("Measuring FAD...")
    output_path = os.path.join(config.output_dir, "res.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--config-file',type=str,default="")
    args = parser.parse_args()
    main(args)
    
    
    
    
