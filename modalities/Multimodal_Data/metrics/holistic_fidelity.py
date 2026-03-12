# from vbench import VBench2


import torch
import os
from .vbench import VBench
from .vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

# def get_holistic_fidelity(dataset, device):
#     my_VBench = VBench2(device, "vbench2/VBench2_full_info.json", "evaluation_results")
#     my_VBench.evaluate(
#         videos_path = "sampled_videos/HunyuanVideo/Human_Interaction",
#         name = "HunyuanVideo_Human_Interaction",
#         dimension_list = ["Human_Interaction"],
#     )
    
    
    
"""
srun -p TDS \
vbench evaluate \
    --dimension "human_action" \
    --videos_path /mnt/petrelfs/wangjiedong/LLMDataBenchmark/Datasets/Multimodal/UCF-101/ApplyEyeMakeup \
    --mode=custom_input
    
"""    



def run_VBench(config, data, device):
    args = config.VBench
    
    
    



"""
srun -p TDS \
python evaluate.py \
    --dimension "human_action" \
    --videos_path /mnt/petrelfs/wangjiedong/LLMDataBenchmark/Datasets/Multimodal/test_videos \
    --mode=custom_input
"""





def run_VBench(config, data):
    args = config.VBench
    

    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")
    my_VBench = VBench(device, args.full_json_dir, args.output_path)
    
    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    prompt = []

    if (args.prompt_file is not None) and (args.prompt != "None"):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "None") and (args.mode!='custom_input'):
        raise Exception("must set --mode=custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
        assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"
    elif args.prompt != "None":
        prompt = [args.prompt]

    if args.category != "":
        kwargs['category'] = args.category

    kwargs['imaging_quality_preprocessing_mode'] = args.imaging_quality_preprocessing_mode

    res_dict = my_VBench.evaluate(
        video_data=data,
        name = f'results_{current_time}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **kwargs
    )
    
    print0('done')

    return res_dict
