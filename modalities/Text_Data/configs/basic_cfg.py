"""
Basic config for Text Evaluation
"""



import os

from yacs.config import CfgNode as CN


_C = CN()

#general setting
_C.jsonl_path = ''
_C.output_dir = ''

_C.mode = "plain text" # ["plain text", "dialogue"]

_C.metrics = CN()

# Plain Text
_C.metrics.grammatical_acceptablity = True


# Dialogue
_C.metrics.ruber = True



def get_cfg(config_file_path):
    """
    Initialize configuration.
    """
    config = _C.clone()
    # merge specific config.
    config.merge_from_file(config_file_path)
    
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    # config.freeze()
    return config
    