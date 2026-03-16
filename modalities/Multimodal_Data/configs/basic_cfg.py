import os

from yacs.config import CfgNode as CN


_C = CN()

#general setting
_C.jsonl_path = ''
_C.output_dir = ''

_C.mode = "Video" # ["video", "image"]

_C.metrics = CN()
_C.metrics.frame_diversity = True
_C.metrics.object_consistency = True
_C.metrics.semantic_diversity = True
_C.metrics.safety_bench = True
_C.metrics.cross_modal_consistency = True
_C.metrics.vbench = True
_C.metrics.style = True

_C.metrics.well_formed_rate = True
_C.metrics.win_rate = False
_C.metrics.safety_asr_rr = False
_C.metrics.validate_cpa = False
_C.metrics.inception_score = False
_C.metrics.prompt_fidelity = False
_C.metrics.subject_fidelity = False

_C.metrics.clap_score = False
_C.metrics.audio_kl = False
_C.metrics.audio_consistency = False

# fad (audio)
_C.metrics.fad = False

# text metrics (reused from Text_Data/metrics for report mode)
_C.metrics.self_cos_sim = False
_C.metrics.grammaticality_rate = False
_C.metrics.text_diversity = False

# ======================================================
# SafetyBench
# ======================================================
_C.SafetyBench = CN()

# ------------------------
# General
# ------------------------
_C.SafetyBench.seed = 1
_C.SafetyBench.classes = 1
_C.SafetyBench.video_model = "opensora"
_C.SafetyBench.gpt_api = "sk-DCmEmJYyx4Mqqln9etb7frZZwcRhy9uf2AhnrdXe7B3AnJuf"

_C.SafetyBench.n_frames = 5
_C.SafetyBench.scale_percent = 20.0
_C.SafetyBench.img_length = 2000

_C.SafetyBench.gpt_gen_prompts = ""
_C.SafetyBench.gpt_eval_prompts = ""
_C.SafetyBench.def_prompt = ""

_C.SafetyBench.save_dir = None
_C.SafetyBench.prompt_path = None

_C.SafetyBench.mode = "video"
_C.SafetyBench.start = 0

# ------------------------
# gpt4
# ------------------------
_C.SafetyBench.max_tokens = 150
_C.SafetyBench.num_text = 1
_C.SafetyBench.max_query = 20
_C.SafetyBench.temperature = 0.7

_C.SafetyBench.eval_each = False
_C.SafetyBench.jsonl_path = None

# ======================================================
# VBench Evaluation
# ======================================================
_C.VBench = CN()

# paths
_C.VBench.output_path = "./evaluation_results/"
_C.VBench.full_json_dir = ""          # default set at runtime

# evaluation setup
_C.VBench.dimension = []              # REQUIRED, list of dimensions
# ['subject_consistency', 'background_consistency', 'temporal_flickering', \
    # 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 、
    # 'object_class', 'multiple_objects', 'human_action', 'color', \
        # 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency']
_C.VBench.load_ckpt_from_local = False
_C.VBench.read_frame = False

# mode control
_C.VBench.mode = "vbench_standard"    # ['custom_input', 'vbench_standard', 'vbench_category']

# prompt related
_C.VBench.prompt = "None"
_C.VBench.prompt_file = None
_C.VBench.category = ""

# ======================================================
# Dimension-specific parameters
# ======================================================
_C.VBench.imaging_quality_preprocessing_mode = "longer"
# options:
# ['shorter', 'longer', 'shorter_centercrop', 'None']


# ======================================================
# Well-Formed Rate (WFR)
# ======================================================
_C.well_formed_rate = CN()
_C.well_formed_rate.enabled = False
_C.well_formed_rate.strict = True                # 是否启用严格模式


# ======================================================
# Inception Score
# ======================================================
_C.inception_score = CN()
_C.inception_score.enabled = False
_C.inception_score.batch_size = 64
_C.inception_score.splits = 10
_C.inception_score.device = None
_C.inception_score.num_workers = 4
_C.inception_score.seed = 0
_C.inception_score.torch_home = None
_C.inception_score.allow_download = True


# ======================================================
# Validate C2PA
# ======================================================
_C.validate_cpa = CN()
_C.validate_cpa.enabled = False
_C.validate_cpa.c2pa_tool = os.path.join("tools", "c2patool", "c2patool")
_C.validate_cpa.output_dir = "evaluation_results"
_C.validate_cpa.save_details = True


# ======================================================
# Safety ASR/RR
# ======================================================
_C.safety_asr_rr = CN()
_C.safety_asr_rr.enabled = False
_C.safety_asr_rr.mode = "field"                  # choices: "field", "global_gpt", "scenario_gpt"
_C.safety_asr_rr.api_key = "sk-iOTmFMqfW87Fs9PguEb9ewNQbaffa8kbErewVnQVFAuESIQZ"
_C.safety_asr_rr.base_url = "http://35.220.164.252:3888/v1/"
_C.safety_asr_rr.judge_model = "gpt-4"
_C.safety_asr_rr.output_dir = "evaluation_results"
_C.safety_asr_rr.sleep = 0.3
_C.safety_asr_rr.writeback = False
_C.safety_asr_rr.safety_field = "is_safe(gpt)"
_C.safety_asr_rr.refusal_field = "is_refusal"
_C.safety_asr_rr.unsafe_value = "unsafe"
_C.safety_asr_rr.refusal_value = "refusal"
_C.safety_asr_rr.refusal_strategy = "gpt"        # choices: "gpt", "heuristic"


# ======================================================
# Report (Image-to-Report) Evaluation
# ======================================================
_C.report = CN()
_C.report.dataset = ''          # 'iu_xray' or 'sharegpt4v'
_C.report.data_path = ''
_C.report.image_base_dir = ''
_C.report.split = 'train'
_C.report.max_samples = 0       # 0 = 全量
_C.report.parallel = False
_C.report.workers = 8
_C.report.skip_llm = False
_C.report.embedding_model = 'all-MiniLM-L6-v2'
_C.report.diversity_method = 'knn'
_C.report.near_dup_threshold = 0.95


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
    