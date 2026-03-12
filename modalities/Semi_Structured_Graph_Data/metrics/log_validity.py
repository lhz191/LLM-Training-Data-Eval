"""
Log Validity — PA / GA / FGA / PTA / RTA / FTA  (Survey §5.2.3)

使用 loghub-2.0 (Jiang et al., 2024, ISSTA'24) 的评估代码:
  - PA  (Parsing Accuracy):  逐条日志的模板+变量精确匹配率
  - GA  (Grouping Accuracy): 日志分组正确率
  - FGA (F1 of GA):          GA 的 F1 版本
  - PTA (Template Precision): 模板级精确率
  - RTA (Template Recall):    模板级召回率
  - FTA (Template F1):        模板级 F1

输入格式: pandas DataFrame，需包含 'EventTemplate' 列。
模板中变量用 <*> 表示，如 "Received block <*> of size <*>"。

依赖: pandas, scipy, tqdm, regex
"""

import sys
import os
import pandas as pd
from typing import Dict, Any, List, Optional

_LOGHUB_ROOT = os.path.join(os.path.dirname(__file__), 'loghub-2.0')
_LOGHUB_BENCHMARK = os.path.join(_LOGHUB_ROOT, 'benchmark')

if _LOGHUB_BENCHMARK not in sys.path:
    sys.path.insert(0, _LOGHUB_BENCHMARK)

from logparser.utils.evaluator import evaluate as _evaluate_ga
from evaluation.utils.PA_calculator import calculate_parsing_accuracy as _calculate_pa
from evaluation.utils.template_level_analysis import evaluate_template_level as _evaluate_template_level


def compute_log_validity(
    groundtruth_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    dataset_name: str = "log",
    filter_templates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    计算日志解析的全套评估指标 (Survey §5.2.3)。

    参数:
        groundtruth_df:  ground truth DataFrame，必须含 'EventTemplate' 列
                         (模板字符串，变量用 <*> 占位)
        parsed_df:       生成/解析结果 DataFrame，必须含 'EventTemplate' 列
                         两个 DataFrame 行数须相同，第 i 行对应同一条日志
        dataset_name:    数据集名（仅用于打印）
        filter_templates: 可选，只评估这些模板对应的日志子集

    返回:
        dict 包含:
          - parsing_accuracy   (PA):  逐条消息模板完全匹配率
          - grouping_accuracy  (GA):  消息分组正确率（双向一致）
          - f1_grouping_accuracy (FGA): GA 的 F1 版本
          - template_precision (PTA): |T_correct| / |T_predicted|
          - template_recall    (RTA): |T_correct| / |T_groundtruth|
          - f1_template_accuracy (FTA): 2·PTA·RTA / (PTA+RTA)
    """
    groundtruth_df = groundtruth_df.copy().reset_index(drop=True)
    parsed_df = parsed_df.copy().reset_index(drop=True)
    groundtruth_df.fillna("", inplace=True)
    parsed_df.fillna("", inplace=True)

    # PA_calculator 内部用 parsedresult_df[['Content']] 计行数，
    # 若 DataFrame 缺 Content 列会 KeyError，这里补一个占位列
    if 'Content' not in parsed_df.columns:
        parsed_df['Content'] = ""
    if 'Content' not in groundtruth_df.columns:
        groundtruth_df['Content'] = ""

    # GA + FGA
    GA, FGA = _evaluate_ga(groundtruth_df, parsed_df, filter_templates)

    # PA
    PA = _calculate_pa(groundtruth_df, parsed_df, filter_templates)

    # PTA / RTA / FTA
    n_identified, n_groundtruth, FTA, PTA, RTA = _evaluate_template_level(
        dataset_name, groundtruth_df, parsed_df, filter_templates)

    return {
        'parsing_accuracy': float(PA),
        'grouping_accuracy': float(GA),
        'f1_grouping_accuracy': float(FGA),
        'template_precision': float(PTA),
        'template_recall': float(RTA),
        'f1_template_accuracy': float(FTA),
        'num_identified_templates': int(n_identified),
        'num_groundtruth_templates': int(n_groundtruth),
        'num_messages': len(groundtruth_df),
    }
