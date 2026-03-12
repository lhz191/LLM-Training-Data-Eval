"""
Log Fidelity — VP/VR/VF1, BLEU, ROUGE, L-ACC, AOD  (Survey §5.2.3)

指标来源: Li et al., 2024 (LogBench, IEEE TSE)

1. Variable Precision / Recall / F1  (VP / VR / VF1)
   比较生成日志中绑定的运行时变量集与真实变量集。
   VP = |Vp ∩ Vt| / |Vp|,  VR = |Vp ∩ Vt| / |Vt|,
   VF1 = 2·VP·VR / (VP+VR)

2. BLEU / ROUGE
   模板或静态文本的 n-gram 重叠度，衡量语言真实性。

3. Log Level Accuracy  (L-ACC)
   日志级别完全匹配率: L-ACC = N_correct_level / N

4. Average Ordinal Distance  (AOD)
   日志级别在序数尺度上的归一化偏差:
   AOD = (1/N) Σ (1 − Dis(l_pred, l_gt) / MaxDis)

依赖: nltk, rouge-score (pip install nltk rouge-score)
"""

from typing import Dict, Any, List, Set, Tuple
import numpy as np


# ─── 日志级别序数映射 (低→高) ───────────────────────────────
DEFAULT_LEVEL_ORDER: Dict[str, int] = {
    'TRACE': 0, 'DEBUG': 1, 'INFO': 2, 'WARN': 3,
    'WARNING': 3, 'ERROR': 4, 'FATAL': 5, 'CRITICAL': 5,
}


# ═══════════════════════════════════════════════════════════
#  1. Variable Precision / Recall / F1
# ═══════════════════════════════════════════════════════════

def _var_prf(pred_vars: Set[str], true_vars: Set[str]) -> Tuple[float, float, float]:
    if not pred_vars and not true_vars:
        return 1.0, 1.0, 1.0
    inter = len(pred_vars & true_vars)
    vp = inter / len(pred_vars) if pred_vars else 0.0
    vr = inter / len(true_vars) if true_vars else 0.0
    vf1 = 2 * vp * vr / (vp + vr) if (vp + vr) else 0.0
    return vp, vr, vf1


def compute_variable_prf(
    pred_var_list: List[Set[str]],
    true_var_list: List[Set[str]],
) -> Dict[str, float]:
    """
    逐条计算 VP/VR/VF1，返回宏平均。

    参数:
        pred_var_list: 每条日志的预测变量集列表
        true_var_list: 每条日志的真实变量集列表
    """
    assert len(pred_var_list) == len(true_var_list), "列表长度不一致"
    vps, vrs, vf1s = [], [], []
    for pv, tv in zip(pred_var_list, true_var_list):
        vp, vr, vf1 = _var_prf(pv, tv)
        vps.append(vp)
        vrs.append(vr)
        vf1s.append(vf1)
    return {
        'variable_precision': float(np.mean(vps)),
        'variable_recall': float(np.mean(vrs)),
        'variable_f1': float(np.mean(vf1s)),
        'num_samples': len(pred_var_list),
    }


# ═══════════════════════════════════════════════════════════
#  2. BLEU / ROUGE
# ═══════════════════════════════════════════════════════════

def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 sentence-level BLEU-4 的宏平均 + corpus-level BLEU。

    参数:
        predictions: 生成文本列表
        references:  参考文本列表 (一对一)
    """
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

    assert len(predictions) == len(references)
    smooth = SmoothingFunction().method1

    sent_scores = []
    refs_tok, preds_tok = [], []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        refs_tok.append([ref_tokens])
        preds_tok.append(pred_tokens)
        if not pred_tokens:
            sent_scores.append(0.0)
        else:
            sent_scores.append(sentence_bleu(
                [ref_tokens], pred_tokens, smoothing_function=smooth))

    corp = corpus_bleu(refs_tok, preds_tok, smoothing_function=smooth)

    return {
        'bleu_sentence_avg': float(np.mean(sent_scores)),
        'bleu_corpus': float(corp),
        'num_samples': len(predictions),
    }


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 ROUGE-1/2/L F1 的平均值。

    参数:
        predictions: 生成文本列表
        references:  参考文本列表 (一对一)
    """
    from rouge_score import rouge_scorer

    assert len(predictions) == len(references)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    r1s, r2s, rls = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1s.append(scores['rouge1'].fmeasure)
        r2s.append(scores['rouge2'].fmeasure)
        rls.append(scores['rougeL'].fmeasure)

    return {
        'rouge1_f1': float(np.mean(r1s)),
        'rouge2_f1': float(np.mean(r2s)),
        'rougeL_f1': float(np.mean(rls)),
        'num_samples': len(predictions),
    }


# ═══════════════════════════════════════════════════════════
#  3. Log Level Accuracy  (L-ACC)
# ═══════════════════════════════════════════════════════════

def compute_level_accuracy(
    pred_levels: List[str],
    true_levels: List[str],
) -> Dict[str, float]:
    """
    L-ACC = N_correct_level / N

    参数:
        pred_levels: 预测日志级别列表 (字符串，如 "INFO", "ERROR")
        true_levels: 真实日志级别列表
    """
    assert len(pred_levels) == len(true_levels)
    n = len(pred_levels)
    correct = sum(
        p.strip().upper() == t.strip().upper()
        for p, t in zip(pred_levels, true_levels)
    )
    return {
        'level_accuracy': correct / n if n else 0.0,
        'num_correct': correct,
        'num_samples': n,
    }


# ═══════════════════════════════════════════════════════════
#  4. Average Ordinal Distance  (AOD)
# ═══════════════════════════════════════════════════════════

def compute_aod(
    pred_levels: List[str],
    true_levels: List[str],
    level_order: Dict[str, int] = None,
) -> Dict[str, float]:
    """
    AOD = (1/N) Σ (1 − Dis(l_pred, l_gt) / MaxDis)

    参数:
        pred_levels: 预测日志级别列表
        true_levels: 真实日志级别列表
        level_order: 级别→序数映射，默认
                     TRACE=0 < DEBUG=1 < INFO=2 < WARN=3 < ERROR=4 < FATAL=5
    """
    if level_order is None:
        level_order = DEFAULT_LEVEL_ORDER

    assert len(pred_levels) == len(true_levels)
    max_dis = max(level_order.values()) - min(level_order.values())
    if max_dis == 0:
        return {'aod': 1.0, 'num_samples': len(pred_levels)}

    scores = []
    for p, t in zip(pred_levels, true_levels):
        p_ord = level_order.get(p.strip().upper())
        t_ord = level_order.get(t.strip().upper())
        if p_ord is None or t_ord is None:
            scores.append(0.0)
            continue
        scores.append(1.0 - abs(p_ord - t_ord) / max_dis)

    return {
        'aod': float(np.mean(scores)) if scores else 0.0,
        'num_samples': len(pred_levels),
    }


# ═══════════════════════════════════════════════════════════
#  聚合入口
# ═══════════════════════════════════════════════════════════

def compute_log_fidelity(
    pred_var_list: List[Set[str]] = None,
    true_var_list: List[Set[str]] = None,
    pred_texts: List[str] = None,
    ref_texts: List[str] = None,
    pred_levels: List[str] = None,
    true_levels: List[str] = None,
    level_order: Dict[str, int] = None,
) -> Dict[str, Any]:
    """
    一站式计算所有 Log Fidelity 指标。
    只传入非 None 的参数即可，缺失的指标会跳过。

    参数:
        pred_var_list / true_var_list: 变量集列表 → VP/VR/VF1
        pred_texts / ref_texts:        文本列表   → BLEU + ROUGE
        pred_levels / true_levels:     级别列表   → L-ACC + AOD
        level_order:                   自定义级别序数映射
    """
    results: Dict[str, Any] = {}

    if pred_var_list is not None and true_var_list is not None:
        results.update(compute_variable_prf(pred_var_list, true_var_list))

    if pred_texts is not None and ref_texts is not None:
        results.update(compute_bleu(pred_texts, ref_texts))
        results.update(compute_rouge(pred_texts, ref_texts))

    if pred_levels is not None and true_levels is not None:
        results.update(compute_level_accuracy(pred_levels, true_levels))
        results.update(compute_aod(pred_levels, true_levels, level_order))

    return results
