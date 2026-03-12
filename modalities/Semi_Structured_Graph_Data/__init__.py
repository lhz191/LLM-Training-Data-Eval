"""
Semi-Structured Data Evaluation (Graph + JSON + Log)

Survey §5.2 Content-level + §5.3 Trustworthy metrics.
"""

from .metrics.graph_validity import compute_graph_validity, text_to_graph
from .metrics.graph_fidelity import compute_graph_mmd
from .metrics.graph_diversity import compute_graph_diversity
from .metrics.graph_robustness import compute_scr, compute_gad, compute_dl2, compute_dl2_batch

from .metrics.json_validity import compute_json_validity
from .metrics.json_fidelity import compute_json_fidelity

from .metrics.log_validity import compute_log_validity
from .metrics.log_fidelity import (
    compute_variable_prf,
    compute_bleu,
    compute_rouge,
    compute_level_accuracy,
    compute_aod,
    compute_log_fidelity,
)
from .metrics.log_privacy import compute_qualitative_score
