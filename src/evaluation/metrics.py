"""
Evaluation metrics for the three research objectives.

Task 1: Competency state prediction — AUC-ROC, Accuracy
Task 2: Competency cluster detection — Silhouette score, NMI
Task 3: Engagement-based pass/fail   — AUC-ROC

Statistical significance: paired t-tests with Bonferroni correction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    silhouette_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)


# ======================================================================
# Prediction metrics
# ======================================================================

def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute prediction metrics for competency state or pass/fail.

    Parameters
    ----------
    y_true : array of shape (N,)
        Ground truth binary labels.
    y_pred : array of shape (N,)
        Predicted labels.
    y_prob : array of shape (N,) or (N, 2), optional
        Predicted probabilities for AUC computation.

    Returns
    -------
    dict with keys: "accuracy", "auc_roc"
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred) * 100

    if y_prob is not None:
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0
            logger.warning("AUC computation failed (single class in y_true?)")
    else:
        metrics["auc_roc"] = 0.0

    return metrics


# ======================================================================
# Clustering metrics
# ======================================================================

def compute_cluster_metrics(
    embeddings: np.ndarray,
    ground_truth_labels: np.ndarray,
    num_clusters: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate competency cluster detection quality.

    Applies spectral clustering to learned embeddings and compares
    against ground-truth concept groupings.

    Parameters
    ----------
    embeddings : array of shape (N, D)
        Learned node embeddings from the model.
    ground_truth_labels : array of shape (N,)
        Ground-truth cluster labels (from course taxonomies).
    num_clusters : int
        Number of clusters (k=10 for MOOCCubeX, k=5 for OULAD).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: "silhouette", "nmi"
    """
    # Spectral clustering on embeddings
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="nearest_neighbors",
        n_neighbors=min(20, len(embeddings) - 1),
        random_state=random_state,
        assign_labels="kmeans",
    )
    predicted_labels = clustering.fit_predict(embeddings)

    metrics = {}

    # Silhouette score
    try:
        metrics["silhouette"] = silhouette_score(embeddings, predicted_labels)
    except ValueError:
        metrics["silhouette"] = 0.0

    # Normalised mutual information
    metrics["nmi"] = normalized_mutual_info_score(
        ground_truth_labels, predicted_labels, average_method="arithmetic"
    )

    return metrics


# ======================================================================
# Engagement prediction metrics (Objective 3)
# ======================================================================

def evaluate_engagement_features(
    structural_features: np.ndarray,
    traditional_features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare structural vs traditional engagement features for pass/fail.

    Uses XGBoost gradient-boosted classifier (per thesis Section V-C).

    Parameters
    ----------
    structural_features : array (N, D1)
        Motif-graphlet structural features per student.
    traditional_features : array (N, D2)
        Traditional metrics (clicks, login freq, time-on-task, sessions).
    labels : array (N,)
        Binary pass/fail labels.
    train_mask, test_mask : boolean arrays (N,)

    Returns
    -------
    dict with keys: "structural", "traditional", "combined"
        Each containing "auc_roc" and "accuracy".
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        return {}

    results = {}
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    feature_sets = {
        "structural": structural_features,
        "traditional": traditional_features,
        "combined": np.concatenate(
            [structural_features, traditional_features], axis=1
        ),
    }

    for name, features in feature_sets.items():
        X_train = features[train_mask]
        X_test = features[test_mask]

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        results[name] = {
            "auc_roc": roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred) * 100,
        }
        logger.info(
            "  %s: AUC=%.3f, Acc=%.1f%%",
            name, results[name]["auc_roc"], results[name]["accuracy"],
        )

    return results


# ======================================================================
# Statistical significance
# ======================================================================

def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
    bonferroni_m: int = 1,
) -> Dict[str, float]:
    """Paired t-test across multiple runs.

    Parameters
    ----------
    scores_a, scores_b : lists of float
        Metric values from each independent run.
    alpha : float
        Significance level (default 0.05).
    bonferroni_m : int
        Number of comparisons for Bonferroni correction.

    Returns
    -------
    dict with "t_statistic", "p_value", "significant", "corrected_alpha"
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    corrected_alpha = alpha / bonferroni_m

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < corrected_alpha,
        "corrected_alpha": corrected_alpha,
    }


def run_significance_tests(
    model_scores: Dict[str, List[float]],
    target_model: str = "mgi_gnn",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Run paired t-tests between target model and all baselines.

    Parameters
    ----------
    model_scores : dict
        {model_name: [score_run1, score_run2, ...]}
    target_model : str
        Name of the proposed model.
    alpha : float
        Significance level before Bonferroni correction.

    Returns
    -------
    dict of {baseline_name: test_results}
    """
    if target_model not in model_scores:
        raise ValueError(f"Target model '{target_model}' not in scores")

    baselines = [m for m in model_scores if m != target_model]
    num_comparisons = len(baselines)
    results = {}

    for baseline in baselines:
        results[baseline] = paired_t_test(
            model_scores[target_model],
            model_scores[baseline],
            alpha=alpha,
            bonferroni_m=num_comparisons,
        )
        sig = "YES" if results[baseline]["significant"] else "no"
        logger.info(
            "  %s vs %s: t=%.3f, p=%.4f, significant=%s",
            target_model, baseline,
            results[baseline]["t_statistic"],
            results[baseline]["p_value"],
            sig,
        )

    return results


# ======================================================================
# Multi-run aggregation
# ======================================================================

def aggregate_runs(
    run_results: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across multiple runs.

    Returns mean ± standard deviation for each metric.

    Parameters
    ----------
    run_results : list of dicts
        Each dict contains metric_name: value for one run.

    Returns
    -------
    dict of {metric_name: (mean, std)}
    """
    all_keys = set()
    for r in run_results:
        all_keys.update(r.keys())

    aggregated = {}
    for key in sorted(all_keys):
        values = [r.get(key, 0.0) for r in run_results]
        aggregated[key] = (np.mean(values), np.std(values))

    return aggregated


def format_results_table(
    model_results: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    """Format results as a markdown table (thesis Tables 3-5 style).

    Parameters
    ----------
    model_results : nested dict
        {model_name: {metric_name: (mean, std)}}
    """
    models = list(model_results.keys())
    if not models:
        return ""

    metrics = sorted(model_results[models[0]].keys())

    # Header
    header = "| Model | " + " | ".join(metrics) + " |"
    sep = "|---" * (len(metrics) + 1) + "|"
    rows = [header, sep]

    for model in models:
        vals = []
        for metric in metrics:
            mean, std = model_results[model].get(metric, (0.0, 0.0))
            if "accuracy" in metric.lower():
                vals.append(f"{mean:.1f} ± {std:.1f}")
            else:
                vals.append(f"{mean:.3f} ± {std:.3f}")
        row = f"| {model} | " + " | ".join(vals) + " |"
        rows.append(row)

    return "\n".join(rows)
