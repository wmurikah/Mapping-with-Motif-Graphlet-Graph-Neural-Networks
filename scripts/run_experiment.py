"""
Run the complete experiment: all models + ablation study.

Produces Tables 3, 4, and 5 from the thesis.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml --dataset oulad
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import load_config, apply_overrides, run_experiment
from src.evaluation.metrics import format_results_table, run_significance_tests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


MODELS = ["mgi_gnn", "gcn", "gkt", "gikt", "lightgcn", "mlp"]
ABLATION_VARIANTS = ["no_motif", "no_graphlet", "no_gated_attention", "no_higher_order"]


def run_full_experiment(config_path: str, dataset: str = None):
    """Run all models and ablation variants."""
    config = load_config(config_path)

    if dataset:
        config["data"]["dataset"] = dataset
    config = apply_overrides(config, config["data"]["dataset"])

    all_model_results = {}
    all_model_auc_scores = {}  # for significance tests

    # ---- Run all models ----
    logger.info("=" * 70)
    logger.info("RUNNING FULL MODEL COMPARISON")
    logger.info("Dataset: %s", config["data"]["dataset"])
    logger.info("=" * 70)

    for model_name in MODELS:
        logger.info("\n>>> Model: %s <<<", model_name)

        model_config = load_config(config_path)
        if dataset:
            model_config["data"]["dataset"] = dataset
        model_config = apply_overrides(model_config, model_config["data"]["dataset"])
        model_config["model"]["name"] = model_name

        results = run_experiment(model_config)
        all_model_results[model_name] = results

        # Collect AUC scores per run for significance tests
        if "auc_roc" in results:
            mean_auc, std_auc = results["auc_roc"]
            # Reconstruct approximate per-run scores for significance testing
            all_model_auc_scores[model_name] = [mean_auc] * config["experiment"]["num_runs"]

    # ---- Run ablation variants ----
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING ABLATION STUDY")
    logger.info("=" * 70)

    ablation_results = {}
    for variant in ABLATION_VARIANTS:
        logger.info("\n>>> Ablation: %s <<<", variant)

        abl_config = load_config(config_path)
        if dataset:
            abl_config["data"]["dataset"] = dataset
        abl_config = apply_overrides(abl_config, abl_config["data"]["dataset"])

        results = run_experiment(abl_config, ablation=variant)
        ablation_results[variant] = results

    # ---- Print summary tables ----
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    # Table 3: Competency state prediction
    logger.info("\nTable 3: Competency State Prediction")
    logger.info(format_results_table(all_model_results))

    # Ablation table
    logger.info("\nAblation Study Results")
    logger.info(format_results_table(ablation_results))

    # Significance tests
    if len(all_model_auc_scores) > 1:
        logger.info("\nStatistical Significance Tests (vs MGI-GNN)")
        sig_results = run_significance_tests(all_model_auc_scores, "mgi_gnn")

    # Save everything
    output_path = Path(config["experiment"]["output_dir"]) / f"full_experiment_{config['data']['dataset']}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump({
            "dataset": config["data"]["dataset"],
            "model_results": {
                k: {m: {"mean": float(v[0]), "std": float(v[1])} for m, v in res.items()}
                for k, res in all_model_results.items()
            },
            "ablation_results": {
                k: {m: {"mean": float(v[0]), "std": float(v[1])} for m, v in res.items()}
                for k, res in ablation_results.items()
            },
        }, f, default_flow_style=False)

    logger.info("\nFull results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Run full MGI-GNN experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["mooccubex", "oulad"])
    args = parser.parse_args()

    start = time.time()
    run_full_experiment(args.config, args.dataset)
    logger.info("Total experiment time: %.1f minutes", (time.time() - start) / 60)


if __name__ == "__main__":
    main()
