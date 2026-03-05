"""
Main training and evaluation pipeline for MGI-GNN.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --dataset oulad
    python scripts/train.py --config configs/default.yaml --ablation no_motif
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mooccubex import MOOCCubeXLoader
from src.data.oulad import OULADLoader
from src.data.graph_builder import (
    build_hetero_graph,
    build_concept_subgraph,
    add_self_loops,
    symmetric_normalize,
)
from src.features.motif_extractor import MotifExtractor, motif_adjacency_to_torch
from src.features.graphlet_extractor import GraphletExtractor
from src.models.mgi_gnn import MGIGNN
from src.models.baselines import build_baseline
from src.evaluation.metrics import (
    compute_prediction_metrics,
    compute_cluster_metrics,
    evaluate_engagement_features,
    aggregate_runs,
    format_results_table,
    run_significance_tests,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, dataset: str) -> dict:
    """Apply dataset-specific overrides."""
    if dataset == "oulad" and "oulad_overrides" in config:
        overrides = config["oulad_overrides"]
        for section, values in overrides.items():
            if section in config:
                config[section].update(values)
    return config


# ======================================================================
# Data pipeline
# ======================================================================

def load_dataset(config: dict):
    """Load and preprocess the specified dataset."""
    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]

    if dataset_name == "mooccubex":
        loader = MOOCCubeXLoader(
            data_dir=os.path.join(data_dir, "mooccubex"),
            min_interactions=config["data"]["min_interactions"],
            min_concepts=config["data"]["min_concepts"],
            subject_filter="computer_science",
        )
        loader.load()
        concept_type = "concept"

    elif dataset_name == "oulad":
        loader = OULADLoader(
            data_dir=os.path.join(data_dir, "oulad"),
            co_occurrence_window=config["data"]["co_occurrence_window"],
            include_withdrawn=True,
        )
        loader.load()
        concept_type = "activity"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return loader, concept_type


def prepare_graph_data(loader, concept_type: str, config: dict):
    """Build graph, extract motifs and graphlets.

    Returns
    -------
    M_norm : torch.Tensor
        Normalised motif adjacency matrix.
    gdv : torch.Tensor
        Graphlet degree vectors.
    labels : np.ndarray
        Student outcome labels.
    node_counts : dict
    """
    logger.info("Building heterogeneous graph...")
    edge_lists = loader.get_edge_lists()
    node_counts = loader.get_node_counts()
    labels = loader.get_student_labels()

    hetero_data = build_hetero_graph(
        edge_lists, node_counts, student_labels=labels,
    )

    # Extract concept-level adjacency for motif/graphlet analysis
    logger.info("Extracting concept subgraph...")
    concept_adj = build_concept_subgraph(hetero_data, concept_type)
    concept_adj_dense = concept_adj.to_dense().numpy()

    # Motif extraction
    logger.info("Extracting motifs...")
    motif_extractor = MotifExtractor(
        normalize=config["features"]["motif"]["normalize"],
    )
    M = motif_extractor.extract(concept_adj_dense)
    M_torch = motif_adjacency_to_torch(M, add_self_loops=True)
    M_norm = symmetric_normalize(M_torch)

    # Graphlet extraction
    logger.info("Extracting graphlet degree vectors...")
    graphlet_extractor = GraphletExtractor(
        max_size=config["features"]["graphlet"]["max_size"],
        normalize=config["features"]["graphlet"]["normalize"],
    )
    gdv = graphlet_extractor.extract(concept_adj_dense)
    gdv_tensor = torch.tensor(gdv, dtype=torch.float32)

    return M_norm, gdv_tensor, labels, node_counts, hetero_data


def create_splits(
    labels: np.ndarray,
    split_ratio: list,
    seed: int,
) -> tuple:
    """Create stratified train/val/test splits.

    Parameters
    ----------
    labels : array (N,)
    split_ratio : [train, val, test] fractions
    seed : random seed

    Returns
    -------
    train_idx, val_idx, test_idx
    """
    n = len(labels)
    test_size = split_ratio[2]
    val_size = split_ratio[1] / (1 - test_size)

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    # Second split: train vs val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    labels_tv = labels[trainval_idx]
    train_rel, val_rel = next(sss2.split(np.zeros(len(trainval_idx)), labels_tv))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    logger.info(
        "Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx)
    )
    return train_idx, val_idx, test_idx


# ======================================================================
# Training loop
# ======================================================================

def train_mgi_gnn(
    model: MGIGNN,
    M_norm: torch.Tensor,
    gdv: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Train MGI-GNN with early stopping.

    Returns
    -------
    dict with "best_val_auc", "best_epoch", "train_history"
    """
    model = model.to(device)
    M_norm = M_norm.to(device)
    gdv = gdv.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
    )
    criterion = nn.CrossEntropyLoss()

    num_nodes = M_norm.shape[0]
    node_indices = torch.arange(num_nodes, device=device)

    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    history = {"train_loss": [], "val_auc": []}

    for epoch in range(config["training"]["epochs"]):
        # ---- Train ----
        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(node_indices, M_norm, gdv)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip"]
        )
        optimizer.step()

        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            logits, _, gate_values = model(node_indices, M_norm, gdv)
            val_probs = F.softmax(logits[val_idx], dim=-1).cpu().numpy()
            val_labels = labels[val_idx].cpu().numpy()

            metrics = compute_prediction_metrics(
                val_labels,
                val_probs.argmax(axis=1),
                val_probs,
            )

        val_auc = metrics["auc_roc"]
        scheduler.step(val_auc)

        history["train_loss"].append(loss.item())
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            gate_msg = ""
            if gate_values is not None:
                gate_msg = f", gate_mean={gate_values.mean().item():.3f}"
            logger.info(
                "Epoch %3d: loss=%.4f, val_AUC=%.4f%s",
                epoch, loss.item(), val_auc, gate_msg,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    logger.info(
        "Training complete. Best val AUC=%.4f at epoch %d",
        best_val_auc, best_epoch,
    )

    return {
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "train_history": history,
    }


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_model(
    model: MGIGNN,
    M_norm: torch.Tensor,
    gdv: torch.Tensor,
    labels: torch.Tensor,
    test_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Evaluate on test set: prediction metrics + clustering.

    Returns
    -------
    dict with prediction and clustering metrics.
    """
    model.eval()
    model = model.to(device)
    M_norm = M_norm.to(device)
    gdv = gdv.to(device)

    num_nodes = M_norm.shape[0]
    node_indices = torch.arange(num_nodes, device=device)

    with torch.no_grad():
        logits, embeddings, gate_values = model(node_indices, M_norm, gdv)

    # Prediction metrics
    test_probs = F.softmax(logits[test_idx], dim=-1).cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy() if isinstance(labels, torch.Tensor) else labels[test_idx]

    pred_metrics = compute_prediction_metrics(
        test_labels, test_probs.argmax(axis=1), test_probs,
    )

    # Clustering metrics
    dataset = config["data"]["dataset"]
    num_clusters = (
        config["model"]["cluster_detector"]["num_clusters_mooccubex"]
        if dataset == "mooccubex"
        else config["model"]["cluster_detector"]["num_clusters_oulad"]
    )

    cluster_metrics = compute_cluster_metrics(
        embeddings.cpu().numpy(),
        labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
        num_clusters=num_clusters,
    )

    # Gate analysis
    gate_info = {}
    if gate_values is not None:
        gate_info["mean_gate"] = gate_values.mean().item()
        gate_info["std_gate"] = gate_values.std().item()

    return {**pred_metrics, **cluster_metrics, **gate_info}


# ======================================================================
# Main experiment
# ======================================================================

def run_experiment(config: dict, ablation: str = None):
    """Run the full experiment pipeline.

    Executes num_runs independent training runs with different seeds,
    collects metrics, and reports mean ± std.
    """
    device = torch.device(
        config["experiment"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    # Apply ablation if specified
    if ablation:
        logger.info("Running ablation variant: %s", ablation)
        ablation_cfg = None
        for v in config.get("ablation", {}).get("variants", []):
            if v["name"] == ablation:
                ablation_cfg = v
                break
        if ablation_cfg is None:
            raise ValueError(f"Unknown ablation: {ablation}")
        if "model_name" in ablation_cfg:
            config["model"]["name"] = ablation_cfg["model_name"]
        if "fusion_method" in ablation_cfg:
            config["model"]["fusion"]["method"] = ablation_cfg["fusion_method"]

    # Load data
    loader, concept_type = load_dataset(config)
    M_norm, gdv, labels, node_counts, hetero_data = prepare_graph_data(
        loader, concept_type, config,
    )

    num_concept_nodes = node_counts.get(concept_type, node_counts.get("concept", 0))
    labels_tensor = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    # Multi-run experiment
    num_runs = config["experiment"]["num_runs"]
    all_results = []

    for run_idx in range(num_runs):
        seed = config["experiment"]["seed"] + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info("=" * 60)
        logger.info("Run %d/%d (seed=%d)", run_idx + 1, num_runs, seed)
        logger.info("=" * 60)

        # Create splits
        train_idx, val_idx, test_idx = create_splits(
            labels, config["data"]["split_ratio"], seed,
        )

        # Build model
        model_name = config["model"]["name"]

        if model_name == "mgi_gnn":
            model = MGIGNN(
                num_nodes=num_concept_nodes,
                embedding_dim=config["model"]["embedding_dim"],
                hidden_dims=config["model"]["motif_encoder"]["hidden_dims"],
                gdv_dim=config["features"]["graphlet"]["orbit_dim"],
                num_classes=config["model"]["competency_predictor"]["num_classes"],
                dropout=config["model"]["dropout"],
                fusion_method=config["model"]["fusion"]["method"],
                gate_hidden_dim=config["model"]["fusion"]["gate_hidden_dim"],
            )

            # Handle ablation: disable components
            if ablation and ablation_cfg:
                disable = ablation_cfg.get("disable")
                if disable == "motif_encoder":
                    # Zero out motif encoder contribution
                    for p in model.motif_encoder.parameters():
                        p.requires_grad = False
                        p.data.zero_()
                elif disable == "graphlet_encoder":
                    for p in model.graphlet_encoder.parameters():
                        p.requires_grad = False
                        p.data.zero_()

            # Train
            train_result = train_mgi_gnn(
                model, M_norm, gdv, labels_tensor,
                train_idx, val_idx, config, device,
            )

            # Evaluate
            run_metrics = evaluate_model(
                model, M_norm, gdv, labels_tensor,
                test_idx, config, device,
            )

        else:
            # Baseline model training
            logger.info("Training baseline: %s", model_name)

            # For baselines, we use standard adjacency instead of motif adjacency
            adj_norm = symmetric_normalize(
                add_self_loops(M_norm.cpu())  # fallback to available adjacency
            ).to(device)

            baseline = build_baseline(
                model_name,
                input_dim=config["model"]["embedding_dim"],
                hidden_dim=config["model"]["embedding_dim"],
                num_classes=config["model"]["competency_predictor"]["num_classes"],
                num_layers=config["model"]["num_gnn_layers"],
                dropout=config["model"]["dropout"],
                **({"num_nodes": num_concept_nodes} if model_name in ["gkt", "gikt", "lightgcn"] else {}),
                **({"num_concepts": num_concept_nodes} if model_name == "gkt" else {}),
            ).to(device)

            # Simplified training loop for baselines
            optimizer = torch.optim.Adam(
                baseline.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            criterion = nn.CrossEntropyLoss()
            node_indices = torch.arange(num_concept_nodes, device=device)

            for epoch in range(config["training"]["epochs"]):
                baseline.train()
                optimizer.zero_grad()
                logits = baseline(
                    baseline.embedding(node_indices) if hasattr(baseline, 'embedding') else torch.randn(num_concept_nodes, config["model"]["embedding_dim"], device=device),
                    adj_norm=adj_norm,
                )
                loss = criterion(logits[train_idx], labels_tensor[train_idx].to(device))
                loss.backward()
                optimizer.step()

            baseline.eval()
            with torch.no_grad():
                logits = baseline(
                    baseline.embedding(node_indices) if hasattr(baseline, 'embedding') else torch.randn(num_concept_nodes, config["model"]["embedding_dim"], device=device),
                    adj_norm=adj_norm,
                )

            test_probs = F.softmax(logits[test_idx], dim=-1).cpu().numpy()
            run_metrics = compute_prediction_metrics(
                labels[test_idx], test_probs.argmax(axis=1), test_probs,
            )

        all_results.append(run_metrics)
        logger.info("Run %d results: %s", run_idx + 1, run_metrics)

    # Aggregate across runs
    aggregated = aggregate_runs(all_results)
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATED RESULTS (%d runs):", num_runs)
    logger.info("=" * 60)
    for metric, (mean, std) in aggregated.items():
        logger.info("  %s: %.4f ± %.4f", metric, mean, std)

    # Save results
    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(
        output_dir,
        f"{config['model']['name']}_{config['data']['dataset']}_results.yaml",
    )
    with open(results_path, "w") as f:
        yaml.dump(
            {
                "config": config,
                "aggregated": {k: {"mean": float(m), "std": float(s)} for k, (m, s) in aggregated.items()},
                "per_run": all_results,
            },
            f,
            default_flow_style=False,
        )
    logger.info("Results saved to %s", results_path)

    return aggregated


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate MGI-GNN for educational assessment."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mooccubex", "oulad"],
        help="Override dataset in config.",
    )
    parser.add_argument(
        "--ablation", type=str,
        choices=["no_motif", "no_graphlet", "no_gated_attention", "no_higher_order"],
        help="Run an ablation variant.",
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"],
        help="Override device.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.device:
        config["experiment"]["device"] = args.device

    config = apply_overrides(config, config["data"]["dataset"])

    start_time = time.time()
    run_experiment(config, ablation=args.ablation)
    elapsed = time.time() - start_time
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
