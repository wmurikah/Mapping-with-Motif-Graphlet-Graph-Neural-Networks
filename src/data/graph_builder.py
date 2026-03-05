"""
Heterogeneous interaction graph construction.

Constructs G = (V, E, T) where V is the set of nodes, E the set of
edges, and T a type mapping function (per thesis Section IV-C).

Node types:
    MOOCCubeX: student, course, concept
    OULAD:     student, module, activity

Edge types:
    MOOCCubeX: enrols, attempts, contains, prerequisite
    OULAD:     clicks, contains, co_occurrence
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def build_hetero_graph(
    edge_lists: Dict[str, np.ndarray],
    node_counts: Dict[str, int],
    node_features: Optional[Dict[str, np.ndarray]] = None,
    student_labels: Optional[np.ndarray] = None,
) -> HeteroData:
    """Build a PyG HeteroData object from edge lists and node counts.

    Parameters
    ----------
    edge_lists : dict
        Mapping from edge-type string to (2, E) numpy arrays.
        Edge-type strings should follow the convention:
            "source_relation_target" (e.g., "student_enrols_course")
    node_counts : dict
        Mapping from node type to number of nodes.
    node_features : dict or None
        Optional initial features per node type: {type: (N, D) array}.
    student_labels : array or None
        Binary labels for student nodes.

    Returns
    -------
    HeteroData
        PyG heterogeneous graph object ready for message passing.
    """
    data = HeteroData()

    # ------------------------------------------------------------------
    # Node features (or learnable embeddings if none provided)
    # ------------------------------------------------------------------
    for ntype, count in node_counts.items():
        if node_features and ntype in node_features:
            data[ntype].x = torch.tensor(
                node_features[ntype], dtype=torch.float32
            )
        else:
            # Initialise with identity-based features (one-hot is too large
            # for 185K students, so we use random normal initialisation that
            # will be replaced by learnable embeddings in the model).
            data[ntype].x = torch.randn(count, 16)
        data[ntype].num_nodes = count

    # ------------------------------------------------------------------
    # Edge indices
    # ------------------------------------------------------------------
    for edge_key, edge_array in edge_lists.items():
        src_type, relation, dst_type = _parse_edge_key(edge_key)
        edge_index = torch.tensor(edge_array, dtype=torch.long)

        # Validate indices
        if edge_index.shape[0] != 2:
            logger.warning("Skipping edge type %s: expected shape (2, E)", edge_key)
            continue

        src_max = node_counts.get(src_type, 0)
        dst_max = node_counts.get(dst_type, 0)
        valid_mask = (edge_index[0] < src_max) & (edge_index[1] < dst_max)
        edge_index = edge_index[:, valid_mask]

        if edge_index.shape[1] == 0:
            logger.warning("No valid edges for %s", edge_key)
            continue

        data[src_type, relation, dst_type].edge_index = edge_index
        logger.info(
            "Edge type (%s, %s, %s): %d edges",
            src_type, relation, dst_type, edge_index.shape[1],
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if student_labels is not None:
        data["student"].y = torch.tensor(student_labels, dtype=torch.long)

    return data


def build_concept_subgraph(
    hetero_data: HeteroData,
    concept_type: str = "concept",
) -> torch.Tensor:
    """Extract the concept-level adjacency matrix for motif analysis.

    Combines concept-prerequisite edges and concept co-participation
    edges (from shared courses) into a single directed adjacency matrix.

    Parameters
    ----------
    hetero_data : HeteroData
        The full heterogeneous graph.
    concept_type : str
        Node type name for concepts ("concept" or "activity").

    Returns
    -------
    torch.Tensor
        Sparse adjacency matrix of shape (num_concepts, num_concepts).
    """
    num_concepts = hetero_data[concept_type].num_nodes
    adj = torch.zeros(num_concepts, num_concepts)

    # Collect all concept-concept edge types
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if src_type == concept_type and dst_type == concept_type:
            ei = hetero_data[edge_type].edge_index
            adj[ei[0], ei[1]] = 1.0

    return adj.to_sparse()


def _parse_edge_key(key: str):
    """Parse edge key string into (src_type, relation, dst_type).

    Handles two conventions:
        "student_enrols_course"       → ("student", "enrols", "course")
        "student_clicks_activity"     → ("student", "clicks", "activity")
        "concept_prerequisite_concept" → ("concept", "prerequisite", "concept")
        "activity_co_occurrence"      → ("activity", "co_occurrence", "activity")
    """
    parts = key.split("_")

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 4:
        # e.g., "concept_prerequisite_concept" split as
        # ["concept", "prerequisite", "concept"] — already correct
        # But "activity_co_occurrence" → ["activity", "co", "occurrence"]
        # Try: first word = src, last word = dst, middle = relation
        return parts[0], "_".join(parts[1:-1]), parts[-1]
    elif len(parts) == 2:
        # e.g., "activity_co_occurrence" where both nodes are same type
        return parts[0], parts[1], parts[0]
    else:
        # Fallback: first and last are types, middle is relation
        return parts[0], "_".join(parts[1:-1]), parts[-1]


def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops to adjacency matrix: M_hat = M + I."""
    if adj.is_sparse:
        adj = adj.to_dense()
    n = adj.shape[0]
    return adj + torch.eye(n, device=adj.device)


def symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}.

    Used in Eq. 1 of the thesis for the motif-augmented GCN layer.
    """
    if adj.is_sparse:
        adj = adj.to_dense()

    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(
        deg > 0, deg.pow(-0.5), torch.zeros_like(deg)
    )
    D = torch.diag(deg_inv_sqrt)
    return D @ adj @ D
