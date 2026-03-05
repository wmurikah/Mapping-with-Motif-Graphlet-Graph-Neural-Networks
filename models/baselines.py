"""
Baseline models for comparison (thesis Table 2).

Five baselines spanning different structural representation levels:
    MLP      — feedforward, no graph structure
    GCN      — pairwise (1-hop neighbours)
    GKT      — pairwise (concept graph), education-specific
    GIKT     — pairwise (question-skill), education-specific
    LightGCN — multi-hop neighbourhood, simplified GCN for rec systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ======================================================================
# MLP Baseline (no graph structure)
# ======================================================================

class MLPBaseline(nn.Module):
    """Multi-layer perceptron operating on flat features.

    Structural level: None (flat features only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.network(x)


# ======================================================================
# GCN Baseline
# ======================================================================

class GCNLayer(nn.Module):
    """Standard graph convolutional layer (Kipf & Welling, 2017).

    h^(l) = σ( D̃^{-1/2} Ã D̃^{-1/2} h^(l-1) W^(l) )
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return adj_norm @ x @ self.weight + self.bias


class GCNBaseline(nn.Module):
    """Graph Convolutional Network baseline.

    Structural level: pairwise (1-hop neighbours).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout

        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(len(dims) - 1):
            self.layers.append(GCNLayer(dims[i], dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, adj_norm: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x, adj_norm)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# ======================================================================
# GKT Baseline (Graph-based Knowledge Tracing)
# ======================================================================

class GKTBaseline(nn.Module):
    """Graph-based Knowledge Tracing (Nakagawa et al., 2019).

    Propagates student mastery signals through concept graph.
    Structural level: pairwise (concept graph).

    Reference:
        Nakagawa et al. "Graph-based Knowledge Tracing:
        Modeling Student Proficiency Using GNN." WI 2019.
    """

    def __init__(
        self,
        num_concepts: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.concept_embed = nn.Embedding(num_concepts, embedding_dim)

        # GCN layers for concept graph
        self.gcn_layers = nn.ModuleList()
        dims = [embedding_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.gcn_layers.append(GCNLayer(dims[i], dims[i + 1]))

        # Mastery prediction
        self.mastery_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.dropout = dropout

    def forward(
        self,
        concept_indices: torch.Tensor,
        adj_norm: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x = self.concept_embed(concept_indices)
        for layer in self.gcn_layers:
            x = F.relu(layer(x, adj_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mastery_predictor(x)


# ======================================================================
# GIKT Baseline (Graph Interaction Knowledge Tracing)
# ======================================================================

class GIKTBaseline(nn.Module):
    """Graph-based Interaction model for Knowledge Tracing.

    Uses GCN to capture correlations between questions and skills
    in a unified embedding space.

    Structural level: pairwise (question-skill interactions).

    Reference:
        Yang et al. "GIKT: A Graph-based Interaction Model
        for Knowledge Tracing." ECML-PKDD 2020.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, embedding_dim)

        # Interaction-aware GCN
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        dims = [embedding_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.gcn_layers.append(GCNLayer(dims[i], dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        # Attention mechanism for interaction modelling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.dropout = dropout

    def forward(
        self,
        node_indices: torch.Tensor,
        adj_norm: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x = self.node_embed(node_indices)

        for layer, bn in zip(self.gcn_layers, self.batch_norms):
            x = layer(x, adj_norm)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Self-attention over node embeddings
        x_unsqueezed = x.unsqueeze(0)  # (1, N, D)
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = x + attn_out.squeeze(0)  # residual

        return self.classifier(x)


# ======================================================================
# LightGCN Baseline
# ======================================================================

class LightGCNBaseline(nn.Module):
    """LightGCN: simplified GCN for collaborative filtering.

    Strips out nonlinear transformations and aggregates multi-hop
    neighbourhood information through simple weighted averaging.

    Structural level: multi-hop neighbourhood.

    Reference:
        He et al. "LightGCN: Simplifying and Powering Graph
        Convolution Network for Recommendation." SIGIR 2020.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout

        # Layer combination weights (learnable)
        self.alpha = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1))

    def forward(
        self,
        node_indices: torch.Tensor,
        adj_norm: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x = self.embedding(node_indices)

        # LightGCN: no nonlinear activation, no feature transformation
        layer_outputs = [x]
        current = x
        for _ in range(self.num_layers):
            current = adj_norm @ current
            current = F.dropout(current, p=self.dropout, training=self.training)
            layer_outputs.append(current)

        # Weighted combination of all layers
        alpha = F.softmax(self.alpha, dim=0)
        x_final = sum(a * h for a, h in zip(alpha, layer_outputs))

        return self.classifier(x_final)


# ======================================================================
# Factory
# ======================================================================

BASELINE_REGISTRY = {
    "mlp": MLPBaseline,
    "gcn": GCNBaseline,
    "gkt": GKTBaseline,
    "gikt": GIKTBaseline,
    "lightgcn": LightGCNBaseline,
}


def build_baseline(name: str, **kwargs) -> nn.Module:
    """Instantiate a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name](**kwargs)
