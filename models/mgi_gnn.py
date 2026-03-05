"""
MGI-GNN: Motif-Graphlet Integrated Graph Neural Network.

Architecture comprising three components:

1. Motif encoder: applies graph convolutional layers over the motif
   adjacency matrix M rather than the standard adjacency matrix.
   At layer l, the node embedding update follows Eq. 1:
       h^(l)_i = σ( D̂^{M-1/2} M̂ D̂^{M-1/2} h^(l-1)_i W^(l) )
   where M̂ = M + I, D̂^M is its degree matrix, W^(l) the learnable
   weight matrix, and σ the ReLU activation function.

2. Graphlet encoder: concatenates each node's GDV with its initial
   feature embedding and passes through a two-layer feedforward network.

3. Gated attention fusion: combines both streams through a learned
   gate that assigns task-specific weights to each view.

Reference:
    Adapted from the digital library recommendation architecture
    described in the thesis Section IV-D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MotifGCNLayer(nn.Module):
    """Single GCN layer operating over the motif adjacency matrix.

    Implements Eq. 1: h^(l) = σ( D̂^{-1/2} M̂ D̂^{-1/2} h^(l-1) W^(l) )

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool
        Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, M_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (N, in_features)
            Node feature matrix.
        M_norm : Tensor of shape (N, N)
            Symmetrically normalised motif adjacency matrix
            (D̂^{-1/2} M̂ D̂^{-1/2}).

        Returns
        -------
        Tensor of shape (N, out_features)
        """
        # h = M_norm @ x @ W
        support = x @ self.weight
        out = M_norm @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class MotifEncoder(nn.Module):
    """Multi-layer GCN encoder over the motif adjacency matrix.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list of int
        Hidden layer dimensions.
    dropout : float
        Dropout rate.
    use_batch_norm : bool
        Whether to apply batch normalisation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = dropout

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.layers.append(MotifGCNLayer(dims[i], dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor, M_norm: torch.Tensor) -> torch.Tensor:
        """Encode nodes through motif-augmented GCN layers.

        Parameters
        ----------
        x : Tensor (N, input_dim)
        M_norm : Tensor (N, N)
            Normalised motif adjacency matrix.

        Returns
        -------
        Tensor (N, hidden_dims[-1])
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, M_norm)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphletEncoder(nn.Module):
    """Feedforward encoder for graphlet degree vectors.

    Concatenates each node's GDV with its initial feature embedding
    and passes through a two-layer feedforward network.

    Parameters
    ----------
    initial_feat_dim : int
        Dimension of the initial node features.
    gdv_dim : int
        Dimension of the graphlet degree vector (73 for size-5 graphlets).
    hidden_dims : list of int
        Hidden layer dimensions.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        initial_feat_dim: int,
        gdv_dim: int = 73,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        input_dim = initial_feat_dim + gdv_dim

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, gdv: torch.Tensor
    ) -> torch.Tensor:
        """Encode node features with graphlet structural information.

        Parameters
        ----------
        x : Tensor (N, initial_feat_dim)
            Initial node feature embeddings.
        gdv : Tensor (N, gdv_dim)
            Graphlet degree vectors.

        Returns
        -------
        Tensor (N, hidden_dims[-1])
        """
        combined = torch.cat([x, gdv], dim=-1)
        return self.network(combined)


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion layer.

    Learns task-specific weights for combining motif and graphlet
    streams through a gating mechanism.

    Parameters
    ----------
    embed_dim : int
        Dimension of each stream's output.
    gate_hidden_dim : int
        Hidden dimension of the gating network.
    """

    def __init__(self, embed_dim: int, gate_hidden_dim: int = 64):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(2 * embed_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, h_motif: torch.Tensor, h_graphlet: torch.Tensor
    ) -> torch.Tensor:
        """Fuse motif and graphlet representations.

        Parameters
        ----------
        h_motif : Tensor (N, embed_dim)
            Motif encoder output.
        h_graphlet : Tensor (N, embed_dim)
            Graphlet encoder output.

        Returns
        -------
        Tensor (N, embed_dim)
            Fused representation.
        gate_values : Tensor (N, 1)
            Learned gate values (for analysis; higher = more motif weight).
        """
        combined = torch.cat([h_motif, h_graphlet], dim=-1)
        gate = self.gate_network(combined)  # (N, 1)
        fused = gate * h_motif + (1 - gate) * h_graphlet
        return fused, gate


class MGIGNN(nn.Module):
    """Motif-Graphlet Integrated Graph Neural Network.

    Full architecture combining:
        1. Learnable node embedding layer
        2. Motif encoder (GCN over motif adjacency)
        3. Graphlet encoder (MLP on features + GDV)
        4. Gated attention fusion
        5. Task-specific prediction head

    Parameters
    ----------
    num_nodes : int
        Total number of concept-level nodes.
    initial_feat_dim : int
        Dimension of initial node features (or embedding dim if learned).
    embedding_dim : int
        Dimension of the learned node embeddings.
    hidden_dims : list of int
        Hidden dimensions for encoders.
    gdv_dim : int
        Graphlet degree vector dimension (default 73).
    num_classes : int
        Number of output classes for prediction.
    dropout : float
        Dropout rate.
    fusion_method : str
        "gated_attention", "concatenation", or "mean".
    gate_hidden_dim : int
        Hidden dimension for the gating network.
    use_batch_norm : bool
        Whether to use batch normalisation in encoders.
    """

    def __init__(
        self,
        num_nodes: int,
        initial_feat_dim: int = 16,
        embedding_dim: int = 128,
        hidden_dims: list = None,
        gdv_dim: int = 73,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_method: str = "gated_attention",
        gate_hidden_dim: int = 64,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.fusion_method = fusion_method

        # Learnable node embeddings
        self.node_embedding = nn.Embedding(num_nodes, initial_feat_dim)

        # Projection to embedding_dim
        self.input_proj = nn.Linear(initial_feat_dim, embedding_dim)

        # Motif encoder
        self.motif_encoder = MotifEncoder(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # Graphlet encoder
        self.graphlet_encoder = GraphletEncoder(
            initial_feat_dim=embedding_dim,
            gdv_dim=gdv_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Fusion
        out_dim = hidden_dims[-1]
        if fusion_method == "gated_attention":
            self.fusion = GatedAttentionFusion(out_dim, gate_hidden_dim)
            self.fused_dim = out_dim
        elif fusion_method == "concatenation":
            self.fusion = None
            self.fused_dim = 2 * out_dim
        elif fusion_method == "mean":
            self.fusion = None
            self.fused_dim = out_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim // 2, num_classes),
        )

    def forward(
        self,
        node_indices: torch.Tensor,
        M_norm: torch.Tensor,
        gdv: torch.Tensor,
        x_initial: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Parameters
        ----------
        node_indices : LongTensor (N,)
            Node indices for embedding lookup.
        M_norm : Tensor (N, N)
            Normalised motif adjacency matrix.
        gdv : Tensor (N, 73)
            Graphlet degree vectors.
        x_initial : Tensor (N, D) or None
            Optional initial node features. If None, uses learned embeddings.

        Returns
        -------
        logits : Tensor (N, num_classes)
            Class logits for each node.
        embeddings : Tensor (N, fused_dim)
            Fused node representations (for clustering).
        gate_values : Tensor (N, 1) or None
            Gate values from gated attention (None if not used).
        """
        # Get node representations
        if x_initial is not None:
            x = self.input_proj(x_initial)
        else:
            x = self.input_proj(self.node_embedding(node_indices))

        # Dual-view encoding
        h_motif = self.motif_encoder(x, M_norm)
        h_graphlet = self.graphlet_encoder(x, gdv)

        # Fusion
        gate_values = None
        if self.fusion_method == "gated_attention":
            fused, gate_values = self.fusion(h_motif, h_graphlet)
        elif self.fusion_method == "concatenation":
            fused = torch.cat([h_motif, h_graphlet], dim=-1)
        elif self.fusion_method == "mean":
            fused = (h_motif + h_graphlet) / 2.0

        # Prediction
        logits = self.predictor(fused)

        return logits, fused, gate_values

    def get_embeddings(
        self,
        node_indices: torch.Tensor,
        M_norm: torch.Tensor,
        gdv: torch.Tensor,
        x_initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get node embeddings without prediction (for clustering)."""
        _, embeddings, _ = self.forward(node_indices, M_norm, gdv, x_initial)
        return embeddings.detach()
