"""
Motif adjacency matrix construction.

Extracts network motifs (recurring subgraph patterns) from the concept-level
graph following Milo et al. (2002). For directed graphs, we enumerate 13
three-node and four-node connected subgraph patterns and construct a motif
adjacency matrix M where M_{i,j} records the number of motif instances in
which nodes i and j co-participate.

References:
    Milo et al. "Network motifs: Simple building blocks of complex networks."
    Science 298.5594 (2002): 824-827.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse

logger = logging.getLogger(__name__)


# ======================================================================
# 13 directed three-node motif types (Milo et al. canonical numbering)
# Each motif is defined by its edge pattern among 3 nodes {0, 1, 2}.
# Format: list of directed edges (source, target).
# ======================================================================
THREE_NODE_MOTIFS = {
    # --- Chains and paths ---
    "M1_chain":         [(0, 1), (1, 2)],                      # A→B→C
    "M2_reverse_chain": [(1, 0), (2, 1)],                      # A←B←C
    "M3_mixed_chain":   [(0, 1), (2, 1)],                      # A→B←C (fan-in)
    "M4_fan_out":       [(1, 0), (1, 2)],                      # A←B→C (fan-out)

    # --- Triangles (3-clique variants) ---
    "M5_cycle":         [(0, 1), (1, 2), (2, 0)],              # directed cycle
    "M6_full_triangle": [(0, 1), (1, 2), (0, 2)],              # feed-forward loop
    "M7_mutual_in":     [(0, 2), (1, 2), (0, 1)],              # convergent triangle
    "M8_mutual_out":    [(2, 0), (2, 1), (0, 1)],              # divergent triangle

    # --- Partial reciprocal ---
    "M9_reciprocal_chain":  [(0, 1), (1, 0), (1, 2)],          # mutual + chain
    "M10_reciprocal_fan":   [(0, 1), (1, 0), (0, 2)],          # mutual + fan
    "M11_reciprocal_cycle": [(0, 1), (1, 0), (1, 2), (2, 0)],  # mutual + cycle

    # --- Full reciprocal ---
    "M12_double_reciprocal": [(0, 1), (1, 0), (0, 2), (2, 0)],
    "M13_full_reciprocal":   [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)],
}


class MotifExtractor:
    """Extract motif adjacency matrix from a directed graph.

    Parameters
    ----------
    motif_types : dict or None
        Dictionary of motif name → edge patterns. If None, uses the
        13 canonical directed three-node motifs.
    normalize : bool
        Whether to row-normalise the resulting motif adjacency matrix.
    """

    def __init__(
        self,
        motif_types: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        normalize: bool = True,
    ):
        self.motif_types = motif_types or THREE_NODE_MOTIFS
        self.normalize = normalize

    def extract(self, adj: np.ndarray) -> np.ndarray:
        """Construct the motif adjacency matrix M.

        For each motif type, enumerates all instances in the graph and
        increments M_{i,j} for every pair (i, j) of nodes that co-participate
        in a motif instance.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix of shape (N, N). Can be dense or will be
            converted from sparse.

        Returns
        -------
        np.ndarray
            Motif adjacency matrix M of shape (N, N).
        """
        if sparse.issparse(adj):
            adj = adj.toarray()
        adj = (adj > 0).astype(np.float32)

        n = adj.shape[0]
        M = np.zeros((n, n), dtype=np.float64)

        logger.info("Extracting motifs from graph with %d nodes...", n)

        total_instances = 0
        motif_counts = {}

        for motif_name, edge_pattern in self.motif_types.items():
            count = self._count_motif(adj, edge_pattern, M, n)
            motif_counts[motif_name] = count
            total_instances += count

        logger.info(
            "Found %d total motif instances across %d types",
            total_instances, len(self.motif_types),
        )
        for name, count in motif_counts.items():
            if count > 0:
                logger.debug("  %s: %d instances", name, count)

        # Symmetrise (M is accumulated from both directions)
        M = (M + M.T) / 2.0

        if self.normalize:
            M = self._row_normalize(M)

        return M

    def _count_motif(
        self,
        adj: np.ndarray,
        edge_pattern: List[Tuple[int, int]],
        M: np.ndarray,
        n: int,
    ) -> int:
        """Count instances of a specific motif pattern.

        Uses the adjacency list representation for efficiency.
        For three-node motifs, iterates over all node triples where
        at least one edge exists.
        """
        count = 0
        motif_size = max(max(e) for e in edge_pattern) + 1

        if motif_size == 3:
            count = self._enumerate_3node(adj, edge_pattern, M, n)
        elif motif_size == 4:
            count = self._enumerate_4node(adj, edge_pattern, M, n)

        return count

    def _enumerate_3node(
        self,
        adj: np.ndarray,
        pattern: List[Tuple[int, int]],
        M: np.ndarray,
        n: int,
    ) -> int:
        """Enumerate three-node motif instances.

        For scalability, we use adjacency list iteration rather than
        brute-force O(n^3) enumeration.
        """
        count = 0

        # Build adjacency lists
        out_nbrs = [set(np.where(adj[i] > 0)[0]) for i in range(n)]
        in_nbrs = [set(np.where(adj[:, i] > 0)[0]) for i in range(n)]

        # Iterate over all candidate triples via edge-based enumeration
        for u in range(n):
            # Collect all nodes connected to u (in or out)
            connected = out_nbrs[u] | in_nbrs[u]
            connected_list = sorted(connected)

            for i, v in enumerate(connected_list):
                for w in connected_list[i + 1:]:
                    if v == u or w == u or v == w:
                        continue
                    nodes = [u, v, w]
                    if self._matches_pattern(adj, nodes, pattern):
                        count += 1
                        # Increment M for all co-participating pairs
                        for a, b in combinations(nodes, 2):
                            M[a, b] += 1
                            M[b, a] += 1

        # Each triple is counted multiple times (once per starting node),
        # so we divide by the motif size to avoid overcounting
        count //= 3
        return count

    def _enumerate_4node(
        self,
        adj: np.ndarray,
        pattern: List[Tuple[int, int]],
        M: np.ndarray,
        n: int,
    ) -> int:
        """Enumerate four-node motif instances (sampling for large graphs)."""
        count = 0

        # For large graphs, use sampling
        if n > 5000:
            return self._sample_4node(adj, pattern, M, n)

        out_nbrs = [set(np.where(adj[i] > 0)[0]) for i in range(n)]
        in_nbrs = [set(np.where(adj[:, i] > 0)[0]) for i in range(n)]

        for u in range(n):
            connected_u = out_nbrs[u] | in_nbrs[u]
            for v in connected_u:
                if v <= u:
                    continue
                connected_v = out_nbrs[v] | in_nbrs[v]
                candidates = (connected_u | connected_v) - {u, v}
                for w in candidates:
                    connected_w = out_nbrs[w] | in_nbrs[w]
                    candidates2 = (candidates & (connected_u | connected_v | connected_w)) - {u, v, w}
                    for x in candidates2:
                        if x <= w:
                            continue
                        nodes = [u, v, w, x]
                        if self._matches_pattern(adj, nodes, pattern):
                            count += 1
                            for a, b in combinations(nodes, 2):
                                M[a, b] += 1
                                M[b, a] += 1

        count //= 4
        return count

    def _sample_4node(
        self,
        adj: np.ndarray,
        pattern: List[Tuple[int, int]],
        M: np.ndarray,
        n: int,
        num_samples: int = 100000,
    ) -> int:
        """Sample-based approximation for four-node motifs in large graphs."""
        rng = np.random.default_rng(42)
        count = 0

        edges = np.array(np.where(adj > 0)).T
        if len(edges) == 0:
            return 0

        for _ in range(num_samples):
            # Sample a seed edge
            ei = rng.integers(len(edges))
            u, v = edges[ei]

            # Sample two more connected nodes
            connected = set(np.where(adj[u] + adj[v] + adj[:, u] + adj[:, v] > 0)[0]) - {u, v}
            if len(connected) < 2:
                continue

            sample = rng.choice(list(connected), size=min(2, len(connected)), replace=False)
            nodes = [u, v] + list(sample)

            if self._matches_pattern(adj, nodes, pattern):
                count += 1
                for a, b in combinations(nodes, 2):
                    M[a, b] += 1
                    M[b, a] += 1

        # Scale estimate
        scale_factor = (n * (n - 1) * (n - 2) * (n - 3)) / (24 * num_samples)
        return int(count * scale_factor)

    @staticmethod
    def _matches_pattern(
        adj: np.ndarray,
        nodes: List[int],
        pattern: List[Tuple[int, int]],
    ) -> bool:
        """Check whether the subgraph induced by `nodes` matches `pattern`."""
        for src_idx, dst_idx in pattern:
            if adj[nodes[src_idx], nodes[dst_idx]] == 0:
                return False
        return True

    @staticmethod
    def _row_normalize(M: np.ndarray) -> np.ndarray:
        """Row-normalise the motif adjacency matrix."""
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return M / row_sums


def motif_adjacency_to_torch(
    M: np.ndarray,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Convert motif adjacency matrix to torch tensor with self-loops.

    Computes M_hat = M + I (Eq. 1 in thesis).

    Parameters
    ----------
    M : np.ndarray
        Motif adjacency matrix of shape (N, N).
    add_self_loops : bool
        Whether to add identity matrix (self-loops).

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (N, N).
    """
    M_t = torch.tensor(M, dtype=torch.float32)
    if add_self_loops:
        M_t = M_t + torch.eye(M_t.shape[0])
    return M_t
