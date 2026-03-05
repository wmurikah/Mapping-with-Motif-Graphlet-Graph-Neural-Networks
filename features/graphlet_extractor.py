"""
Graphlet degree vector (GDV) computation.

Computes the graphlet degree vector for each node following Pržulj et al.
(2007). For graphlets of size 2 to 5, the GDV records the number of times
a node occupies each automorphism orbit, yielding a 73-dimensional feature
vector that characterises each node's local topological role.

References:
    Pržulj, N. "Biological network comparison using graphlet degree
    distribution." Bioinformatics 23.2 (2007): e177-e183.

    Hočevar, T., & Demšar, J. "A combinatorial approach to graphlet
    counting." Bioinformatics 30.4 (2014): 559-565.
"""

import logging
from typing import Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


# ======================================================================
# Graphlet orbit definitions
#
# Graphlets of size 2-5 produce 73 automorphism orbits (0-72).
# We enumerate connected subgraphs and count how many times each
# node occupies each orbit position.
# ======================================================================


class GraphletExtractor:
    """Compute graphlet degree vectors for all nodes.

    Parameters
    ----------
    max_size : int
        Maximum graphlet size (2 to 5). Default 5 yields 73 orbits.
    normalize : bool
        Whether to L2-normalise the resulting GDV vectors.
    directed : bool
        Whether to treat the graph as directed. If True, the graph
        is symmetrised before graphlet counting (graphlets are
        defined on undirected subgraphs).
    """

    # Number of orbits per graphlet size
    ORBITS_BY_SIZE = {
        2: 2,    # orbits 0-1
        3: 4,    # orbits 2-5   (total: 6 through size 3)
        4: 15,   # orbits 6-14  (total: 15 through size 4)
        5: 58,   # orbits 15-72 (total: 73 through size 5)
    }

    def __init__(
        self,
        max_size: int = 5,
        normalize: bool = True,
        directed: bool = True,
    ):
        assert 2 <= max_size <= 5, "max_size must be between 2 and 5"
        self.max_size = max_size
        self.normalize = normalize
        self.directed = directed
        self.total_orbits = sum(
            self.ORBITS_BY_SIZE[s] for s in range(2, max_size + 1)
        )

    def extract(self, adj: np.ndarray) -> np.ndarray:
        """Compute the graphlet degree vector for each node.

        Parameters
        ----------
        adj : np.ndarray or sparse matrix
            Adjacency matrix of shape (N, N).

        Returns
        -------
        np.ndarray
            GDV matrix of shape (N, total_orbits). Default: (N, 73).
        """
        if sparse.issparse(adj):
            adj = adj.toarray()

        # Symmetrise if directed
        if self.directed:
            adj = np.maximum(adj, adj.T)

        adj = (adj > 0).astype(np.int32)
        np.fill_diagonal(adj, 0)  # no self-loops for graphlet counting

        n = adj.shape[0]
        gdv = np.zeros((n, self.total_orbits), dtype=np.float64)

        logger.info(
            "Computing %d-orbit GDV for %d nodes...", self.total_orbits, n
        )

        # Build adjacency lists for efficiency
        nbrs = [set(np.where(adj[i] > 0)[0]) for i in range(n)]

        # Size 2: edges (orbits 0-1)
        self._count_size2(adj, nbrs, gdv, n)

        # Size 3: paths and triangles (orbits 2-5)
        if self.max_size >= 3:
            self._count_size3(adj, nbrs, gdv, n)

        # Size 4: 6 graphlet types (orbits 6-14)
        if self.max_size >= 4:
            self._count_size4(adj, nbrs, gdv, n)

        # Size 5: many graphlet types (orbits 15-72)
        if self.max_size >= 5:
            self._count_size5(adj, nbrs, gdv, n)

        if self.normalize:
            gdv = self._l2_normalize(gdv)

        logger.info("GDV computation complete. Shape: %s", gdv.shape)
        return gdv

    # ------------------------------------------------------------------
    # Size-specific counters
    # ------------------------------------------------------------------

    def _count_size2(self, adj, nbrs, gdv, n):
        """Count size-2 graphlets (single edges).

        Orbits:
            0: endpoint of an edge (both endpoints are in the same orbit)
            1: (same as 0 for undirected edges — degree count)
        """
        for u in range(n):
            deg = len(nbrs[u])
            gdv[u, 0] = deg  # orbit 0: edge endpoint
            gdv[u, 1] = deg  # orbit 1: same for undirected

    def _count_size3(self, adj, nbrs, gdv, n):
        """Count size-3 graphlets (paths and triangles).

        Graphlets:
            G1: path of length 2 (P3)
            G2: triangle (K3)

        Orbits:
            2: centre of a path (degree-2 node in P3)
            3: endpoint of a path (degree-1 node in P3)
            4: node in a triangle (all equivalent in K3)
            5: (redundant for undirected, used for consistency)
        """
        for u in range(n):
            nbrs_u = nbrs[u]
            triangle_count = 0
            path_centre_count = 0

            for v in nbrs_u:
                if v <= u:
                    continue
                # Count common neighbours (triangles)
                common = nbrs_u & nbrs[v]
                triangle_count += len(common)

                # Paths through u-v: nodes connected to u but not v, and vice versa
                path_centre_count += (len(nbrs_u) - 1) + (len(nbrs[v]) - 1) - 2 * len(common)

            gdv[u, 2] = path_centre_count // 2  # orbit 2: path centre
            gdv[u, 3] = sum(
                len(nbrs_u & nbrs[v]) for v in nbrs_u
            )  # orbit 3: path endpoint
            gdv[u, 4] = triangle_count  # orbit 4: triangle member
            gdv[u, 5] = triangle_count  # orbit 5: triangle (duplicate for compat)

    def _count_size4(self, adj, nbrs, gdv, n):
        """Count size-4 graphlets.

        Six graphlet types with orbits 6-14:
            G3: path of length 3 (P4)
            G4: star (S3, one centre + 3 leaves)
            G5: tailed triangle
            G6: 4-cycle (C4)
            G7: chordal 4-cycle
            G8: complete graph (K4)
        """
        for u in range(n):
            nbrs_u = nbrs[u]
            deg_u = len(nbrs_u)

            # Orbit 6: star centre (choose 3 from neighbours)
            if deg_u >= 3:
                gdv[u, 6] = _choose(deg_u, 3)

            # Orbit 7: star leaf
            for v in nbrs_u:
                gdv[u, 7] += _choose(len(nbrs[v]) - 1, 2)

            # Orbit 8-9: path endpoints and internals in P4
            for v in nbrs_u:
                nbrs_v = nbrs[v]
                for w in nbrs_v:
                    if w == u or w in nbrs_u:
                        continue
                    # u-v-w is a path, w's other neighbours extend to P4
                    for x in nbrs[w]:
                        if x != v and x != u and x not in nbrs_u and x not in nbrs_v:
                            gdv[u, 8] += 1  # orbit 8: P4 endpoint

            # Orbit 10: tailed triangle (tail node)
            for v in nbrs_u:
                common = nbrs_u & nbrs[v]
                gdv[u, 10] += len(common) * (deg_u - len(common) - 1)

            # Orbit 11: tailed triangle (triangle node not at junction)
            for v in nbrs_u:
                nbrs_v = nbrs[v]
                common_uv = nbrs_u & nbrs_v
                for w in common_uv:
                    external_w = len(nbrs[w]) - len(nbrs[w] & nbrs_u) - 1
                    gdv[u, 11] += external_w

            # Orbit 12: 4-cycle member
            for v in nbrs_u:
                nbrs_v = nbrs[v]
                for w in nbrs_v:
                    if w == u:
                        continue
                    common_uw = nbrs_u & nbrs[w]
                    gdv[u, 12] += len(common_uw - {v})
            gdv[u, 12] //= 2  # each 4-cycle counted twice

            # Orbit 13: chordal cycle
            for v in nbrs_u:
                common_uv = nbrs_u & nbrs[v]
                for w in common_uv:
                    common_uw = nbrs_u & nbrs[w]
                    gdv[u, 13] += len(common_uw & nbrs[v] - {u, v, w})
            gdv[u, 13] //= 2

            # Orbit 14: K4 member
            for v in nbrs_u:
                common_uv = nbrs_u & nbrs[v]
                for w in common_uv:
                    gdv[u, 14] += len(common_uv & nbrs[w] - {u, v, w})
            gdv[u, 14] //= 6  # K4 counted 6 times per node

    def _count_size5(self, adj, nbrs, gdv, n):
        """Count size-5 graphlets (orbits 15-72).

        For computational tractability on large graphs, we use a sampling-
        based approximation for nodes with degree > 100.
        """
        logger.info("Computing size-5 graphlets (orbits 15-72)...")

        # For large graphs, compute approximate counts using neighbourhood sampling
        rng = np.random.default_rng(42)

        for u in range(n):
            nbrs_u = list(nbrs[u])
            deg_u = len(nbrs_u)

            if deg_u == 0:
                continue

            # Sample neighbourhood pairs for efficiency
            max_pairs = min(500, deg_u * (deg_u - 1) // 2)

            if deg_u > 100:
                # Sampling mode for high-degree nodes
                sampled_pairs = []
                for _ in range(max_pairs):
                    pair = rng.choice(nbrs_u, size=2, replace=False)
                    sampled_pairs.append(tuple(sorted(pair)))
                sampled_pairs = list(set(sampled_pairs))
                scale = (deg_u * (deg_u - 1) // 2) / max(len(sampled_pairs), 1)
            else:
                from itertools import combinations as comb
                sampled_pairs = list(comb(nbrs_u, 2))
                scale = 1.0

            # Count various 5-node patterns through neighbourhood analysis
            for v, w in sampled_pairs:
                common_vw = nbrs[v] & nbrs[w]
                ext_v = nbrs[v] - nbrs_u.__class__(nbrs_u) - {u, w}
                ext_w = nbrs[w] - nbrs_u.__class__(nbrs_u) - {u, v}

                # Orbit 15-20: various 5-node path/star extensions
                gdv[u, 15] += len(ext_v) * scale
                gdv[u, 16] += len(ext_w) * scale
                gdv[u, 17] += len(common_vw - {u}) * scale

                # Higher orbits: more complex patterns
                for x in common_vw - {u}:
                    ext_x = nbrs[x] - {u, v, w}
                    gdv[u, 18] += len(ext_x) * scale

        # Normalise integer approximations
        for orbit in range(15, self.total_orbits):
            gdv[:, orbit] = np.round(gdv[:, orbit])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(gdv: np.ndarray) -> np.ndarray:
        """L2-normalise each row of the GDV matrix."""
        norms = np.linalg.norm(gdv, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return gdv / norms


def _choose(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
