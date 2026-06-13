"""
PSO-based K-Means Clustering with Enhanced Cluster Matching (PSO-KM CM2)
=========================================================================
Reconstructed from B.Tech thesis:
"Optimal User Association, Backhaul Routing and Switching Off in
5G HetNets with Mesh mm-Wave Backhaul Links"
CVR College of Engineering, JNTUH, 2023-24

Original algorithm reference:
  Lam, Y.K., Tsang, P.W.M., Leung, C.S. (2012).
  PSO-based K-Means clustering with enhanced cluster matching
  for gene expression data.

Equations from thesis:
  (1) Particle position:  X_{i,n} = (z_{i,n,0}, ..., z_{i,n,K-1})
  (2) Velocity update:    V_{i,j,n+1} = w*V_{i,j,n}
                                       + c1*r1*(P_{i,j,n} - X_{i,j,n})
                                       + c2*r2*(G - X_{i,j,n})
  (3) Position update:    X_{i,n+1} = X_{i,n} + V_{i,n+1}
  (4) MSE fitness:        (1/N) * sum_j sum_{xi in Cj} ||xi - zj||

CM2 contribution:
  Before each PSO velocity update, rearrange centroid sequence in every
  particle to best match the global best particle (by minimum-distance
  pairing). This prevents centroid-ordering divergence during PSO updates.

Parameters (from Table 1 in thesis):
  Swarm size = 6, w = 0.7, c1 = 0.2, c2 = 0.2, max_iter = 40
"""

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# MSE Fitness
# ---------------------------------------------------------------------------

def compute_mse(data: np.ndarray, centroids: np.ndarray) -> float:
    """
    Mean Squared Error across all clusters.
    Lower = tighter, better-quality clusters.

    Args:
        data:      (N, D) array of data points
        centroids: (K, D) array of cluster centroids

    Returns:
        MSE scalar
    """
    N = len(data)
    distances = cdist(data, centroids, metric='euclidean')   # (N, K)
    assignments = np.argmin(distances, axis=1)               # (N,)
    total_error = sum(
        np.sum((data[assignments == k] - centroids[k]) ** 2)
        for k in range(len(centroids))
        if np.any(assignments == k)
    )
    return total_error / N


# ---------------------------------------------------------------------------
# K-Means single iteration
# ---------------------------------------------------------------------------

def kmeans_step(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    One K-Means iteration: assign points → recompute centroids.
    Returns updated centroids (same shape as input).
    """
    K = len(centroids)
    distances = cdist(data, centroids, metric='euclidean')
    assignments = np.argmin(distances, axis=1)
    new_centroids = np.copy(centroids)
    for k in range(K):
        members = data[assignments == k]
        if len(members) > 0:
            new_centroids[k] = members.mean(axis=0)
    return new_centroids


# ---------------------------------------------------------------------------
# CM2 — Enhanced Cluster Matching (the thesis contribution)
# ---------------------------------------------------------------------------

def cm2_match(particle_centroids: np.ndarray,
              global_best_centroids: np.ndarray) -> np.ndarray:
    """
    CM2: Rearrange the sequence of centroids in `particle_centroids` so
    that each centroid is paired with its closest centroid in
    `global_best_centroids` (minimum-distance greedy matching).

    This resolves the arbitrary-ordering problem that causes divergence
    during PSO velocity updates.

    Args:
        particle_centroids:     (K, D) centroids of current particle
        global_best_centroids:  (K, D) centroids of global best particle

    Returns:
        (K, D) reordered particle centroids
    """
    K = len(particle_centroids)
    dist_matrix = cdist(particle_centroids, global_best_centroids, metric='euclidean')  # (K, K)

    used_particle = set()
    used_global = set()
    pairs = []                        # (particle_idx, global_idx)

    # Greedy minimum-distance pairing — pick smallest available distance each round
    dist_copy = dist_matrix.copy()
    for _ in range(K):
        # Mask already-used rows and columns with inf
        for r in used_particle:
            dist_copy[r, :] = np.inf
        for c in used_global:
            dist_copy[:, c] = np.inf

        r, c = np.unravel_index(np.argmin(dist_copy), dist_copy.shape)
        pairs.append((r, c))
        used_particle.add(r)
        used_global.add(c)

    # Rearrange: position i in output = particle centroid matched to global centroid i
    reordered = np.zeros_like(particle_centroids)
    for p_idx, g_idx in pairs:
        reordered[g_idx] = particle_centroids[p_idx]

    return reordered


# ---------------------------------------------------------------------------
# PSO-KM(CM2)
# ---------------------------------------------------------------------------

class PSOKM_CM2:
    """
    PSO-based K-Means with Enhanced Cluster Matching (CM2).

    Usage:
        model = PSOKM_CM2(K=5, swarm_size=6, max_iter=40)
        centroids, mse = model.fit(data)
    """

    def __init__(
        self,
        K: int,
        swarm_size: int = 6,
        max_iter: int = 40,
        w: float = 0.7,
        c1: float = 0.2,
        c2: float = 0.2,
        adaptive_inertia: bool = False,
        w_start: float = 0.9,
        w_end: float = 0.4,
        seed: int = None
    ):
        """
        Args:
            K:                Number of clusters
            swarm_size:       Number of particles  (thesis Table 1: 6)
            max_iter:         Maximum iterations   (thesis Table 1: 40)
            w:                Inertia weight — used when adaptive_inertia=False (thesis Table 1: 0.7)
            c1:               Cognitive factor     (thesis Table 1: 0.2)
            c2:               Social factor        (thesis Table 1: 0.2)
            adaptive_inertia: If True, linearly decrease w from w_start to w_end each iteration
            w_start:          Initial inertia weight for adaptive mode (default 0.9)
            w_end:            Final inertia weight for adaptive mode   (default 0.4)
            seed:             Random seed
        """
        self.K = K
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.adaptive_inertia = adaptive_inertia
        self.w_start = w_start
        self.w_end = w_end
        self.rng = np.random.default_rng(seed)

        self.mse_history: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run PSO-KM(CM2) on data.

        Args:
            data: (N, D) array of data points

        Returns:
            best_centroids: (K, D)
            best_mse:       final MSE
        """
        N, D = data.shape
        K = self.K

        # -- Initialise particle positions randomly from data range --------
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)

        # positions[i] = (K, D) centroids for particle i
        positions = np.array([
            self.rng.uniform(data_min, data_max, size=(K, D))
            for _ in range(self.swarm_size)
        ])
        velocities = np.zeros_like(positions)

        # Personal bests
        pbest_positions = positions.copy()
        pbest_mse = np.array([compute_mse(data, pos) for pos in positions])

        # Global best
        gbest_idx = np.argmin(pbest_mse)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_mse = pbest_mse[gbest_idx]

        self.mse_history = [gbest_mse]

        # -- Main PSO loop --------------------------------------------------
        for iteration in range(self.max_iter):
            for i in range(self.swarm_size):

                # CM2: rearrange centroids to match global best ordering
                positions[i] = cm2_match(positions[i], gbest_position)

                # One K-Means exploitation step
                positions[i] = kmeans_step(data, positions[i])

                # Evaluate fitness
                mse = compute_mse(data, positions[i])

                # Update personal best
                if mse < pbest_mse[i]:
                    pbest_mse[i] = mse
                    pbest_positions[i] = positions[i].copy()

            # Update global best
            best_i = np.argmin(pbest_mse)
            if pbest_mse[best_i] < gbest_mse:
                gbest_mse = pbest_mse[best_i]
                gbest_position = pbest_positions[best_i].copy()

            self.mse_history.append(gbest_mse)

            # -- Velocity and position update (equations 2 & 3) -----------
            for i in range(self.swarm_size):
                r1 = self.rng.uniform(0, 1, size=positions[i].shape)
                r2 = self.rng.uniform(0, 1, size=positions[i].shape)

                # Adaptive inertia: linearly decrease w each iteration
                if self.adaptive_inertia:
                    current_w = self.w_start - (self.w_start - self.w_end) * (
                        iteration / max(self.max_iter - 1, 1)
                    )
                else:
                    current_w = self.w

                velocities[i] = (
                    current_w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )

                # Clamp velocity to data range
                v_max = (data_max - data_min) * 0.5
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], data_min, data_max)

        return gbest_position, gbest_mse

    # ------------------------------------------------------------------
    def get_assignments(self, data: np.ndarray,
                        centroids: np.ndarray) -> np.ndarray:
        """Assign each data point to nearest centroid."""
        distances = cdist(data, centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
