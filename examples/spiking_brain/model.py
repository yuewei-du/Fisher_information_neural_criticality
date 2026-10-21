"""
The spiking network used as the running example in this repo: ``N`` nodes,
each inhibitory (-g) with probability ``alpha`` and excitatory (+1) otherwise,
connected with probability ``p`` and scaled so that the largest eigenvalue of the
connectivity matrix equals ``lambda``. The state is binary and updated
synchronously at every time step.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

__all__ = ["NetworkParams", "generate_random_network", "evolve"]


@dataclass(frozen=True)
class NetworkParams:
    """
    Structural parameters of the random network.

    Args:
        N (int): number of nodes
        alpha (float): fraction of inhibitory nodes
        p (float): connection probability
        g (float): relative strength
    """
    N: int = 400
    alpha: float = 0.2
    p: float = 0.2
    g: float = 2.0

    @property
    def pext_ratio(self):
        """Probability of an external input."""
        return 1.0 / (self.N * 200)


def generate_random_network(rng, N, alpha=0.2, p=0.2, g=2.0):
    """
    Draw one random connectivity matrix, unscaled.

    Args:
        rng (Generator): numpy random generator
        N (int): number of nodes
        alpha (float): fraction of inhibitory nodes
        p (float): connection probability
        g (float): relative strength

    Returns:
        csr_matrix: the base network, whose largest eigenvalue the caller
            rescales to the target control parameter.
    """
    neuron_types = np.where(rng.random(N) > alpha, 1, -g)
    connection_mask = rng.random((N, N)) < p
    np.fill_diagonal(connection_mask, False)
    base_weights = rng.random((N, N)) * neuron_types[None, :]
    base_weights[~connection_mask] = 0
    return csr_matrix(base_weights)


@njit
def evolve(W_data, W_indices, W_indptr, rand_buffer, pext_ratio, T, N):
    """
    Time evolution of the binary state.

    Args:
        W_data, W_indices, W_indptr: the three CSR arrays of the rescaled
            connectivity matrix
        rand_buffer (ndarray): shape ``(T, N)`` of uniform draws
        pext_ratio (float): probability of an external input
        T (int): number of time steps
        N (int): number of nodes

    Returns:
        ndarray: the state at every time step, shape ``(T, N)``.
    """
    s = np.zeros((T, N), dtype=np.uint8)

    for i in range(N):
        if rand_buffer[0, i] < 0.5:
            s[0, i] = 1

    for t in range(1, T):
        for i in range(N):
            post_input = 0.0
            for j_idx in range(W_indptr[i], W_indptr[i + 1]):
                neighbor_idx = W_indices[j_idx]
                if s[t - 1, neighbor_idx] == 1:
                    post_input += W_data[j_idx]

            prob = post_input
            if prob < 0:
                prob = 0.0
            if prob > 1:
                prob = 1.0

            if rand_buffer[t, i] < prob or rand_buffer[t, i] < pext_ratio:
                s[t, i] = 1

    return s
