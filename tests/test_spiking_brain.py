import numpy as np
import pytest
from scipy.sparse import csr_matrix

from examples.spiking_brain.model import (
    NetworkParams,
    evolve,
    generate_random_network,
)
from examples.spiking_brain.simulate import (
    _dominant_eigenvalue,
    _is_complex,
    simulate_para_lambda,
    simulate_para_sigma,
)


def run_evolve(W, rand_buffer, pext_ratio):
    W = csr_matrix(W)
    T, N = rand_buffer.shape
    return evolve(W.data, W.indices, W.indptr, rand_buffer, pext_ratio, T, N)


def test_evolve_without_coupling_fires_only_from_external_input():
    rng = np.random.default_rng(0)
    rand_buffer = rng.random((200, 10))

    s = run_evolve(np.zeros((10, 10)), rand_buffer, pext_ratio=0.3)

    np.testing.assert_array_equal(s[0], rand_buffer[0] < 0.5)
    np.testing.assert_array_equal(s[1:], rand_buffer[1:] < 0.3)


def test_evolve_propagates_along_excitatory_edge():
    # W[i, j] is the weight from j onto i: unit 0 copies unit 1 with weight 1
    W = np.array([[0.0, 1.0], [0.0, 0.0]])
    rand_buffer = np.random.default_rng(1).random((100, 2))

    s = run_evolve(W, rand_buffer, pext_ratio=0.0)

    np.testing.assert_array_equal(s[1:, 0], s[:-1, 1])


def test_evolve_inhibition_is_clipped_to_zero():
    W = np.array([[0.0, -5.0], [0.0, 0.0]])
    rand_buffer = np.zeros((10, 2))  # every draw would fire if prob > 0

    s = run_evolve(W, rand_buffer, pext_ratio=0.0)

    assert s[0, 1] == 1
    assert s[1, 0] == 0


def test_generate_random_network_structure():
    N, alpha, p, g = 1000, 0.2, 0.2, 2.0
    W = generate_random_network(np.random.default_rng(0), N, alpha, p, g)
    dense = W.toarray()

    assert np.all(np.diag(dense) == 0)
    assert np.count_nonzero(dense) / (N * (N - 1)) == pytest.approx(p, abs=0.01)

    # columns are presynaptic units: each has a single sign
    inhibitory = (dense < 0).any(axis=0)
    excitatory = (dense > 0).any(axis=0)
    assert not np.any(inhibitory & excitatory)
    assert inhibitory.mean() == pytest.approx(alpha, abs=0.05)
    assert dense.min() >= -g and dense.max() < 1


def test_rescaling_sets_spectral_radius():
    lambda_target = 1.0
    for seed in range(5):
        base = generate_random_network(np.random.default_rng(seed), 200)
        eigenvalue = _dominant_eigenvalue(base, seed)
        if eigenvalue is None or _is_complex(eigenvalue):
            continue
        scaled = base.multiply(lambda_target / abs(eigenvalue)).toarray()
        radius = np.max(np.abs(np.linalg.eigvals(scaled)))
        assert radius == pytest.approx(lambda_target, rel=1e-6)
        return
    pytest.fail("no seed gave a real dominant eigenvalue")


def test_simulate_para_lambda_smoke():
    rates = simulate_para_lambda(1.0, NetworkParams(N=50), T=300, n_trials=3,
                                 transient_cutoff=100, n_jobs=1)
    assert rates.shape == (3,)
    assert np.all((rates >= 0) & (rates <= 1))


def test_simulate_para_sigma_is_ratio_of_consecutive_activity():
    T, cutoff, n_trials = 300, 100, 2
    S_t, sigma = simulate_para_sigma(1.0, NetworkParams(N=50), T=T,
                                     n_trials=n_trials,
                                     transient_cutoff=cutoff, n_jobs=1)

    per_trial = T - 1 - cutoff
    assert S_t.shape == sigma.shape == (n_trials * per_trial,)
    assert np.all(np.isnan(sigma[S_t == 0]))

    for trial in range(n_trials):
        S = S_t[trial * per_trial:(trial + 1) * per_trial]
        sig = sigma[trial * per_trial:(trial + 1) * per_trial]
        nonzero = S[:-1] > 0
        np.testing.assert_allclose(sig[:-1][nonzero],
                                   S[1:][nonzero] / S[:-1][nonzero])
