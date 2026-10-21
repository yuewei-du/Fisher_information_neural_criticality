"""
Simulation for the spiking network example: ``simulate_para_lambda``
produces one scalar observable per trial at a fixed control parameter, and
``simulate_para_sigma`` produces the activity and branching-ratio time series.
"""

import gc
import logging
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import ArpackNoConvergence, eigs
from tqdm.auto import tqdm
from .model import NetworkParams, evolve, generate_random_network

logger = logging.getLogger(__name__)

__all__ = ["simulate_para_lambda", "simulate_para_sigma"]

COMPLEX_TOL = 1e-8

_MAX_EXTRA_FACTOR = 10


def _dominant_eigenvalue(sparse_base, trial_id):
    """
    Largest eigenvalue of the connectivity matrix.

    Args:
        sparse_base (csr_matrix): the unscaled network
        trial_id (int): trial index, for logging

    Returns:
        complex or None: the eigenvalue, or ``None`` if ARPACK did not converge.
    """
    try:
        values, _ = eigs(sparse_base, k=1, which="LM", maxiter=10000, tol=1e-6)
    except ArpackNoConvergence:
        logger.warning("Trial %s: eigs did not converge, skipping.", trial_id)
        return None
    return complex(values[0])


def _is_complex(value):
    """Whether the eigenvalue is a genuine complex pair, not float noise."""
    return abs(value.imag) > COMPLEX_TOL * max(abs(value), 1.0)


def _lambda_single_trial(lambda_target, params, T, transient_cutoff, trial_id,
                         testing):
    """
    One realisation at a fixed lambda.

    Args:
        lambda_target (float): the spectral radius to rescale the network to
        params (NetworkParams): structural parameters of the network
        T (int): number of time steps
        transient_cutoff (int): steps discarded as transients
        trial_id (int): trial index
        testing (bool): return the cumulative mean activity at 25/50/75/100% of
            the recorded window instead of the mean activity

    Returns:
        float or ndarray or None: the time-averaged firing rate.
    """
    rng = np.random.default_rng(trial_id)

    sparse_base = generate_random_network(rng, params.N, params.alpha,params.p, params.g)


    eigenvalue = _dominant_eigenvalue(sparse_base, trial_id)
    if eigenvalue is None:
        return None
    if _is_complex(eigenvalue):
        logger.info(
            "Trial %s: dominant eigenvalue %.6g%+.6gj is complex, skipping.",
            trial_id, eigenvalue.real, eigenvalue.imag,
        )
        return None
    base_lambda = abs(eigenvalue)

    w_scale = lambda_target / base_lambda if base_lambda > 1e-10 else 0
    adj = sparse_base.multiply(w_scale).tocsr()

    rand_buffer = rng.random((T, params.N))
    s = evolve(adj.data, adj.indices, adj.indptr, rand_buffer,
               params.pext_ratio, T, params.N)
    s = s[transient_cutoff:, :]

    if testing:
        t_eff = s.shape[0]
        return np.array([s[: (t_eff // 4) * i, :].mean() for i in range(1, 5)])

    return s.mean()


def simulate_para_lambda(lambda_target, params=None, T=10000, n_trials=10000,
                         transient_cutoff=2000, testing=False, n_jobs=-2):
    """
    Collect ``n_trials`` realisations at a fixed lambda.

    Args:
        lambda_target (float): the spectral radius to rescale the network to
        params (NetworkParams or None): structural parameters of the network
        T (int): number of time steps per realisation
        n_trials (int): number of valid realisations to collect
        transient_cutoff (int): steps discarded as transients
        testing (bool): see :func:`_lambda_single_trial`
        n_jobs (int): joblib worker count

    Returns:
        ndarray: one time-averaged firing rate per kept realisation, shape
            ``(n_trials,)``.
    """
    params = params or NetworkParams()

    kept = []
    trial_id = 0
    n_dropped = 0
    trial_budget = _MAX_EXTRA_FACTOR * n_trials

    with tqdm(total=n_trials, desc=f"lambda={lambda_target:.3f}",
              unit="trial") as bar:
        while len(kept) < n_trials and trial_id < trial_budget:
            n_batch = min(n_trials - len(kept), trial_budget - trial_id)
            first_id = trial_id
            trial_id += n_batch

            for result in Parallel(n_jobs=n_jobs,
                                   return_as="generator_unordered")(
                delayed(_lambda_single_trial)(lambda_target, params, T,
                                              transient_cutoff,
                                              first_id + offset, testing)
                for offset in range(n_batch)
            ):
                if result is None:
                    n_dropped += 1
                else:
                    kept.append(result)
                    bar.update(1)
                bar.set_postfix(attempted=trial_id, dropped=n_dropped)

    if n_dropped:
        logger.warning(
            f"lambda={lambda_target:.3f}: {n_dropped} of {trial_id} realisations dropped (complex or non-convergent dominant eigenvalue)")
    if len(kept) < n_trials:
        raise RuntimeError(
            f"only {len(kept)} of the requested {n_trials} realisations at "
            f"lambda={lambda_target:.3f} survived after {trial_id} attempts."
        )

    return np.array(kept)


def _sigma_single_trial(lambda_target, params, T, transient_cutoff, trial_id):
    """
    One realisation at a fixed lambda, as time series.

    Args:
        lambda_target (float): the spectral radius to rescale the network to
        params (NetworkParams): structural parameters of the network
        T (int): number of time steps
        transient_cutoff (int): steps discarded as transients
        trial_id (int): trial index, which also seeds the generator

    Returns:
        dict or None: ``{"S_t": activity, "sigma": branching ratio}``, or
            ``None`` if the network was dropped.
    """
    rng = np.random.default_rng(trial_id)
    sparse_base = generate_random_network(rng, params.N, params.alpha,params.p, params.g)

    eigenvalue = _dominant_eigenvalue(sparse_base, trial_id)
    if eigenvalue is None:
        return None
    if _is_complex(eigenvalue):
        logger.info(
            "Trial %s: dominant eigenvalue %.6g%+.6gj is complex, skipping.",
            trial_id, eigenvalue.real, eigenvalue.imag,
        )
        return None
    base_lambda = abs(eigenvalue)

    w_scale = lambda_target / base_lambda if base_lambda > 1e-10 else 0
    adj = sparse_base.multiply(w_scale).tocsr()

    rand_buffer = rng.random((T, params.N))
    s = evolve(adj.data, adj.indices, adj.indptr, rand_buffer,
               params.pext_ratio, T, params.N)

    S_t = s.mean(axis=1)
    del s
    gc.collect()

    valid_idx = np.arange(transient_cutoff, T - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            S_t[valid_idx] > 0,
            S_t[valid_idx + 1] / S_t[valid_idx],
            np.nan,
        )

    return {"S_t": S_t[valid_idx], "sigma": sigma}


def simulate_para_sigma(lambda_target, params=None, T=10000, n_trials=2000,
                        transient_cutoff=2000, n_jobs=-2):
    """
    Collect ``n_trials`` realisations at a fixed lambda and return
    their activity and branching-ratio time series.

    Args:
        lambda_target (float): the spectral radius to rescale the network to
        params (NetworkParams or None): structural parameters of the network
        T (int): number of time steps per realisation
        n_trials (int): number of valid realisations to collect
        transient_cutoff (int): steps discarded before the series starts
        n_jobs (int): joblib worker count

    Returns:
        (S_t, sigma): two ndarrays of shape
            ``(n_trials * n_valid_steps,)``, concatenated across trials.
    """
    params = params or NetworkParams()

    kept = []
    trial_id = 0
    n_dropped = 0
    trial_budget = _MAX_EXTRA_FACTOR * n_trials

    with tqdm(total=n_trials, desc=f"lambda={lambda_target:.3f}",
              unit="trial") as bar:
        while len(kept) < n_trials and trial_id < trial_budget:
            n_batch = min(n_trials - len(kept), trial_budget - trial_id)
            first_id = trial_id
            trial_id += n_batch

            for result in Parallel(n_jobs=n_jobs,
                                   return_as="generator_unordered")(
                delayed(_sigma_single_trial)(lambda_target, params, T,
                                             transient_cutoff,
                                             first_id + offset)
                for offset in range(n_batch)
            ):
                if result is None:
                    n_dropped += 1
                else:
                    kept.append(result)
                    bar.update(1)
                bar.set_postfix(attempted=trial_id, dropped=n_dropped)

    if n_dropped:
        logger.warning(
            f"lambda={lambda_target:.3f}: {n_dropped} of {trial_id} realisations dropped (complex or non-convergent dominant eigenvalue)")
    if len(kept) < n_trials:
        raise RuntimeError(
            f"only {len(kept)} of the requested {n_trials} realisations at "
            f"lambda={lambda_target:.3f} survived after {trial_id} attempts."
        )

    S_t = np.concatenate([r["S_t"] for r in kept])
    sigma = np.concatenate([r["sigma"] for r in kept])

    logger.info("S_t shape: %s, sigma shape: %s", S_t.shape,
                sigma.shape)

    return S_t, sigma
