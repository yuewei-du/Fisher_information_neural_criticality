import logging

import numpy as np
import pytest

from fim import FisherInfoCalculator, compute_fim, compute_fim_metrics
from fim.estimator import (
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_TOLERANCE,
    _per_parameter_samples,
    _sample_axis,
)


# ---- input layouts ---------------------------------------------------------

def test_sample_axis_standard_and_transposed():
    assert _sample_axis(np.zeros((100, 5)), 5) == 1
    assert _sample_axis(np.zeros((5, 100)), 5) == 0


def test_sample_axis_square_numeric_is_treated_as_standard():
    assert _sample_axis(np.zeros((5, 5)), 5) == 1


def test_sample_axis_mismatch_raises():
    with pytest.raises(ValueError, match="neither axis"):
        _sample_axis(np.zeros((100, 4)), 5)


def test_per_parameter_samples_2d_layouts_agree():
    data = np.arange(30.0).reshape(10, 3)  # (n_samples, n_param)
    standard = _per_parameter_samples(data, 3)
    transposed = _per_parameter_samples(data.T, 3)
    assert len(standard) == len(transposed) == 3
    for a, b, j in zip(standard, transposed, range(3)):
        np.testing.assert_array_equal(a, data[:, j])
        np.testing.assert_array_equal(b, data[:, j])


def test_per_parameter_samples_ragged_inputs():
    # Pipeline B's digitisation returns bins with unequal sample counts
    ragged = [np.ones(4), np.ones(7), np.ones(5)]
    as_object = np.empty(3, dtype=object)
    as_object[:] = ragged
    for data in (ragged, tuple(ragged), as_object):
        entries = _per_parameter_samples(data, 3)
        assert [len(e) for e in entries] == [4, 7, 5]


def test_per_parameter_samples_wrong_count_raises():
    with pytest.raises(ValueError, match="expected 4"):
        _per_parameter_samples([np.ones(3)] * 3, 4)


def test_per_parameter_samples_3d_raises():
    with pytest.raises(ValueError, match="3D"):
        _per_parameter_samples(np.zeros((2, 3, 4)), 3)


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        FisherInfoCalculator(G=50, alpha=1, N_rep=1, N_sim=10,
                             calc_method="sigma", direction="sideways")

    calculator = FisherInfoCalculator(G=50, alpha=1, N_rep=1, N_sim=10,
                                      calc_method="sigma")
    with pytest.raises(ValueError, match="direction"):
        calculator.calculate_fisher_info(np.zeros((10, 3)), 1, 1, 0.1,
                                         method="sigma", direction="sideways")


@pytest.mark.parametrize("calc_method", ["branchingratio", "central", ""])
def test_unknown_calc_method_raises(calc_method):
    # the pre-rename "branchingratio" used to fall back to central silently
    with pytest.raises(ValueError, match="calc_method"):
        compute_fim(np.zeros((10, 3)), np.arange(3.0), [1],
                    calc_method=calc_method)


def test_importing_fim_does_not_configure_logging():
    assert logging.getLogger("fim.estimator").handlers == []
    assert logging.getLogger("fim.estimator").level == logging.NOTSET


# ---- summary metrics -------------------------------------------------------

def test_compute_fim_metrics_gaussian_curve():
    width, height, center = 0.02, 7.0, 1.0
    sigma = np.linspace(0.8, 1.2, 4001)
    curve = height * np.exp(-0.5 * ((sigma - center) / width) ** 2)

    peak_height, peak_width = compute_fim_metrics(sigma, curve, center=center)

    assert peak_height == pytest.approx(height)
    assert peak_width == pytest.approx(width, rel=1e-3)


def test_compute_fim_metrics_width_is_measured_about_center():
    sigma = np.array([0.9, 1.0, 1.1])
    fim = np.array([0.0, 1.0, 0.0])
    _, width_at_peak = compute_fim_metrics(sigma, fim, center=1.0)
    _, width_off_peak = compute_fim_metrics(sigma, fim, center=0.9)
    assert width_at_peak == pytest.approx(0.0)
    assert width_off_peak == pytest.approx(0.1)


# ---- end to end ------------------------------------------------------------
#
# Tolerances reflect the estimator's measured precision on this problem.
# Over 8 independent data sets (10^4 samples per parameter value), the
# central scheme averaged 4.32 +/- 0.41 and the forward/backward average
# 4.96 +/- 0.53 against the true 4.0: sampling noise in the PDF differences
# biases the FI upwards, more so for one-sided differences. The bounds below
# still catch gross errors such as a constant-factor bug in a difference
# scheme.

LOCATION_SIGMA = 0.5        # N(theta, sigma^2) has FI = 1 / sigma^2 = 4
N_SAMPLES = 10000


@pytest.fixture(scope="module")
def gaussian_location_data():
    rng = np.random.default_rng(0)
    theta = np.linspace(0.0, 1.0, 21)
    data = np.column_stack([rng.normal(t, LOCATION_SIGMA, N_SAMPLES)
                            for t in theta])
    return data, theta


def run_seeded(data, theta, index_range, **kwargs):
    kwargs = {"n_jobs": 1, "seed": 0, **kwargs}
    return compute_fim(data, theta, index_range, G=100, alpha=1, N_rep=2,
                       N_sim=N_SAMPLES, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("logarithmic", [False, True])
def test_compute_fim_central_recovers_gaussian_fi(gaussian_location_data,
                                                  logarithmic):
    data, theta = gaussian_location_data
    index_range = np.array([5, 10, 15])

    fi_mean, fi_low, fi_up, epsilon = run_seeded(
        data, theta, index_range, logarithmic=logarithmic,
    )

    assert fi_mean.shape == fi_low.shape == fi_up.shape == epsilon.shape == (3,)
    assert np.mean(fi_mean) == pytest.approx(1 / LOCATION_SIGMA ** 2, rel=0.25)
    assert np.all(fi_low <= fi_mean) and np.all(fi_mean <= fi_up)
    # the step search should land epsilon in its tolerance band
    assert np.all(np.abs(epsilon - DEFAULT_EPSILON)
                  <= DEFAULT_EPSILON_TOLERANCE + 1e-3)


@pytest.mark.slow
def test_compute_fim_one_sided_recovers_gaussian_fi(gaussian_location_data):
    # Pipeline B's setting: forward and backward, then their average
    data, theta = gaussian_location_data
    index_range = np.array([5, 10, 15])

    estimates = [
        run_seeded(data, theta, index_range, calc_method="sigma",
                   direction=direction, logarithmic=True)[0]
        for direction in ("forward", "backward")
    ]

    assert np.mean(estimates) == pytest.approx(1 / LOCATION_SIGMA ** 2, rel=0.4)


@pytest.mark.slow
def test_seed_makes_bootstrap_reproducible_for_any_n_jobs(
        gaussian_location_data):
    data, theta = gaussian_location_data
    serial = run_seeded(data, theta, [10], n_jobs=1)
    parallel = run_seeded(data, theta, [10], n_jobs=2)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_different_seeds_give_different_resamples(gaussian_location_data):
    data, theta = gaussian_location_data
    first = run_seeded(data, theta, [10], seed=0)
    second = run_seeded(data, theta, [10], seed=1)
    assert not np.array_equal(first[0], second[0])
