"""
Finite-difference FIM on exact densities, where the Fisher information is
known in closed form. No density estimation is involved, so these tests pin
down the difference schemes themselves.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import expon, norm

from fim.npfi import get_bbox, get_pdfs_from_data, npfi


def gaussian_location(sigma):
    """p(x | mu) = N(mu, sigma^2); FI(mu) = 1 / sigma^2."""
    return lambda mu: (lambda x: np.asarray(norm.pdf(x, mu, sigma)))


def gaussian_scale(theta):
    """p(x | theta) = N(0, theta^2); FI(theta) = 2 / theta^2."""
    return lambda x: np.asarray(norm.pdf(x, 0.0, theta))


def exponential_rate(rate):
    """p(x | rate) = rate * exp(-rate * x); FI(rate) = 1 / rate^2."""
    return lambda x: np.asarray(expon.pdf(x, scale=1.0 / rate))


def fi_at(family, theta, dt, bounds, direction, logarithmic):
    if direction == "central":
        pdfs = [family(theta), family(theta - dt), family(theta + dt)]
    elif direction == "forward":
        pdfs = [family(theta), family(theta + dt)]
    else:
        pdfs = [family(theta), family(theta - dt)]
    fi, _, _ = npfi(pdfs, dt, bounds=bounds, logarithmic=logarithmic,
                    direction=direction)
    return fi


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("direction", ["central", "forward", "backward"])
def test_gaussian_location(direction, logarithmic):
    sigma = 0.5
    fi = fi_at(gaussian_location(sigma), 0.0, 1e-3, (-6, 6),
               direction, logarithmic)
    assert fi == pytest.approx(1 / sigma ** 2, rel=1e-3)


@pytest.mark.parametrize("logarithmic", [False, True])
def test_gaussian_scale_central(logarithmic):
    theta = 1.5
    fi = fi_at(gaussian_scale, theta, 1e-3, (-15, 15), "central", logarithmic)
    assert fi == pytest.approx(2 / theta ** 2, rel=1e-3)


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_gaussian_scale_one_sided(direction, logarithmic):
    # one-sided differences carry an O(dt) bias, hence the looser tolerance
    theta = 1.5
    fi = fi_at(gaussian_scale, theta, 1e-3, (-15, 15), direction, logarithmic)
    assert fi == pytest.approx(2 / theta ** 2, rel=1e-2)


@pytest.mark.parametrize("logarithmic", [False, True])
def test_one_sided_average_cancels_first_order_bias(logarithmic):
    # Pipeline B averages forward and backward: the O(dt) terms cancel
    theta, dt = 1.5, 1e-2
    forward = fi_at(gaussian_scale, theta, dt, (-15, 15), "forward", logarithmic)
    backward = fi_at(gaussian_scale, theta, dt, (-15, 15), "backward", logarithmic)
    assert (forward + backward) / 2 == pytest.approx(2 / theta ** 2, rel=1e-3)


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("direction", ["central", "forward", "backward"])
def test_exponential_rate(direction, logarithmic):
    rate = 2.0
    tol = 1e-3 if direction == "central" else 1e-2
    fi = fi_at(exponential_rate, rate, 1e-3, (0, 40 / rate),
               direction, logarithmic)
    assert fi == pytest.approx(1 / rate ** 2, rel=tol)


@pytest.mark.parametrize("direction", ["central", "forward"])
def test_epsilon_formula(direction):
    family, dt, N = gaussian_location(0.5), 1e-2, 10000
    pdfs = ([family(0), family(-dt), family(dt)] if direction == "central"
            else [family(0), family(dt)])
    fi, _, epsilon = npfi(pdfs, dt, bounds=(-6, 6), N=N, direction=direction)
    assert epsilon == pytest.approx(np.sqrt(2.0 / (N * fi * dt ** 2)))


def test_epsilon_is_none_without_N():
    family = gaussian_location(0.5)
    _, _, epsilon = npfi([family(0), family(-0.1), family(0.1)], 0.1,
                         bounds=(-6, 6))
    assert epsilon is None


def test_invalid_direction_raises():
    family = gaussian_location(0.5)
    with pytest.raises(ValueError, match="direction"):
        npfi([family(0), family(0.1)], 0.1, direction="sideways")


@pytest.mark.parametrize("direction, n_pdfs", [
    ("forward", 3), ("backward", 1), ("central", 2), ("central", 4),
])
def test_wrong_number_of_pdfs_raises(direction, n_pdfs):
    family = gaussian_location(0.5)
    with pytest.raises(ValueError):
        npfi([family(0)] * n_pdfs, 0.1, direction=direction)


def test_get_bbox_pads_by_factor():
    assert get_bbox([np.array([0.0, 1.0]), np.array([2.0])],
                    factor=0.5) == (-1.0, 3.0)
    assert get_bbox(np.array([0.0, 4.0]), multi_dim=False,
                    factor=0.25) == (-1.0, 5.0)


def test_deft_density_is_normalised_and_close_to_truth():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, 20000)
    pdfs, bbox = get_pdfs_from_data([samples], G=100, alpha=1)
    pdf = pdfs[0]

    total, _ = quad(lambda x: pdf(x).item(), *bbox, limit=200)
    assert total == pytest.approx(1.0, abs=1e-3)

    x = np.linspace(-2, 2, 21)
    estimate = np.array([pdf(xi).item() for xi in x])
    np.testing.assert_allclose(estimate, norm.pdf(x), atol=0.03)
