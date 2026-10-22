import numpy as np
import pytest

from fim import digitize_into_bins_adaptive

N_SIM = 1000


@pytest.fixture(scope="module")
def uniform_cloud():
    # parameter uniform on [0.9, 1.1]: 10^6 samples per unit, so the
    # narrowest feasible bin is about N_SIM / 10^6 = 0.001 wide
    rng = np.random.default_rng(0)
    para = rng.uniform(0.9, 1.1, 200000)
    variable = 10 * para  # lets each sample be traced back to its bin
    return para, variable


def digitize(para, variable, **kwargs):
    kwargs = {"initial_bin_width": 0.01, "N_sim": N_SIM, "center": 1.0,
              "bins_per_side": 5, "max_bin": 20, **kwargs}
    return digitize_into_bins_adaptive(para, variable, **kwargs)


def test_every_bin_is_full_and_holds_its_own_samples(uniform_cloud):
    samples, centers, width = digitize(*uniform_cloud)

    assert len(samples) == len(centers) > 0
    for bin_samples, center in zip(samples, centers):
        assert len(bin_samples) >= N_SIM
        para = bin_samples / 10
        assert np.all(para >= center - width / 2 - 1e-12)
        assert np.all(para < center + width / 2 + 1e-12)


def test_bins_are_contiguous_and_include_center(uniform_cloud):
    _, centers, width = digitize(*uniform_cloud)

    np.testing.assert_allclose(np.diff(centers), width)
    assert np.min(np.abs(centers - 1.0)) == pytest.approx(0.0, abs=1e-12)


def test_bin_width_is_close_to_narrowest_feasible(uniform_cloud):
    _, _, width = digitize(*uniform_cloud)
    assert 0.001 <= width <= 0.0013


def test_max_bin_limits_the_span(uniform_cloud):
    _, centers, _ = digitize(*uniform_cloud, max_bin=3)
    assert len(centers) == 7


def test_nan_and_inf_parameters_are_dropped(uniform_cloud):
    para, variable = uniform_cloud
    dirty_para = np.concatenate([para, [np.nan, np.inf, -np.inf]])
    dirty_variable = np.concatenate([variable, [1.0, 2.0, 3.0]])

    clean = digitize(para, variable)
    dirty = digitize(dirty_para, dirty_variable)

    assert dirty[2] == clean[2]
    np.testing.assert_array_equal(dirty[1], clean[1])
    for a, b in zip(dirty[0], clean[0]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kwargs", [
    {"initial_bin_width": 1e-4},     # too narrow to hold N_SIM samples
    {"center": 2.0},                 # outside the data
])
def test_infeasible_returns_empty(uniform_cloud, kwargs):
    samples, centers, _ = digitize(*uniform_cloud, **kwargs)
    assert len(samples) == 0 and len(centers) == 0
