"""
Tests for WaterML_on.models.families

ASSUMPTION: families live at WaterML_on.models.families. If yours are
elsewhere (e.g. a top-level WaterML_on.families module), just fix the
import line below — nothing else changes.
"""
import numpy as np
import pytest

from WaterML_on.models.families import Gaussian, Binomial, Poisson, Gamma, InverseGaussian


@pytest.mark.parametrize("family_class,mu,expected_variance", [
    (Gaussian, np.array([0.5, 2.0, 10.0]), np.array([1.0, 1.0, 1.0])),
    (Binomial, np.array([0.5]), np.array([0.25])),          # p(1-p) at p=0.5
    (Binomial, np.array([0.1]), np.array([0.09])),           # 0.1 * 0.9
    (Poisson, np.array([3.0, 7.5]), np.array([3.0, 7.5])),   # variance == mean
    (Gamma, np.array([2.0]), np.array([4.0])),                # mu^2
    (InverseGaussian, np.array([2.0]), np.array([8.0])),       # mu^3
])
def test_variance_formulas(family_class, mu, expected_variance):
    family = family_class()
    result = family.variance(mu)
    assert np.allclose(result, expected_variance)


@pytest.mark.parametrize("family_class", [Gaussian, Binomial, Poisson, Gamma, InverseGaussian])
def test_link_inverse_link_are_true_inverses(family_class):
    """
    link(inverse_link(eta)) should return eta (up to clipping at extreme
    values). This is the property that actually matters for a GLM — if
    these aren't inverses of each other, predict() and the solvers will
    silently disagree with each other's math.
    """
    family = family_class()
    eta = np.array([0.5, 1.0, 2.0])  # moderate values, away from clipping edges
    mu = family.inverse_link(eta)
    eta_roundtrip = family.link(mu)
    assert np.allclose(eta, eta_roundtrip, atol=1e-6)


def test_binomial_inverse_link_matches_sigmoid():
    binomial = Binomial()
    eta = np.array([-2.0, 0.0, 2.0])
    expected = 1 / (1 + np.exp(-eta))
    assert np.allclose(binomial.inverse_link(eta), expected)


def test_poisson_inverse_link_matches_exp():
    poisson = Poisson()
    eta = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(poisson.inverse_link(eta), np.exp(eta))


def test_binomial_link_handles_boundary_values_without_error():
    """Regression test for the clipping you added around log(p/(1-p))."""
    binomial = Binomial()
    mu = np.array([0.0, 1.0, 0.5])
    result = binomial.link(mu)  # should not raise, should not be inf/nan
    assert np.all(np.isfinite(result))
