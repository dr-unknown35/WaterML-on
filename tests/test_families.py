import numpy as np
import pytest

from WaterML_on.models.families import (
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    Multinomial,
    Poisson,
)


@pytest.mark.parametrize(
    "family_class,mu,expected_variance",
    [
        (Gaussian, np.array([0.5, 2.0, 10.0]), np.array([1.0, 1.0, 1.0])),
        (Binomial, np.array([0.5]), np.array([0.25])),
        (Binomial, np.array([0.1]), np.array([0.09])),
        (Poisson, np.array([3.0, 7.5]), np.array([3.0, 7.5])),
        (Gamma, np.array([2.0]), np.array([4.0])),
        (InverseGaussian, np.array([2.0]), np.array([8.0])),
    ],
)
def test_variance_formulas(family_class, mu, expected_variance):
    assert np.allclose(family_class().variance(mu), expected_variance)


@pytest.mark.parametrize(
    "family_class", [Gaussian, Binomial, Poisson, Gamma, InverseGaussian]
)
def test_link_inverse_link_are_true_inverses(family_class):
    eta = np.array([0.5, 1.0, 2.0])
    roundtrip = family_class().link(family_class().inverse_link(eta))

    assert np.allclose(eta, roundtrip, atol=1e-6)


def test_binomial_inverse_link_matches_sigmoid():
    eta = np.array([-2.0, 0.0, 2.0])
    expected = 1 / (1 + np.exp(-eta))

    assert np.allclose(Binomial().inverse_link(eta), expected)


def test_poisson_inverse_link_matches_exp():
    eta = np.array([-1.0, 0.0, 1.0])

    assert np.allclose(Poisson().inverse_link(eta), np.exp(eta))


def test_binomial_link_handles_boundary_values_without_error():
    result = Binomial().link(np.array([0.0, 1.0, 0.5]))

    assert np.all(np.isfinite(result))


def test_exponential_families_clip_extreme_linear_predictors():
    eta = np.array([-1000.0, 1000.0])

    for family in (Binomial, Poisson, Gamma, InverseGaussian):
        result = family().inverse_link(eta)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


def test_multinomial_inverse_link_matches_stable_softmax():
    rng = np.random.default_rng(3)
    eta = rng.normal(size=(12, 4))
    shifted = eta - eta.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    actual = Multinomial().inverse_link(eta)

    assert np.allclose(actual, expected)
    assert np.allclose(actual.sum(axis=1), 1)


def test_multinomial_link_inverse_link_roundtrip_is_centered():
    rng = np.random.default_rng(4)
    eta = rng.normal(size=(10, 3))
    mu = Multinomial().inverse_link(eta)

    roundtrip = Multinomial().link(mu)

    assert np.allclose(roundtrip, eta - eta[:, [-1]])


def test_multinomial_variance_matches_categorical_covariance():
    mu_1d = np.array([0.2, 0.5, 0.3])
    expected_1d = np.diag(mu_1d) - np.outer(mu_1d, mu_1d)

    assert np.allclose(Multinomial().variance(mu_1d), expected_1d)

    rng = np.random.default_rng(5)
    mu_2d = rng.dirichlet(np.ones(3), size=8)
    actual_2d = Multinomial().variance(mu_2d)

    assert actual_2d.shape == (8, 3, 3)
    for row_idx in range(mu_2d.shape[0]):
        expected = np.diag(mu_2d[row_idx]) - np.outer(mu_2d[row_idx], mu_2d[row_idx])
        assert np.allclose(actual_2d[row_idx], expected)
