"""
Shared fixtures for the WaterML_on test suite.
Drop this file directly in your tests/ folder — pytest auto-discovers
conftest.py and makes its fixtures available to every test file in the
same directory, no imports needed.
"""
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer


@pytest.fixture
def diabetes_data():
    """Regression dataset — Gaussian family territory."""
    X, y = load_diabetes(return_X_y=True)
    return X, y


@pytest.fixture
def breast_cancer_data():
    """Binary classification dataset — Binomial family territory."""
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


@pytest.fixture
def poisson_data():
    """
    Synthetic count data with a known ground-truth coefficient vector.
    Useful because there's no clean sklearn toy dataset for Poisson —
    generating it ourselves lets us check whether the model recovers
    something close to the true weights, which is a stronger test than
    just "did it run".
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 4
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([0.5, -0.3, 0.2, 0.1])
    true_bias = 0.1
    mu = np.exp(X @ true_weights + true_bias)
    y = rng.poisson(mu)
    return X, y, true_weights, true_bias
