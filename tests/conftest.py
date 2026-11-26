import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification


@pytest.fixture
def diabetes_data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


@pytest.fixture
def breast_cancer_data():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 4
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([0.5, -0.3, 0.2, 0.1])
    true_bias = 0.1
    mu = np.exp(X @ true_weights + true_bias)
    y = rng.poisson(mu)
    return X, y, true_weights, true_bias


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(7)
    n_samples, n_features = 200, 5
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    true_bias = 0.25
    y = X @ true_weights + true_bias + rng.normal(scale=0.1, size=n_samples)
    return X, y, true_weights, true_bias


@pytest.fixture
def logistic_data():
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        flip_y=0.15,
        class_sep=0.7,
        random_state=11,
    )
    return X, y


@pytest.fixture
def binary_features():
    rng = np.random.default_rng(13)
    X = rng.integers(0, 2, size=(300, 4))
    y = X[:, 0] ^ X[:, 1]
    return X, y


@pytest.fixture
def count_features():
    rng = np.random.default_rng(17)
    X = rng.poisson(2.0, size=(300, 4))
    y = (X[:, 0] > X[:, 1]).astype(int)
    return X, y
