import numpy as np
from sklearn.linear_model import LinearRegression as SKLinearRegression

from WaterML_on.models.Linear_models import LinearRegression
from WaterML_on.optim.gradient_methods import NormalEquations


def test_normal_equations_matches_sklearn_on_diabetes(diabetes_data):
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations()).fit(X, y)
    sk_model = SKLinearRegression().fit(X, y)

    assert np.allclose(model.weights, sk_model.coef_, atol=1e-6)
    assert np.isclose(model.bias, sk_model.intercept_, atol=1e-6)


def test_default_linear_regression_matches_sklearn_on_synthetic_data(linear_data):
    X, y, _, _ = linear_data
    model = LinearRegression().fit(X, y)
    sk_model = SKLinearRegression().fit(X, y)

    assert np.allclose(model.weights, sk_model.coef_, atol=1e-8)
    assert np.isclose(model.bias, sk_model.intercept_, atol=1e-8)


def test_linear_regression_predictions_match_sklearn(linear_data):
    X, y, _, _ = linear_data
    model = LinearRegression().fit(X, y)
    sk_model = SKLinearRegression().fit(X, y)

    assert np.allclose(model.predict(X), sk_model.predict(X))
