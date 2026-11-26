import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from WaterML_on.data_utils.scaler import StandardScaler
from WaterML_on.models.Linear_models import LinearRegression, LogisticRegression
from WaterML_on.optim.gradient_methods import GradientDescent, NewtonsMethod, NormalEquations


def test_normal_equations_matches_sklearn(diabetes_data):
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations()).fit(X, y)
    sk_model = SKLinearRegression().fit(X, y)

    assert np.allclose(model.weights, sk_model.coef_, atol=1e-6)
    assert np.isclose(model.bias, sk_model.intercept_, atol=1e-6)


def test_gradient_descent_matches_closed_form_on_synthetic_data(linear_data):
    X, y, _, _ = linear_data
    X_scaled = StandardScaler().fit_transform(X)
    gd_model = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=2000, penalty=None)
    ).fit(X_scaled, y)
    reference = LinearRegression(solver=NormalEquations()).fit(X_scaled, y)
    sk_model = SKLinearRegression().fit(X_scaled, y)

    assert np.allclose(gd_model.weights, reference.weights, atol=1e-6)
    assert np.isclose(gd_model.bias, reference.bias, atol=1e-6)
    assert np.allclose(gd_model.weights, sk_model.coef_, atol=1e-6)
    assert np.isclose(gd_model.bias, sk_model.intercept_, atol=1e-6)


def test_gradient_descent_explicit_full_batch_matches_default(linear_data):
    X, y, _, _ = linear_data
    X_scaled = StandardScaler().fit_transform(X)
    default_model = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=500, penalty=None)
    ).fit(X_scaled, y)
    explicit_model = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=500, penalty=None)
    ).fit(X_scaled, y, batch_size=X_scaled.shape[0])

    assert np.allclose(explicit_model.weights, default_model.weights)
    assert np.isclose(explicit_model.bias, default_model.bias)


def test_gradient_descent_minibatch_stays_finite(linear_data):
    X, y, _, _ = linear_data
    model = LinearRegression(
        solver=GradientDescent(learning_rate=0.05, epochs=500, penalty=None)
    ).fit(X, y, batch_size=32)

    assert np.all(np.isfinite(model.weights))
    assert np.isfinite(model.bias)


def test_gradient_descent_l2_shrinks_weights(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    unregularized = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=3000, penalty=None)
    ).fit(X_scaled, y)
    regularized = LinearRegression(
        solver=GradientDescent(
            learning_rate=0.1, epochs=3000, penalty="L2", alpha=10.0
        )
    ).fit(X_scaled, y)

    assert np.linalg.norm(regularized.weights) < np.linalg.norm(unregularized.weights)


def test_gradient_descent_l1_produces_exact_zeros(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    model = LinearRegression(
        solver=GradientDescent(
            learning_rate=0.1, epochs=3000, penalty="L1", alpha=0.5
        )
    ).fit(X_scaled, y)

    assert np.any(model.weights == 0.0)


def test_gradient_descent_elastic_net_is_sparse_and_shrunk(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    unregularized = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=3000, penalty=None)
    ).fit(X_scaled, y)
    regularized = LinearRegression(
        solver=GradientDescent(
            learning_rate=0.1,
            epochs=3000,
            penalty="EN",
            alpha=1.0,
            l1_ratio=0.5,
        )
    ).fit(X_scaled, y)

    assert np.any(regularized.weights == 0.0)
    assert np.linalg.norm(regularized.weights) < np.linalg.norm(unregularized.weights)


def test_newtons_method_matches_unpenalized_sklearn_logistic(logistic_data):
    X, y = logistic_data
    model = LogisticRegression(
        solver=NewtonsMethod(epochs=50, learning_rate=1.0, penalty=None)
    ).fit(X, y)
    sk_model = SKLogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(X, y)

    assert np.allclose(model.weights, sk_model.coef_.ravel(), atol=1e-5)
    assert np.isclose(model.bias, sk_model.intercept_[0], atol=1e-5)
    assert np.allclose(model.predict(X), sk_model.predict_proba(X)[:, 1], atol=1e-5)


def test_newtons_method_l2_shrinks_weights(logistic_data):
    X, y = logistic_data
    unregularized = LogisticRegression(
        solver=NewtonsMethod(epochs=50, learning_rate=1.0, penalty=None)
    ).fit(X, y)
    regularized = LogisticRegression(
        solver=NewtonsMethod(epochs=50, learning_rate=1.0, penalty="L2", alpha=5.0)
    ).fit(X, y)

    assert np.linalg.norm(regularized.weights) < np.linalg.norm(unregularized.weights)


@pytest.mark.parametrize("solver", [NormalEquations(), GradientDescent(epochs=10)])
def test_solvers_accept_unused_batch_size_argument(solver):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.0, -1.0, 0.5])

    model = LinearRegression(solver=solver).fit(X, y, batch_size=8)

    assert np.all(np.isfinite(model.weights))
    assert np.isfinite(model.bias)
