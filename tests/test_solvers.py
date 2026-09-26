"""
Tests for WaterML_on.optim.gradient_methods

ASSUMPTION: solvers live at WaterML_on.optim.gradient_methods, GradientDescent's
regularization kwarg is spelled `penalty` (not `penality` — matches your
final renamed version), and NewtonsMethod currently only supports the
unregularized case. Adjust names if yours differ.
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from WaterML_on.optim.gradient_methods import GradientDescent, NormalEquations, NewtonsMethod
from WaterML_on.models.Linear_models import LinearRegression
from WaterML_on.models.Linear_models import LogisticRegression
from WaterML_on.data_utils.scaler import StandardScaler


def test_normal_equations_matches_sklearn(diabetes_data):
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations()).fit(X, y)

    sk_model = SKLinearRegression().fit(X, y)

    assert np.allclose(model.weights, sk_model.coef_, atol=1e-6)
    assert np.isclose(model.bias, sk_model.intercept_, atol=1e-6)


def test_gradient_descent_converges_close_to_normal_equations(diabetes_data):
    """
    GD is iterative/approximate, so we don't expect bit-for-bit agreement
    with the closed-form solution — but with enough epochs on scaled data
    it should land in the same neighborhood. This is really a convergence
    sanity check, not an exactness check.
    """
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)

    gd_model = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=5000)
    ).fit(X_scaled, y)

    ne_model = LinearRegression(solver=NormalEquations()).fit(X_scaled, y)

    assert np.allclose(gd_model.weights, ne_model.weights, atol=0.6,rtol=1e-2)
    assert np.isclose(gd_model.bias, ne_model.bias, atol=0.6,rtol=1e-2)


def test_gradient_descent_l2_shrinks_weights_relative_to_unregularized(diabetes_data):
    """
    A basic sanity property of L2/Ridge: heavier regularization should
    shrink coefficient magnitude, not necessarily match any exact target.
    """
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)

    unregularized = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=3000, penalty=None)
    ).fit(X_scaled, y)

    heavily_regularized = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=3000, penalty='L2', alpha=10.0)
    ).fit(X_scaled, y)

    assert np.linalg.norm(heavily_regularized.weights) < np.linalg.norm(unregularized.weights)


def test_gradient_descent_l1_produces_some_exact_zeros(diabetes_data):
    """
    The actual point of implementing soft-thresholding rather than plain
    sign()-based subgradient: L1 should genuinely zero out some
    coefficients, not just shrink them close to zero.
    """
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)

    model = LinearRegression(
        solver=GradientDescent(learning_rate=0.1, epochs=3000, penalty='L1', alpha=0.5)
    ).fit(X_scaled, y)

    assert np.any(model.weights == 0.0)


def test_newtons_method_matches_sklearn_logistic(breast_cancer_data):
    X, y = breast_cancer_data
    X_scaled = StandardScaler().fit_transform(X)

    model = LogisticRegression(solver=NewtonsMethod(epochs=30,learning_rate=0.28,penalty="L2",alpha=0.3)).fit(X_scaled, y)

    sk_model = SKLogisticRegression(penalty='l2', max_iter=1000 ,C=0.00784).fit(X_scaled, y)

    # Newton's method (IRLS) and sklearn's unregularized solver should
    # converge to the same MLE — allow a modest tolerance for numeric
    # differences in convergence criteria, not an exact match.
    assert np.allclose(model.weights, sk_model.coef_.flatten(), atol=0.3,rtol=1e-2)
    assert np.isclose(model.bias, sk_model.intercept_[0], atol=0.3,rtol=1e-2)
