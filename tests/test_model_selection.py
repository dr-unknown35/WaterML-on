"""
Tests for WaterML_on.model_selection

ASSUMPTIONS: cross_val_score in WaterML_on.model_selection.validation,
grid_search in WaterML_on.model_selection.search.
"""
import numpy as np

from WaterML_on.model_selection.validation import cross_val_score
from WaterML_on.model_selection.search import grid_search
from WaterML_on.models.Linear_models import LinearRegression
from WaterML_on.optim.gradient_methods import NormalEquations, GradientDescent
from WaterML_on.metrics.Regression import r2_score, MSE
from WaterML_on.data_utils.scaler import StandardScaler


def r2_metric(y_true, y_pred):
    return r2_score(y_pred, y_true)


def mse_metric(y_true, y_pred):
    return MSE(y_pred, y_true)


def test_cross_val_score_returns_reasonable_r2_on_diabetes(diabetes_data):
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations())

    mean_score = cross_val_score(model, X, y, nb_folds=5, metric=r2_metric, random_state=0)

    # Diabetes + plain OLS typically lands somewhere in the 0.4-0.5 R^2
    # range via CV — this is a loose sanity bound, not a precise target.
    assert 0.0 < mean_score < 1.0


def test_cross_val_score_does_not_mutate_the_passed_in_model(diabetes_data):
    """
    Regression test for the exact bug discussed while building this:
    cross_val_score should deepcopy per fold, so the original model
    object passed in should remain unfitted afterward.
    """
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations())

    cross_val_score(model, X, y, nb_folds=5, metric=r2_metric, random_state=0)

    assert model.weights is None
    assert model.bias is None


def test_grid_search_picks_the_lower_error_alpha_on_synthetic_data():
    """
    Construct a scenario where we know which alpha SHOULD win: very
    little regularization needed (low-noise linear data), so a small
    alpha should beat a huge one on MSE (greater_is_better=False).
    """
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 5
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ true_weights + rng.normal(scale=0.1, size=n_samples)
    X_scaled = StandardScaler().fit_transform(X)

    def factory(**params):
        return LinearRegression(
            solver=GradientDescent(learning_rate=0.1, epochs=2000, penalty='L2', **params)
        )

    best_params, best_score = grid_search(
        model_factory=factory,
        param_grid={'alpha': [0.001, 1.0, 100.0]},
        X=X_scaled, y=y,
        nb_folds=5,
        metric=mse_metric,
        greater_is_better=False,
        random_state=0,
    )

    assert best_params['alpha'] in (0.001, 1.0)  # not the huge, over-regularized option
