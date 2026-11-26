import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from WaterML_on.data_utils.scaler import StandardScaler
from WaterML_on.data_utils.splitter import Kfolds
from WaterML_on.metrics.Regression import MSE, r2_score
from WaterML_on.model_selection.search import grid_search
from WaterML_on.model_selection.validation import cross_val_score
from WaterML_on.models.Linear_models import LinearRegression, LogisticRegression
from WaterML_on.optim.gradient_methods import GradientDescent, NewtonsMethod, NormalEquations


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.value, dtype=float)


def r2_metric(y_true, y_pred):
    return r2_score(y_pred, y_true)


def mse_metric(y_true, y_pred):
    return MSE(y_pred, y_true)


def test_cross_val_score_returns_reasonable_r2_on_diabetes(diabetes_data):
    X, y = diabetes_data
    splitter = Kfolds(nb_folds=5, random_state=0)
    model = LinearRegression(solver=NormalEquations())

    mean_score = cross_val_score(model, X, y, r2_metric, "predict", splitter)

    assert 0.0 < mean_score < 1.0


def test_cross_val_score_does_not_mutate_passed_in_model(diabetes_data):
    X, y = diabetes_data
    model = LinearRegression(solver=NormalEquations())

    cross_val_score(model, X, y, r2_metric, "predict", Kfolds(nb_folds=5, random_state=0))

    assert model.weights is None
    assert model.bias is None


def test_cross_val_score_uses_requested_response(logistic_data):
    X, y = logistic_data
    model = LogisticRegression(
        solver=NewtonsMethod(epochs=30, learning_rate=1.0, penalty=None)
    )

    mean_accuracy = cross_val_score(
        model,
        X,
        y,
        accuracy_score,
        "predict_class",
        Kfolds(nb_folds=5, random_state=0),
    )

    assert np.isfinite(mean_accuracy)
    assert 0.5 < mean_accuracy <= 1.0


def test_cross_val_score_rejects_missing_response(diabetes_data):
    X, y = diabetes_data

    with pytest.raises(AttributeError):
        cross_val_score(
            LinearRegression(solver=NormalEquations()),
            X,
            y,
            r2_metric,
            "predict_proba",
            Kfolds(nb_folds=5, random_state=0),
        )


def test_grid_search_picks_the_lower_error_alpha_on_synthetic_data(linear_data):
    X, y, _, _ = linear_data
    X_scaled = StandardScaler().fit_transform(X)
    splitter = Kfolds(nb_folds=5, random_state=0)

    def factory(**params):
        return LinearRegression(
            solver=GradientDescent(
                learning_rate=0.1, epochs=2000, penalty="L2", **params
            )
        )

    best_params, best_score, all_scores = grid_search(
        model_factory=factory,
        param_grid={"alpha": [0.001, 1.0, 100.0]},
        X=X_scaled,
        y=y,
        metric=mse_metric,
        response="predict",
        splitter=splitter,
        greater_is_better=False,
    )

    assert best_params["alpha"] in (0.001, 1.0)
    assert np.isfinite(best_score)
    assert len(all_scores) == 3


def test_grid_search_honors_greater_is_better():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    splitter = Kfolds(nb_folds=5, shuffle=False)

    def metric(y_true, y_pred):
        return y_pred[0]

    best_high, _, scores_high = grid_search(
        model_factory=ConstantModel,
        param_grid={"value": [1, 2, 3]},
        X=X,
        y=y,
        metric=metric,
        splitter=splitter,
        greater_is_better=True,
    )
    best_low, _, scores_low = grid_search(
        model_factory=ConstantModel,
        param_grid={"value": [1, 2, 3]},
        X=X,
        y=y,
        metric=metric,
        splitter=splitter,
        greater_is_better=False,
    )

    assert best_high == {"value": 3}
    assert best_low == {"value": 1}
    assert [score for _, score in scores_high] == [1, 2, 3]
    assert [score for _, score in scores_low] == [1, 2, 3]
