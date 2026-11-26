import numpy as np

from WaterML_on.data_utils.scaler import StandardScaler
from WaterML_on.data_utils.splitter import Kfolds
from WaterML_on.metrics.Regression import MSE
from WaterML_on.models.Linear_models import LinearRegression, LocalWeightedLR
from WaterML_on.optim.gradient_methods import NormalEquations


def test_lwr_predictions_are_finite_and_correct_shape(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    model = LocalWeightedLR(tau=5.0).fit(X_scaled, y)

    preds = model.predict(X_scaled[:20])

    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))


def test_lwr_local_params_match_manual_weighted_normal_equations(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    query = X_scaled[0]
    tau = 5.0
    model = LocalWeightedLR(tau=tau).fit(X_scaled, y)

    bias, weights = model.local_params(query)
    X_b = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    sq_distances = np.sum((X_scaled - query) ** 2, axis=1)
    W = np.diag(np.exp(-sq_distances / (2 * tau**2)))
    expected = np.linalg.solve(X_b.T @ W @ X_b, X_b.T @ W @ y)

    assert np.allclose(np.r_[bias, weights], expected, atol=1e-8)


def test_lwr_small_tau_fits_training_data_more_tightly(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    small_tau = LocalWeightedLR(tau=0.5).fit(X_scaled, y)
    large_tau = LocalWeightedLR(tau=50.0).fit(X_scaled, y)

    assert MSE(small_tau.predict(X_scaled), y) <= MSE(
        large_tau.predict(X_scaled), y
    )


def test_lwr_is_competitive_with_global_ols_on_held_out_data(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    train_idx, test_idx = next(Kfolds(nb_folds=5, random_state=0).split(X_scaled, y))

    lwr = LocalWeightedLR(tau=3.0).fit(X_scaled[train_idx], y[train_idx])
    linear = LinearRegression(solver=NormalEquations()).fit(
        X_scaled[train_idx], y[train_idx]
    )

    lwr_mse = MSE(lwr.predict(X_scaled[test_idx]), y[test_idx])
    linear_mse = MSE(linear.predict(X_scaled[test_idx]), y[test_idx])

    assert lwr_mse < linear_mse * 2.0
