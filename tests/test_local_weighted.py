"""
Tests for LocalWeightedLR / LocalWeightedSolver

ASSUMPTION: LocalWeightedLR lives at WaterML_on.models.local — adjust if
you put it somewhere else (e.g. inside linear.py).
"""
import numpy as np

from WaterML_on.models.Linear_models import LocalWeightedLR
from WaterML_on.data_utils.scaler import StandardScaler


def test_lwr_predictions_are_finite_and_correct_shape(diabetes_data):
    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)

    model = LocalWeightedLR(tau=5.0).fit(X_scaled, y)
    preds = model.predict(X_scaled[:20])

    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))


def test_lwr_outperforms_or_matches_global_linear_regression_locally(diabetes_data):
    """
    LWR's whole premise is fitting local structure better than a single
    global line where the relationship is non-linear. This isn't a strict
    guarantee on every dataset, but on diabetes (known mild non-linearity)
    a well-tuned tau should be at least competitive with global OLS on
    held-out data — mainly a smoke test that LWR isn't badly broken.
    """
    from WaterML_on.models.Linear_models import LinearRegression
    from WaterML_on.optim.gradient_methods import NormalEquations
    from WaterML_on.data_utils.splitter import train_test_split
    from WaterML_on.metrics.Regression import MSE

    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    lwr = LocalWeightedLR(tau=3.0).fit(X_train, y_train)
    lwr_preds = lwr.predict(X_test)

    linear = LinearRegression(solver=NormalEquations()).fit(X_train, y_train)
    linear_preds = linear.predict(X_test)

    lwr_mse = MSE(lwr_preds, y_test)
    linear_mse = MSE(linear_preds, y_test)

    # Generous check: LWR shouldn't be drastically worse than global OLS.
    # If this fails badly, it's worth checking the tau value and the
    # closed-form weighted normal equations math directly.
    assert lwr_mse < linear_mse * 2.0


def test_lwr_smaller_tau_fits_training_data_more_tightly(diabetes_data):
    """
    Sanity check on the bandwidth's actual effect: a small tau should
    weight nearby points much more heavily, generally producing a tighter
    (lower-error) fit ON THE TRAINING SET itself than a very large tau,
    which approaches a global fit.
    """
    from WaterML_on.metrics.Regression import MSE

    X, y = diabetes_data
    X_scaled = StandardScaler().fit_transform(X)

    small_tau_model = LocalWeightedLR(tau=0.5).fit(X_scaled, y)
    large_tau_model = LocalWeightedLR(tau=50.0).fit(X_scaled, y)

    small_tau_mse = MSE(small_tau_model.predict(X_scaled), y)
    large_tau_mse = MSE(large_tau_model.predict(X_scaled), y)

    assert small_tau_mse <= large_tau_mse
