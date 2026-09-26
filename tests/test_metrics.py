"""
Tests for WaterML_on.metrics

ASSUMPTIONS: regression metrics in WaterML_on.metrics.regression
(MSE, MAE, RMSE, r2_score, MAPE), classification metrics in
WaterML_on.metrics.classification (confusion_matrix, accuracy, precision,
recall, specificity, f1_score, deviance). Adjust names/paths if yours differ.
"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score as sk_r2_score,
    accuracy_score, precision_score, recall_score, f1_score as sk_f1_score,
    confusion_matrix as sk_confusion_matrix, log_loss as sk_log_loss,
)

from WaterML_on.metrics.Regression import MSE, MAE, RMSE, r2_score, MAPE
from WaterML_on.metrics.Classification import (
    confusion_matrix, accuracy, precision, recall, specificity, f1_score, deviance,
)


def test_regression_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.5, size=50)

    assert np.isclose(MSE(y_pred, y_true), mean_squared_error(y_true, y_pred))
    assert np.isclose(MAE(y_pred, y_true), mean_absolute_error(y_true, y_pred))
    assert np.isclose(RMSE(y_pred, y_true), np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(r2_score(y_pred, y_true), sk_r2_score(y_true, y_pred))


def test_mape_is_reasonable_on_known_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # errors are 10%, 5%, 0% -> mean = 5%
    expected_mape = (0.10 + 0.05 + 0.0) / 3
    assert np.isclose(MAPE(y_pred, y_true), expected_mape, atol=1e-6)


def test_confusion_matrix_matches_sklearn():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    cm = confusion_matrix(y_true, y_pred)
    sk_cm = sk_confusion_matrix(y_true, y_pred)

    assert np.array_equal(cm, sk_cm.ravel())


def test_classification_metrics_match_sklearn():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

    cm = confusion_matrix(y_true, y_pred)

    assert np.isclose(accuracy(cm), accuracy_score(y_true, y_pred))
    assert np.isclose(precision(cm), precision_score(y_true, y_pred))
    assert np.isclose(recall(cm), recall_score(y_true, y_pred))
    assert np.isclose(f1_score(cm), sk_f1_score(y_true, y_pred))


def test_specificity_is_recall_of_the_negative_class():
    """
    sklearn has no direct specificity function, but specificity is
    recall_score computed on the flipped/negative class — a good
    independent way to check the formula rather than trusting it blindly.
    """
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    expected_specificity = recall_score(1 - y_true, 1 - y_pred)
    assert np.isclose(specificity(cm), expected_specificity)


def test_deviance_matches_sklearn_log_loss_up_to_constant():
    """
    Binomial deviance = 2 * n * log_loss (log_loss is the mean, deviance
    is the raw sum times 2). Checking this relationship is a more honest
    test than hand-deriving deviance independently.
    """
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=100)
    p_pred = np.clip(rng.uniform(size=100), 1e-10, 1 - 1e-10)

    sk_loss = sk_log_loss(y_true, p_pred)
    expected_deviance = 2 * len(y_true) * sk_loss

    assert np.isclose(deviance(y_true, p_pred), expected_deviance, rtol=1e-4)


def test_deviance_handles_extreme_probabilities_without_nan():
    y_true = np.array([1, 0, 1, 0])
    p_pred = np.array([1.0, 0.0, 1.0, 0.0])  # exactly at the boundary
    result = deviance(y_true, p_pred)
    assert np.isfinite(result)
