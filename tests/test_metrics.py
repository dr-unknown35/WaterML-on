import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1_score,
    log_loss as sk_log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score as sk_r2_score,
    recall_score,
)

from WaterML_on.metrics.Classification import (
    accuracy,
    confusion_matrix,
    deviance,
    f1_score,
    log_loss,
    precision,
    recall,
    specificity,
)
from WaterML_on.metrics.Regression import MAE, MAPE, MSE, RMSE, poisson_deviance, r2_score


def test_regression_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.5, size=50)

    assert np.isclose(MSE(y_pred, y_true), mean_squared_error(y_true, y_pred))
    assert np.isclose(MAE(y_pred, y_true), mean_absolute_error(y_true, y_pred))
    assert np.isclose(RMSE(y_pred, y_true), np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(r2_score(y_pred, y_true), sk_r2_score(y_true, y_pred))


def test_r2_score_handles_shape_errors_and_constant_targets():
    y_true = np.array([2.0, 2.0, 2.0])

    assert r2_score(y_true, y_true) == 1.0
    assert r2_score(y_true + 1.0, y_true) == 0.0

    with pytest.raises(ValueError):
        r2_score(np.ones(3), np.ones(2))


def test_mape_matches_sklearn_on_positive_and_negative_targets():
    y_true = np.array([100.0, -200.0, 300.0, -50.0])
    y_pred = np.array([110.0, -190.0, 285.0, -60.0])

    assert np.isclose(
        MAPE(y_pred, y_true), mean_absolute_percentage_error(y_true, y_pred)
    )


def test_mape_zero_handling_matches_masked_sklearn_reference():
    y_true = np.array([100.0, 0.0, 200.0, 0.0])
    y_pred = np.array([110.0, 5.0, 190.0, 7.0])
    mask = y_true != 0

    assert np.isclose(
        MAPE(y_pred, y_true, ignore_zeros=True),
        mean_absolute_percentage_error(y_true[mask], y_pred[mask]),
    )
    with pytest.raises(ValueError):
        MAPE(y_pred, y_true)
    with pytest.raises(ValueError):
        MAPE(np.array([1.0, 2.0]), np.zeros(2), ignore_zeros=True)


def test_mape_rejects_scalars_and_mismatched_shapes():
    with pytest.raises(ValueError):
        MAPE(1.0, np.array([1.0]))
    with pytest.raises(ValueError):
        MAPE(np.array([1.0]), np.array([1.0, 2.0]))


def test_poisson_deviance_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.poisson(3.0, size=100)
    y_pred = rng.uniform(0.5, 5.0, size=100)

    actual = poisson_deviance(y_true, y_pred)
    expected = len(y_true) * mean_poisson_deviance(y_true, y_pred)

    assert np.isclose(actual, expected, rtol=1e-9)


def test_confusion_matrix_matches_sklearn():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    assert np.array_equal(
        confusion_matrix(y_true, y_pred), sk_confusion_matrix(y_true, y_pred).ravel()
    )


def test_classification_metrics_match_sklearn():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred)

    assert np.isclose(accuracy(cm), accuracy_score(y_true, y_pred))
    assert np.isclose(precision(cm), precision_score(y_true, y_pred))
    assert np.isclose(recall(cm), recall_score(y_true, y_pred))
    assert np.isclose(f1_score(cm), sk_f1_score(y_true, y_pred))


def test_specificity_is_recall_of_the_negative_class():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred)

    assert np.isclose(specificity(cm), recall_score(1 - y_true, 1 - y_pred))


def test_binary_log_loss_matches_sklearn():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=80)
    p_pred = rng.uniform(0.01, 0.99, size=80)

    assert np.isclose(log_loss(y_true, p_pred), sk_log_loss(y_true, p_pred))


def test_deviance_matches_sklearn_log_loss_up_to_constant():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=100)
    p_pred = np.clip(rng.uniform(size=100), 1e-10, 1 - 1e-10)

    expected = 2 * len(y_true) * sk_log_loss(y_true, p_pred)

    assert np.isclose(deviance(y_true, p_pred), expected, rtol=1e-8)


def test_deviance_handles_extreme_probabilities_without_nan():
    y_true = np.array([1, 0, 1, 0])
    p_pred = np.array([1.0, 0.0, 1.0, 0.0])

    assert np.isfinite(deviance(y_true, p_pred))
