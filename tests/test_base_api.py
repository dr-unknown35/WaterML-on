import numpy as np

from WaterML_on.models.base import BaseEstimator, baseSVM


class ConcreteEstimator(BaseEstimator):
    def fit(self, X, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.value)


class ConcreteSVM(baseSVM):
    def fit(self):
        return self

    def predict(self, X):
        return np.where(self.predict_raw(X) >= 0, 1, 0)


def test_base_estimator_score_passes_y_true_before_predictions():
    X = np.arange(8).reshape(4, 2)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    calls = []
    model = ConcreteEstimator().fit(X, y)

    def metric(y_true, y_pred):
        calls.append((np.asarray(y_true), np.asarray(y_pred)))
        return np.mean(y_pred - y_true)

    score = model.score(X, y, metric)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], y)
    assert np.array_equal(calls[0][1], model.predict(X))
    assert np.isclose(score, 0.0)


def test_base_svm_predict_raw_matches_numpy():
    X = np.array([[1.0, 2.0], [-1.0, 0.5]])
    model = ConcreteSVM()
    model.weights = np.array([0.5, -1.0])
    model.bias = 0.25

    expected = X @ model.weights + model.bias

    assert np.allclose(model.predict_raw(X), expected)
    assert np.array_equal(model.predict(X), (expected >= 0).astype(int))
