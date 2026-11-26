import numpy as np

from WaterML_on.models.SVM import MulticlassOvR, PrimalSubGD
from WaterML_on.models.perceptron import Perceptron


class FakeMachine:
    def __init__(self, offset):
        self.offset = offset

    def predict_raw(self, X):
        return X[:, 0] + self.offset


def test_perceptron_separates_linearly_separable_data():
    rng = np.random.default_rng(5)
    X = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.5, size=(50, 2)),
            rng.normal(loc=2.0, scale=0.5, size=(50, 2)),
        ]
    )
    y = np.array([0] * 50 + [1] * 50)
    model = Perceptron(learning_rate=0.1, epochs=100).fit(X, y)

    assert np.array_equal(model.predict(X), y)


def test_perceptron_explicit_full_batch_matches_default():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    kwargs = {"learning_rate": 0.05, "epochs": 50}
    default_model = Perceptron(**kwargs).fit(X, y)
    explicit_model = Perceptron(**kwargs).fit(X, y, batch_size=X.shape[0])

    assert np.allclose(explicit_model.weights, default_model.weights)
    assert np.isclose(explicit_model.bias, default_model.bias)


def test_primal_svm_gradient_matches_manual_hinge_subgradient():
    X = np.array([[1.0, 0.0], [-1.0, 0.5], [0.5, -1.0], [-0.5, -0.5]])
    y = np.array([1, -1, 1, -1])
    weights = np.array([0.3, -0.2])
    bias = 0.1
    model = PrimalSubGD(lr=0.1, epochs=1, decay=0.0, C=1.0, solver=None)

    bias_gradient, weights_gradient = model.get_gradient(X, y, weights, bias)

    violating = y * (X @ weights + bias) < 1
    expected_bias = -np.sum(y[violating]) / X.shape[0]
    expected_weights = -(X[violating].T @ y[violating]) / X.shape[0]

    assert np.isclose(bias_gradient, expected_bias)
    assert np.allclose(weights_gradient, expected_weights)


def test_primal_svm_predict_uses_zero_threshold():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    model = PrimalSubGD(lr=0.1, epochs=1, decay=0.0, C=1.0, solver=None)
    model.weights = np.array([1.0, 0.0])
    model.bias = 0.0

    assert np.array_equal(model.predict(X), np.array([1, 0, 1]))


def test_multiclass_ovr_predict_uses_argmax_raw_scores():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 2))
    model = MulticlassOvR(lr=0.1, epochs=1, decay=0.0, C=1.0, solver=None)
    model.classes = np.array([2, 4, 7])
    model.machines = [FakeMachine(0.0), FakeMachine(0.5), FakeMachine(-0.5)]

    raw = np.column_stack(
        [X[:, 0], X[:, 0] + 0.5, X[:, 0] - 0.5]
    )
    expected = model.classes[np.argmax(raw, axis=1)]

    assert np.array_equal(model.predict(X), expected)
    assert np.allclose(model.predict_raw(X), raw)
