import numpy as np
from sklearn.metrics import mean_poisson_deviance

from WaterML_on.metrics.Regression import poisson_deviance
from WaterML_on.models.Linear_models import (
    GLM,
    GammaRegression,
    InverseGaussianRegression,
    PoissonRegression,
    SoftmaxRegression,
)
from WaterML_on.models.families import (
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    Multinomial,
)
from WaterML_on.optim.gradient_methods import GradientDescent, NewtonsMethod, NormalEquations


def test_glm_get_gradient_matches_numpy_definition():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    weights = rng.normal(size=3)
    bias = 0.7
    model = GLM(family=Gaussian(), solver=NormalEquations())

    bias_gradient, weights_gradient = model.get_gradient(X, y, weights, bias)

    errors = X @ weights + bias - y
    assert np.isclose(bias_gradient, np.mean(errors))
    assert np.allclose(weights_gradient, X.T @ errors / X.shape[0])


def test_glm_constructor_configures_default_solver():
    model = GLM(learning_rate=0.01, epochs=17, penalty="L2", alpha=0.4)

    assert isinstance(model.solver, GradientDescent)
    assert model.solver.learning_rate == 0.01
    assert model.solver.epochs == 17
    assert model.solver.penalty == "L2"
    assert model.solver.alpha == 0.4


def test_glm_predict_class_applies_threshold(logistic_data):
    X, y = logistic_data
    model = GLM(
        family=Binomial(),
        solver=NewtonsMethod(epochs=30, learning_rate=1.0, penalty=None),
    ).fit(X, y)

    probabilities = model.predict(X)
    classes = model.predict_class(X, threshold=0.7)

    assert np.array_equal(classes, (probabilities >= 0.7).astype(int))
    assert np.mean(classes == y) > 0.5


def test_glm_fit_accepts_explicit_full_batch_size(poisson_data):
    X, y, _, _ = poisson_data
    kwargs = {"learning_rate": 0.05, "epochs": 1000, "penalty": None}
    default_model = PoissonRegression(solver=GradientDescent(**kwargs)).fit(X, y)
    explicit_model = PoissonRegression(solver=GradientDescent(**kwargs)).fit(
        X, y, batch_size=X.shape[0]
    )

    assert np.allclose(explicit_model.weights, default_model.weights)
    assert np.isclose(explicit_model.bias, default_model.bias)


def test_poisson_regression_reduces_deviance_relative_to_intercept(poisson_data):
    X, y, _, _ = poisson_data
    model = PoissonRegression(
        solver=GradientDescent(learning_rate=0.05, epochs=2000, penalty=None)
    ).fit(X, y)

    fitted_deviance = poisson_deviance(y, model.predict(X))
    intercept_deviance = poisson_deviance(y, np.full_like(y, y.mean(), dtype=float))

    assert np.isfinite(fitted_deviance)
    assert fitted_deviance < intercept_deviance
    assert np.isclose(
        fitted_deviance / len(y),
        mean_poisson_deviance(y, model.predict(X)),
    )


def test_softmax_regression_predict_proba_rows_sum_to_one():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(30, 4))
    model = SoftmaxRegression()
    model.weights = rng.normal(size=(4, 3))
    model.bias = rng.normal(size=3)

    probabilities = model.predict(X)

    assert isinstance(model.family, Multinomial)
    assert probabilities.shape == (30, 3)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.all(probabilities >= 0)


def test_gamma_and_inverse_gaussian_models_expose_expected_families():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(20, 2))
    weights = np.array([0.2, -0.1])
    bias = 0.3

    gamma = GammaRegression()
    gamma.weights = weights
    gamma.bias = bias
    inverse_gaussian = InverseGaussianRegression()
    inverse_gaussian.weights = weights
    inverse_gaussian.bias = bias

    expected = np.exp(X @ weights + bias)

    assert isinstance(gamma.family, Gamma)
    assert isinstance(inverse_gaussian.family, InverseGaussian)
    assert np.allclose(gamma.predict(X), expected)
    assert np.allclose(inverse_gaussian.predict(X), expected)
