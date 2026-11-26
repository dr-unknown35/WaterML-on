import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from WaterML_on.models.GDA import Linear, Quadratic
from WaterML_on.models.Naive_Bayes import BernoulliEM, MultinomialEM


def test_linear_gda_matches_sklearn_lda(logistic_data):
    X, y = logistic_data
    model = Linear()
    model.fit(X, y)
    sk_model = LinearDiscriminantAnalysis().fit(X, y)

    assert np.allclose(model.predict_proba(X), sk_model.predict_proba(X))
    assert np.array_equal(model.predict(X), sk_model.predict(X))


def test_quadratic_gda_matches_sklearn_qda(logistic_data):
    X, y = logistic_data
    model = Quadratic()
    model.fit(X, y)
    sk_model = QuadraticDiscriminantAnalysis().fit(X, y)

    assert np.allclose(model.predict_proba(X), sk_model.predict_proba(X))
    assert np.array_equal(model.predict(X), sk_model.predict(X))


def test_bernoulli_naive_bayes_matches_sklearn(binary_features):
    X, y = binary_features
    model = BernoulliEM()
    model.fit(X, y)
    sk_model = BernoulliNB(alpha=0.5, force_alpha=True, binarize=None).fit(X, y)

    assert np.array_equal(model.predict(X), sk_model.predict(X))


def test_multinomial_naive_bayes_matches_sklearn(count_features):
    X, y = count_features
    model = MultinomialEM()
    model.fit(X, y)
    sk_model = MultinomialNB(alpha=1.0, force_alpha=True).fit(X, y)

    assert np.array_equal(model.predict(X), sk_model.predict(X))
