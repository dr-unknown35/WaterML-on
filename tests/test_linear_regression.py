import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SKLR
from WaterML_on.models.Linear_models import *

def test_matches_sklearn():
    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression().fit(X, y)
    sk_model = SKLR().fit(X, y)
    print(model.bias,model.weights,sk_model.intercept_,sk_model.coef_)
    assert np.allclose(model.weights, sk_model.coef_, atol=1e-4)