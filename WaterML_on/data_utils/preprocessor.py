import numpy as np





def _add_intercept(X):
    return np.c_[np.ones(X.shape[0]),X]

