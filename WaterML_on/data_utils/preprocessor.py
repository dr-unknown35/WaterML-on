import numpy as np





def _add_intercept(X):
    return np.c_[np.ones(X.shape[0]),X]


def one_hot(y,true_count=None):
    if true_count is None:
        unique_vals,indices=np.unique(y,return_inverse=True)
        return np.eye(len(unique_vals),dtype=int)[indices]
    else:
        return np.eye(true_count,dtype=int)[y]



