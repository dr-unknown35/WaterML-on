import numpy as np


def _shuffle_indices(n_samples, random_state):
    rng=np.random.default_rng(random_state)
    indices=np.arange(n_samples)
    indices = rng.permutation(indices)
    return indices


def Kfolds(n_samples,nb_folds=5,random_state=None):
    return np.array_split(_shuffle_indices(n_samples, random_state), nb_folds)


def Kfolds_split(n_samples,nb_folds=5,random_state=None):
    folds=Kfolds(n_samples,nb_folds,random_state)
    for i in range(nb_folds):
        yield np.concatenate(folds[:i]+folds[i+1:]),folds[i]


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    n_samples=X.shape[0]
    indices=_shuffle_indices(n_samples, random_state) if shuffle else np.arange(n_samples)
    cutoff=int(n_samples * test_size)
    return X[indices[cutoff:]],X[indices[:cutoff]],y[indices[cutoff:]],y[indices[:cutoff]]






