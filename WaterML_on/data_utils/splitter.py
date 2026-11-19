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





def stratified_Kfolds(y, nb_folds=5,random_state=None,shuffle=True):
    y=np.asarray(y)
    classes,counts=np.unique(y, return_counts=True)
    minimum=np.min(counts)
    if minimum <nb_folds:
        raise ValueError(f"Cannot Stratify: Class with {min} members has less than number of folds {nb_folds}")
    rng=np.random.default_rng(random_state)
    chunks=[]
    for c in classes:
        idx_c=rng.permutation(np.where(y==c)[0]) if shuffle else np.where(y==c)[0]
        chunks.append(np.array_split(idx_c,nb_folds))

    for i in range(nb_folds):
        yield np.concatenate([chunk[idx] for chunk in chunks for idx in range(nb_folds) if idx!=i  ]),np.concatenate([chunk[i] for chunk in chunks])

