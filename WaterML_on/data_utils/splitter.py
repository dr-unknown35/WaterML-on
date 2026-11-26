import numpy as np
from .base import BaseSplitter




#deprecated until maybe not
""" def train_test_split(self,X, y):
        n_samples=X.shape[0]
        indices=self._shuffle_indices(n_samples, self.random_state) if self.shuffle else np.arange(n_samples)
        cutoff=int(n_samples * self.test_size)
        return X[indices[cutoff:]],X[indices[:cutoff]],y[indices[cutoff:]],y[indices[:cutoff]]"""






class Kfolds(BaseSplitter):

    def __init__(self,nb_folds=5,random_state=28,test_size=0.2,shuffle=True):
        self.nb_folds=nb_folds
        self.random_state=random_state
        self.shuffle=shuffle
        self.test_size=test_size



    def _shuffle_indices(self,n_samples):
        rng=np.random.default_rng(self.random_state)
        indices=np.arange(n_samples)
        indices = rng.permutation(indices)
        return indices



    def split(self,X,y=None):
        n_samples=X.shape[0]
        folds=np.array_split(self._shuffle_indices(n_samples) if self.shuffle else np.arange(n_samples), self.nb_folds)
        for i in range(self.nb_folds):
            yield np.concatenate(folds[:i]+folds[i+1:]),folds[i]

    
   








class Stratified_Kfolds(BaseSplitter):

    def __init__(self,nb_folds=5,random_state=28,shuffle=True):
        self.nb_folds=nb_folds
        self.random_state=random_state
        self.shuffle=shuffle



    def split(self,X,y):
        y=np.asarray(y)
        classes,counts=np.unique(y, return_counts=True)
        minimum=np.min(counts)
        if minimum <self.nb_folds:
            raise ValueError(f"Cannot Stratify. Class with {minimum} members is smaller than number of folds {self.nb_folds}.")
        rng=np.random.default_rng(self.random_state)
        chunks=[]
        for c in classes:
            idx_c=rng.permutation(np.where(y==c)[0]) if self.shuffle else np.where(y==c)[0]
            chunks.append(np.array_split(idx_c,self.nb_folds))

        for i in range(self.nb_folds):
            yield np.concatenate([chunk[idx] for chunk in chunks for idx in range(self.nb_folds) if idx!=i  ]),np.concatenate([chunk[i] for chunk in chunks])