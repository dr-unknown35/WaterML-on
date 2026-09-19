from .base import BaseScaler
import numpy as np


class StandardScaler(BaseScaler):

    def __init__(self):
        self.mean=None
        self.std=None

    def fit(self,X):
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)

    def transform(self,X):
        return (X-self.mean)/self.std



class MinMaxScaler(BaseScaler):

    def __init__(self):
        self.min=None
        self.max=None

    def fit(self,X):
        self.min=np.min(X,axis=0)
        self.max=np.max(X,axis=0)

    def transform(self,X):
        return (X-self.min)/(self.max-self.min)