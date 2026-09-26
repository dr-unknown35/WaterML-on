from .base import BaseEstimator
from ..optim.gradient_methods import PerceptronSolver
import numpy as np



class Perceptron(BaseEstimator):
    def __init__(self,learning_rate=0.01,epochs=2000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.solver=PerceptronSolver()
        self.weights=None
        self.bias=None
        
    @staticmethod
    def _step(x):
        return np.where(x >= 0, 1, 0)



    def fit(self,X,y):
        self.bias,self.weights=self.solver.optimize(self,X,y)
        return self
    
    
    
    def predict(self, X):
        return self._step(X @ self.weights +self.bias)







