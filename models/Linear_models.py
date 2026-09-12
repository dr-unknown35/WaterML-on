import numpy as np
from .base import BaseEstimator






class GLM(BaseEstimator):
    def __init__(self, family, solver,learning_rate=0.01,epochs=2000, penality=None, alpha=0.1):
        self.family=family
        self.solver=solver
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.penality=penality
        self.alpha=alpha
        self.weights=None
        self.bias=None
        
    
    
    
    def fit(self,X,y):
        self.weights,self.bias=self.solver.optimize(self,X,y)
        return self
    
    
    def predict(self,X):
        return self.family.inverse_link(X@self.weights +self.bias)
        
        
class Linear_regression(GLM):
    def __init__(self,solver=NormalEquations()):
        super().__init__(family=GaussianFamily(),solver=solver)
        
        
    