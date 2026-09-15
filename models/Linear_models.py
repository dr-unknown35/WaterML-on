from .base import BaseEstimator
from ..optim.gradient_methods import GradientDescent , NormalEquations, NewtonsMethod,LocalWeightedSolver
from .families import Gaussian, Binomial , Poisson , Gamma
import numpy as np

class GLM(BaseEstimator):
    def __init__(self, family=None, solver=None ,learning_rate=0.01,epochs=2000, penalty=None, alpha=0.1):
        self.family=family if family !=None else Gaussian()
        self.solver=solver if solver !=None else GradientDescent()
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.penalty=penalty
        self.alpha=alpha
        self.weights=None
        self.bias=None
        

    def fit(self,X,y):
        self.bias,self.weights=self.solver.optimize(self,X,y)
        return self
    
    
    def predict(self,X):
        return self.family.inverse_link(X@self.weights +self.bias)


    def predict_class(self,X,threshold=0.5):
        return (self.predict(X)>=threshold).astype(int)



class LocalWeightedLR(BaseEstimator):
    def __init__(self,tau=1,solver=None,family=None):
        self.tau=tau
        self.solver=solver if solver !=None else LocalWeightedSolver()
        self.family=family if family !=None else Gaussian()
        self.X=None
        self.y=None
        self.weights=None
        self.bias=None


    def fit(self,X,y):
        self.X=X
        self.y=y
        return self


    def predict(self,test_X):
        pred=[]
        for row in test_X:
            self.bias,self.weights=self.solver.optimize_local(self, self.X, self.y, row)
            pred.append(self.family.inverse_link(row@self.weights +self.bias))
        return pred


    














class LinearRegression(GLM):
    def __init__(self,solver=NormalEquations()):
        super().__init__(family=Gaussian(),solver=solver)


class LogisticRegression(GLM):
    def __init__(self,solver=GradientDescent()):
        super().__init__(family=Binomial(),solver=solver)


class PoissonRegression(GLM):
    def __init__(self,solver=GradientDescent()):
        super().__init__(family=Poisson(),solver=solver)

        
class GammaRegression(GLM):
    def __init__(self,solver=GradientDescent()):
        super().__init__(family=Gamma(),solver=solver)


class InverseGaussianRegression(GLM):
    def __init__(self,solver=GradientDescent()):
        super().__init__(family=InverseGaussianRegression(),solver=solver)