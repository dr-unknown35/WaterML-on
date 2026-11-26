from .base import BaseEstimator
from ..optim.gradient_methods import GradientDescent , NormalEquations, NewtonsMethod,LocalWeightedSolver
from .families import Gaussian, Binomial , Poisson , Gamma, Multinomial,InverseGaussian
import numpy as np

class GLM(BaseEstimator):
    def __init__(self,*,family=None,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        self.family=family if family is not None else Gaussian()
        self.learning_rate=learning_rate if learning_rate is not None else 0.028
        self.epochs=epochs if epochs is not None else 2800
        self.penalty=penalty
        self.alpha=alpha if alpha is not None else 0.28
        self.solver=solver if solver is not None else GradientDescent(epochs=self.epochs,learning_rate=self.learning_rate,penalty=self.penalty,alpha=self.alpha)
        self.weights=None
        self.bias=None
        

    def fit(self,X,y,batch_size=None):
        self.bias,self.weights=self.solver.optimize(self,X,y,batch_size)
        return self
    
    
    def predict(self,X):
        return self.family.inverse_link(X@self.weights +self.bias)


    def predict_class(self,X,threshold=0.5):
        return (self.predict(X)>=threshold).astype(int)
    
    
    def get_gradient(self,X,y,weights,bias):
        predictions=self.family.inverse_link(X@weights +bias)
        errors=predictions-y
        
        return (np.mean(errors,axis=0),X.T@errors/X.shape[0])
        







class LocalWeightedLR(BaseEstimator):
    def __init__(self,tau=1,solver=None,family=None):
        self.tau=tau
        self.solver=solver if solver is not None else LocalWeightedSolver()
        self.family=family if family is not None else Gaussian()
        self.X=None
        self.y=None


    def fit(self,X,y):
        self.X=X
        self.y=y
        return self


    def predict(self,test_X):
        pred=np.array([])
        bias, weights=0,np.zeros(test_X.shape[1])
        for row in test_X:
            bias,weights=self.solver.optimize(self, self.X, self.y, row)
            pred=np.append(pred,self.family.inverse_link(row@weights +bias))
        return pred
    
    
    def local_params(self, x):
        return self.solver.optimize(self,self.X,self.y, x)














class LinearRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        solver=NormalEquations() if solver is None else solver
        super().__init__(family=Gaussian(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)


class LogisticRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        super().__init__(family=Binomial(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)


class PoissonRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        super().__init__(family=Poisson(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)

        
class GammaRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        super().__init__(family=Gamma(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)


class InverseGaussianRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
        super().__init__(family=InverseGaussian(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)

class SoftmaxRegression(GLM):
    def __init__(self,*,solver=None,learning_rate=None,epochs=None,penalty=None,alpha=None):
            super().__init__(family=Multinomial(),solver=solver,learning_rate=learning_rate,epochs=epochs,penalty=penalty,alpha=alpha)







