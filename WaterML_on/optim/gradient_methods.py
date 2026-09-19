from .base import BaseSolver
from ..data_utils.preprocessor import _add_intercept
import numpy as np





class GradientDescent(BaseSolver):
    def __init__(self,epochs=1000,batch_size=None,learning_rate=0.28,penalty=None,alpha=1,l1_ratio=0):
        self.epochs=epochs
        self.batch_size=batch_size 
        self.learning_rate=learning_rate
        self.penalty=penalty
        self.alpha=alpha
        self.weights=None
        self.bias=None
        self.l1_ratio=l1_ratio


    def optimize(self,model, X, y):
        n_samples,n_features=X.shape
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.epochs):
            for i in range(0,n_samples,self.batch_size):
                X_batch=X[i:i+self.batch_size]
                y_batch=y[i:i+self.batch_size]
                y_pred= model.family.inverse_link((X_batch @ self.weights) + self.bias)
                errors= y_pred-y_batch
                error_gradient= (1/self.batch_size) * X_batch.T @ errors
                bias_gradient= (1/self.batch_size) * np.sum(errors)
                weights_gradient= error_gradient+self._penalty_gradient(self.weights)
                self.weights-= self.learning_rate*weights_gradient
                if self.penalty in ('L1', 'EN'):
                    self.weights=self._soft_threshold(self.weights,self.learning_rate* self.alpha*(self.l1_ratio if self.penalty=="EN" else 1 ))
                self.bias-= self.learning_rate*bias_gradient

        return [self.bias,self.weights]

class NormalEquations(BaseSolver):
    def optimize(self,model, X, y):
        X_b=_add_intercept(X)
        weights=np.linalg.inv(X_b.T@X_b) @ X_b.T @ y
        return [weights[0],weights[1:]]


class NewtonsMethod(BaseSolver):
    def __init__(self,learning_rate=1.0,epochs=10):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None


    def optimize(self,model,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.epochs):
            y_pred= model.family.inverse_link((X @ self.weights) + self.bias)
            variance=model.family.variance(y_pred)
            errors= y_pred-y
            
            weights_gradient= (1/n_samples) * X.T @ errors
            bias_gradient= (1/n_samples) * np.sum(errors)
            weights_hessian=(1/n_samples)*(X.T*(variance) )@ X
            bias_hessian=(1/n_samples)* sum(variance)
            
            
            #note to self : apparently it's more numerically stable to use np.solve instead of explicitly calculating the inverse and multiplying...
            #self.weights-= self.learning_rate*   np.linalg.inv(weights_hessian)@ weights_gradient
            self.weights-= self.learning_rate*np.linalg.solve(weights_hessian, weights_gradient)
            self.bias-= self.learning_rate*bias_gradient /bias_hessian
        return [self.bias,self.weights]

    


class LocalWeightedSolver(BaseSolver):
    def optimize_local(self,model, X_train, y_train, query_point):
        errors=np.sum((X_train-query_point)**2, axis=1)
        W = np.diag(np.exp(-errors /(2*model.tau**2)))
        X_b=_add_intercept(X_train)
        weights=np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y_train)
        return [weights[0],weights[1:]]