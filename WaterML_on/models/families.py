from .base import BaseFamily
import numpy as np



class Gaussian(BaseFamily):
    def link(self, mu):
        return mu

    def inverse_link(self, eta):
        return eta

    def variance(self, mu):
        return np.ones_like(mu,dtype=float)



class Binomial(BaseFamily):
    def link(self, mu):
        mu=np.clip(mu,1e-15,1-1e-15)
        return np.log(mu/(1-mu))

    def inverse_link(self, eta):
        eta=np.clip(eta, -700, 700)
        return 1/(1+np.exp(-eta))

    def variance(self, mu):
        return mu*(1-mu)



class Poisson(BaseFamily):
    def link(self, mu):
        mu=np.maximum(mu,1e-15)
        return np.log(mu)

    def inverse_link(self, eta):
        eta=np.clip(eta, -700, 700)
        return np.exp(eta)

    def variance(self, mu):
        return mu


class Gamma(BaseFamily):
    def link(self, mu):
        mu=np.maximum(mu,1e-15)
        return np.log(mu)

    def inverse_link(self, eta):
        eta=np.clip(eta,-700,700)
        return np.exp(eta)

    def variance(self,mu):
        return mu**2



class InverseGaussian(BaseFamily):
    def link(self, mu):
        mu=np.maximum(mu,1e-15)
        return np.log(mu)

    def inverse_link(self, eta):
        eta=np.clip(eta,-700,700)
        return np.exp(eta)

    def variance(self,mu):
        return mu**3



class Multinomial(BaseFamily):
    def __init__(self,base_idx=-1):
        self.base_idx=base_idx
        
        
        
    def link(self,mu):
        mu=np.clip(mu, 1e-15, 1.0 - 1e-15)
        baseline_col=mu[:, self.base_idx] 
        
        return np.log(mu) -np.log(baseline_col[:, np.newaxis])


    def inverse_link(self,eta):
        exp_eta=np.exp(eta-eta.max(axis=1,keepdims=True))
        return exp_eta/exp_eta.sum(axis=1,keepdims=True)
    
    def variance(self,mu):
        mu=np.asarray(mu)
        if mu.ndim==1:
            return np.diag(mu)-np.outer(mu,mu)
        
        diag=np.zeros((*mu.shape,mu.shape[1]),dtype=mu.dtype)
        # unpack and create n=mu.shape[0] (mu.shape[0],mu.shape[1]) zero matrices
        idx=np.arange(mu.shape[1])
        diag[:, idx, idx] = mu
        # fill diag entries with mu for all n matrices
        return diag-mu[:,:,None]@mu[:,None,:]