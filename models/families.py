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