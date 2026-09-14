from abc import ABC,abstractmethod
import numpy as np

class BaseSolver(ABC):
    @abstractmethod
    def optimize(self,X,y,epochs ,batch_size,learning_rate,penalty,alpha):
        pass



    def _penalty_gradient(self, weights):
        match self.penalty:
            case 'L2':
                return (self.alpha/self.batch_size)*weights
            case 'EN':
                return (self.alpha/self.batch_size)*(1-self.l1_ratio)*weights
            case _:
                return 0
    def _soft_threshold(self, weights, threshold):
        return np.sign(weights)*np.maximum(0,abs(weights)-(threshold)) # Soft thresholding