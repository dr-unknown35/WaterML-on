from abc import ABC,abstractmethod
import numpy as np

class BaseSolver(ABC):
    @abstractmethod
    def optimize(self,model,X,y,batch_size=None):
        pass
    


    def _penalty_gradient(self, weights,batch_size):
        match self.penalty:
            case 'L2':
                return (self.alpha/batch_size)*weights
            case 'EN':
                return (self.alpha/batch_size)*(1-self.l1_ratio)*weights
            case _:
                return 0
    
    
    
    def _soft_threshold(self, weights, threshold):
        return np.sign(weights)*np.maximum(0,abs(weights)-(threshold)) # Soft thresholding



class BaseLocalSolver(ABC):
    @abstractmethod
    def optimize(self,model,X,y,query_point):
        pass