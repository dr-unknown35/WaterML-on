from abc import ABC,abstractmethod


class BaseEstimator(ABC):

    
    @abstractmethod
    def fit(self,X,y):
        pass
    
    
    @abstractmethod
    def predict(self,X):
        pass
    
    
    def score(self,X,y,metric):
        predictions=self.predict(X)
        return metric(y,predictions)
    
    
    
class BaseFamily(ABC):


    @abstractmethod
    def link(self,X):
        pass


    @abstractmethod
    def reverse_link(self,X):
        pass

    @abstractmethod
    def variance(self,X):
        pass
