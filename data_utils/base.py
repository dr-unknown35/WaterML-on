from abc import ABC, abstractmethod



class BaseScaler(ABC):
    @abstractmethod
    def fit(self,X):
        pass


    @abstractmethod
    def transfrom(self,X):
        pass


    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)