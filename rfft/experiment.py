from abc import ABCMeta
from abc import abstractmethod


class Experiment(metaclass=ABCMeta):
    
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def load_hypothesis(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def explain(self):
        pass
