from abc import ABCMeta
from abc import abstractmethod

# Experiment types
IMAGE = 1
TEXT = 2
TABULAR = 3


class Experiment(metaclass=ABCMeta):

    @abstractmethod
    def domain(self):
        pass
    
    @abstractmethod
    def generate_dataset(self, *args):
        pass
    
    @abstractmethod
    def load_annotations(self, *args):
        pass
    
    @abstractmethod
    def clear_annotations(self, *args):
        pass
    
    @abstractmethod
    def add_annotation(self, *args):
        pass
    
    @abstractmethod
    def train(self, *args):
        pass
    
    @abstractmethod
    def score_model(self, *args):
        pass
    
    @abstractmethod
    def explain(self, *args):
        pass
