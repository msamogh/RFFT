from abc import ABCMeta
from abc import abstractmethod

from enum import Enum


class ExperimentType(Enum):
    IMAGE = 1
    TEXT = 2
    TABULAR = 3


class ExperimentStatus(Enum):
    INITIALIZED = 0
    DATASET_GENERATED = 1
    ANNOTATIONS_LOADED = 2
    TRAINED = 3


class Experiment(metaclass=ABCMeta):

    def __init__(self):
        self.status = ExperimentStatus.INITIALIZED

    @abstractmethod
    def domain(self):
        pass

    @abstractmethod
    def status(self):
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
    def set_annotation(self, *args):
        pass

    @abstractmethod
    def get_annotation(self, *args):
        pass

    @abstractmethod
    def delete_annotation(self, *args):
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
