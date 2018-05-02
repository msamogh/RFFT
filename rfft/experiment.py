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
    """Represents an experiment.
    """

    def __init__(self):
        self.status = ExperimentStatus.INITIALIZED

    @abstractmethod
    def domain(self):
        """Returns the nature of the domain of the experiment - text, image, or tabular.
        The values can take on any of the values from ExperimentType.
        """
        pass

    @abstractmethod
    def status(self):
        """Returns the current state of the experiment.
        The values can take on any of the values from ExperimentStatus.
        """
        pass
    
    @abstractmethod
    def generate_dataset(self, *args):
        """Loads and preprocesses the dataset."""
        pass
    
    @abstractmethod
    def load_annotations(self, *args):
        """Loads and processed annotations."""
        pass
    
    @abstractmethod
    def clear_annotations(self, *args):
        """Removes any loaded annotations from the state."""
        pass
    
    @abstractmethod
    def set_annotation(self, *args):
        """Specifies the annotation for the given input sample."""
        pass

    @abstractmethod
    def get_annotation(self, *args):
        """Returns the annotation of the given input sample."""
        pass

    @abstractmethod
    def delete_annotation(self, *args):
        """Deletes annotation corresponding to the given input sample."""
        pass
    
    @abstractmethod
    def train(self, *args):
        """Initializes and trains a model on the generated train data."""
        pass
    
    @abstractmethod
    def score_model(self, *args):
        """Runs prediction of the model on train and test sets and returns the performance metrics."""
        pass
    
    @abstractmethod
    def explain(self, *args):
        """Explains the reasons for the prediction of the given input sample."""
        pass
