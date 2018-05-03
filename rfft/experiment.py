from abc import ABCMeta
from abc import abstractmethod

from collections import namedtuple
from enum import Enum


class ExperimentType(Enum):
    IMAGE = 1
    TEXT = 2
    TABULAR = 3


ExperimentStatus = namedtuple('ExperimentStatus', ['dataset_generated',
                                                   'annotations_loaded',
                                                   'trained'])


class Experiment(metaclass=ABCMeta):
    """Represents an experiment."""

    def __init__(self):
        self.status = ExperimentStatus(dataset_generated=False,
                                       annotations_loaded=False,
                                       trained=False)

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
    def generate_dataset(self):
        """Loads and preprocesses the dataset."""
        pass

    @abstractmethod
    def load_annotations(self, **hypothesis_params):
        """Loads and processed annotations."""
        pass

    @abstractmethod
    def clear_annotations(self):
        """Removes any loaded annotations from the state."""
        pass

    @abstractmethod
    def set_annotation(self, sample_idx, annotation):
        """Specifies the annotation for the given input sample."""
        pass

    @abstractmethod
    def get_annotation(self, sample_idx):
        """Returns the annotation of the given input sample."""
        pass

    @abstractmethod
    def delete_annotation(self, sample_idx):
        """Deletes annotation corresponding to the given input sample."""
        pass

    @abstractmethod
    def train(self, num_epochs):
        """Initializes and trains a model on the generated train data."""
        pass

    @abstractmethod
    def score_model(self):
        """Runs prediction of the model on train and test sets and returns the performance metrics."""
        pass

    @abstractmethod
    def explain(self, sample):
        """Explains the reasons for the prediction of the given input sample."""
        pass
