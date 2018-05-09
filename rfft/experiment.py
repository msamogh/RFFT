import os
import pickle

from abc import ABCMeta
from abc import abstractmethod

from enum import Enum


class ExperimentType(Enum):
    IMAGE = 1
    TEXT = 2
    TABULAR = 3


class Dataset(Enum):
    TRAIN = 1
    TEST = 2


class ExperimentStatus(object):

    def __init__(self, initialized=False, trained=False):
        self.initialized = initialized
        self.trained = trained


class Experiment():
    """Represents an experiment."""

    __metaclass__ = ABCMeta

    def __init__(self):
        self.status = ExperimentStatus()
        self.name = None

    @abstractmethod
    def domain(self):
        """Returns the data domain of the experiment - text, image, or tabular.
        The values can take on any of the values from ExperimentType.
        """
        pass

    @abstractmethod
    def pretty_name(self):
        """Returns human readable name of the experiment."""
        pass

    @abstractmethod
    def description(self):
        """Returns description of the experiment."""
        pass

    @abstractmethod
    def get_status(self):
        """Returns the current state of the experiment.
        The values can take on any of the values from ExperimentStatus.
        """
        pass

    @abstractmethod
    def generate_dataset(self):
        """Loads and preprocesses the dataset."""
        pass

    @abstractmethod
    def get_sample(self, dataset, sample_idx):
        """Returns the input sample from the train dataset."""
        pass

    @abstractmethod
    def load_annotations(self, **hypothesis_params):
        """Loads and processes annotations."""
        pass

    @abstractmethod
    def unload_annotations(self):
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

    @classmethod
    def get_saved_experiments(cls):
        """Returns list of paths for saved trained experiments."""
        try:
            saved_models = os.listdir(cls.MODELS_DIR)
            return [os.path.join(cls.MODELS_DIR, model) for model in saved_models]
        except Exception:
            return []

    @classmethod
    def load_experiment(cls, filepath, prepend_path=False):
        """Loads experiment from saved file and returns it."""
        experiment = cls()
        if prepend_path:
            filepath = os.path.join(cls.MODELS_DIR, filepath)
        exp_dict = pickle.load(open(filepath))
        experiment.__dict__.update(exp_dict)
        return experiment

    @abstractmethod
    def save_experiment(self):
        """Save experiment state to file."""
        pass

    @abstractmethod
    def score_model(self):
        """Runs prediction of the model on train and test sets and returns the performance
        metrics."""
        pass

    @abstractmethod
    def explain(self, sample, **experiment_params):
        """Explains the reasons for the prediction of the given input sample."""
        pass
