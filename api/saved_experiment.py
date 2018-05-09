from flask import jsonify
from flask_restful import Resource

from .experiment.experiment_cache import get_experiment_class_from_name


class SavedExperiment(Resource):

    def get(self, experiment_name, save_id):
        experiment_class = get_experiment_class_from_name(experiment_name)
        experiment = experiment_class.load_experiment(save_id, prepend_path=True)
        return jsonify({
            'name': experiment.name,
            'hypothesis_weight': experiment.hypothesis.weight,
            'per_annotation': experiment.hypothesis.per_annotation,
            'n_annotations': experiment.hypothesis.num_annotations,
            'train_accuracy': experiment.train_accuracy,
            'test_accuracy': experiment.test_accuracy,
            'num_epochs': experiment.num_epochs
        })
