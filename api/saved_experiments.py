import json

from flask import jsonify
from flask_restful import Resource

from .experiment.experiment_cache import get_experiment_class_from_name


class SavedExperiments(Resource):

    def get(self, experiment_name):
        experiment_class = get_experiment_class_from_name(experiment_name)

        saved_experiments = json.load(open(experiment_class.SAVED_EXPERIMENTS_CACHE_FILE, 'rb'))
        return jsonify(saved_experiments)

        # saved_experiments = [experiment_class.load_experiment(x) for x in experiment_class.get_saved_experiments()]
        # saved_experiments_result = []

        # for experiment in saved_experiments:
        #     if experiment.hypothesis.per_annotation:
        #         weight = experiment.hypothesis.weight / max(experiment.hypothesis.num_annotations, 1)
        #     else:
        #         weight = experiment.hypothesis.weight
        #     saved_experiments_result.append({
        #         'name': experiment.name,
        #         'hypothesis_weight': weight,
        #         'per_annotation': experiment.hypothesis.per_annotation,
        #         'n_annotations': experiment.hypothesis.num_annotations,
        #         'train_accuracy': experiment.train_accuracy,
        #         'test_accuracy': experiment.test_accuracy,
        #         'num_epochs': experiment.num_epochs
        #     })

        return jsonify(saved_experiments_result)
