from flask import jsonify
from flask_restful import request, Resource

from experiment.experiment_cache import ExperimentCache


class Experiment(Resource):

    def get(self, experiment_name):
        experiment = ExperimentCache().get_experiment(experiment_name)
        return jsonify(experiment.status.__dict__)
