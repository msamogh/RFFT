from flask import jsonify
from flask_restful import request, Resource

from experiment.experiment_cache import ExperimentCache


class Experiment(Resource):

    def post(self, experiment_name):
        experiment = ExperimentCache().get_experiment(experiment_name)
        experiment.generate_dataset()
