from flask import jsonify
from flask_restful import request, Resource

from experiment.experiment_cache import ExperimentCache


class Experiments(Resource):

    def get(self):
        experiment_cache = ExperimentCache().get_experiment_cache()
        experiments = []
        for exp in experiment_cache.values():
            experiments.append({
                'name': exp.pretty_name(),
                'description': exp.description(),
                'domain': exp.domain().value,
                'started': exp.status.started
            })
        return jsonify({'all_experiments': experiments})
