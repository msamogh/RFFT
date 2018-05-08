import json

from flask_restful import request, Resource
from experiment.experiment_cache import ExperimentCache


class Explanation(Resource):

    def post(self, experiment_name, sample_idx):
        req_json = json.loads(request.data.decode('utf-8'))
        try:
            experiment = ExperimentCache().get_experiment(experiment_name)
            experiment.explain(sample_idx, **req_json)
        except KeyError as ke:
            return str(ke), 400
        except Exception as ex:
            return str(ex), 500
