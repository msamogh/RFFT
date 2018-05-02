from flask_restful import request, Resource
from experiment.experiment_cache import ExperimentCache

import json
import numpy as np


class Annotation(Resource):

    def put(self, experiment_name, sample_idx):
        req_json = request.get_json()
        annotation = req_json['annotation']
        try:
            experiment = ExperimentCache().get_experiment(experiment_name)
            experiment.set_annotation(sample_idx, annotation)
        except KeyError as ke:
            return str(ke), 400
        except Exception as ex:
            return str(ex), 500
        return 'Annotation added.', 200

    def get(self, experiment_name, sample_idx):
        try:
            experiment = ExperimentCache().get_experiment(experiment_name)
            annotation = experiment.get_annotation(sample_idx)
            return {'annotation': json.dumps(annotation)}
        except KeyError as ke:
            return str(ke), 400
        except Exception as ex:
            return str(ex), 500

    def delete(self, experiment_name, sample_idx):
        try:
            experiment = ExperimentCache().get_experiment(experiment_name)
            annotation = experiment.delete_annotation(sample_idx)
            return 'Annotation deleted', 200
        except KeyError as ke:
            return str(ke), 400
        except Exception as ex:
            return str(ex), 500
