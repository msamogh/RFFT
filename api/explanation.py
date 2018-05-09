from flask import jsonify
from flask_restful import Resource

from .experiment.experiment_cache import get_experiment_class_from_name


class Explanation(Resource):

    def get(self, experiment_name, saved_experiment_id):
        try:
            experiment_class = get_experiment_class_from_name(experiment_name)
            experiment = experiment_class.load_experiment(saved_experiment_id, True)
            masked_image = experiment.explain()
            return jsonify(masked_image)
        except KeyError as ke:
            return str(ke), 400
        except Exception as ex:
            return str(ex), 500
