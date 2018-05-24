from flask import jsonify
from flask_restful import Resource

from .experiment.experiment_cache import get_experiment_class_from_name


class Explanation(Resource):

    def __init__(self):
        Resource.__init__(self)
        self.saved_experiments = {}

    def get(self, experiment_name, saved_experiment_id):
        try:
            experiment_class = get_experiment_class_from_name(experiment_name)
            if saved_experiment_id in self.saved_experiments:
                experiment = self.saved_experiments[saved_experiment_id]
            else:
                experiment = experiment_class.load_experiment(saved_experiment_id, True)
                self.saved_experiments[saved_experiment_id] = experiment
            masked_image = experiment.explain()
            return jsonify(masked_image)
        except KeyError as ke:
            print(ke)
            return str(ke), 400
        except Exception as ex:
            print(ex)
            return str(ex), 500
