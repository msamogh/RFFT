from flask_restful import request, Resource


EXPERIMENTS = {
    'decoy_mnist': rfft.experiments.decoy_mnist.DecoyMNIST,
}


class Annotation(Resource):

    def put(self, experiment_name, sample_idx):
        experiment = EXPERIMENTS[experiment_name]()
        experiment.


    def get(self, experiment_name, sample_idx):
        pass
