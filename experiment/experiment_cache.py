import importlib
import rfft.applications.decoy_mnist


EXPERIMENTS = {
    'DecoyMNIST': 'rfft.applications.decoy_mnist',
}


class BaseSingleton(type):
    instance = None

    def __call__(cls, *args, **kw):
        if not cls.instance:
            cls.instance = super(BaseSingleton, cls).__call__(*args, **kw)
        return cls.instance


class ExperimentCache(metaclass=BaseSingleton):

    def __init__(self):
        self._experiment_cache = {}


    def get_experiment_cache(self):
        return self._experiment_cache


    def get_experiment(self, experiment_name):
        if experiment_name in self._experiment_cache:
            return self._experiment_cache[experiment_name]

        if experiment_name not in EXPERIMENTS:
            raise KeyError('Could not find experiment named {}.'.format(experiment_name))
        module = importlib.import_module(EXPERIMENTS[experiment_name])
        experiment = getattr(module, experiment_name)()
        self._experiment_cache[experiment_name] = experiment
        return self._experiment_cache[experiment_name]


    def remove_experiment(self, experiment_name):
        self._experiment_cache.pop(experiment_name)