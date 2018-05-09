import importlib


EXPERIMENTS = {
    'DecoyMNIST': 'rfft.applications.decoy_mnist',
    # 'NewsGroup': 'rfft.applications.newsgroup',
}


class BaseSingleton(type):
    instance = None

    def __call__(cls, *args, **kw):
        if not cls.instance:
            cls.instance = super(BaseSingleton, cls).__call__(*args, **kw)
        return cls.instance


def get_experiment_class_from_name(name):
    module = importlib.import_module(EXPERIMENTS[name])
    experiment_class = getattr(module, name)
    return experiment_class


class ExperimentCache(object):

    __metaclass__ = BaseSingleton

    def __init__(self):
        def get_experiment_from_name(name):
            experiment_class = get_experiment_class_from_name(name)
            experiment = experiment_class()
            return experiment

        self._experiment_cache = {name: get_experiment_from_name(name) for name in EXPERIMENTS}

    def get_experiment_cache(self):
        return self._experiment_cache

    def get_experiment(self, experiment_name):
        if experiment_name not in self._experiment_cache:
            raise KeyError('Could not find experiment named {}.'.format(experiment_name))
        experiment = self._experiment_cache[experiment_name]
        if not experiment.status.initialized:
            experiment.generate_dataset()
        return experiment
