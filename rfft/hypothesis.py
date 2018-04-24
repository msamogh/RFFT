class Hypothesis(object):

    def __init__(
        self,
        A,
        weight=1.0,
        normalize=False
    ):
        self.A = A
        self.weight = weight
        self.normalize = normalize
