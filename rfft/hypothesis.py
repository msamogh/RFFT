class Hypothesis(object):

    def __init__(
        self,
        A,
        weight=10,
        normalize=False
    ):
        self.A = A
        self.weight = weight
        self.normalize = normalize
