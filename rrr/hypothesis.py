from utils import l2_norm


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

    def __call__(self, i, input_grads):
        mask = self.A[i]
        if self.normalize:
            normalization_factor = max(1., float(mask.sum()))
            mask /= normalization_factor
        return (
            self.weight *
            l2_norm(input_grads[mask])
        )
