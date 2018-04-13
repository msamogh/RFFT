from utils import l2_norm


class Hypothesis(object):

    def __init__(
        self,
        A,
        weight=1.0,
        affected_indices=None,
        normalize=False
    ):
        if affected_indices:
            assert len(A) == len(affected_indices), "Number of rows of mask matrix A must be equal to length of affected_indices."
        self.A = A
        self.weight = weight
        self.affected_indices = affected_indices
        self.normalize = normalize

    def __call__(self, i, input_grads):
        if self.affected_indices and i not in self.affected_indices:
            return 0
        mask = self.A[i]
        if self.normalize:
            normalization_factor = max(1., float(mask.sum()))
            mask /= normalization_factor
        return (
            self.weight *
            l2_norm(input_grads[mask])
        )
