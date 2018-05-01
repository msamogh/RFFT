import random
import autograd.numpy as np


class Hypothesis(object):

    def __init__(
        self,
        A,
        weight=10,
        normalize=False
    ):
        self.A = A
        self.A_inverse = np.array(A, copy=True)
        self.A_inverse[self.A_inverse == 1] = 2
        self.A_inverse[self.A_inverse == 0] = 1
        self.A_inverse[self.A_inverse == 2] = 0
        self.weight = weight
        self.normalize = normalize

    @staticmethod
    def incrementally_sample(annotations, hypothesis_load_fn, mask_shape, increment=5, shuffle=True):
        if shuffle:
            random.shuffle(annotations)
        for i in range(0, len(annotations) + 1, increment):
            yield hypothesis_load_fn(mask_shape, annotations[:i])

    def __repr__(self):
        return 'Hypothesis: (weight = %f)' % self.weight
