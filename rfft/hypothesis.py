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
        invert = np.vectorize(lambda x: not x)
        self.A_inverse = invert(A)
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
