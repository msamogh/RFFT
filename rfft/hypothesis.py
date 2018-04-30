import random
import numpy as np


class Hypothesis(object):

    def __init__(
        self,
        A,
        weight=10,
        per_annotation=True,
        normalize=False
    ):
        self.A = A
        if per_annotation:
            num_annotations = len(np.where(A != 0)[0])
            self.weight = weight * num_annotations
        else:
            self.weight = weight
        self.normalize = normalize
        self.per_annotation = per_annotation

    @staticmethod
    def incrementally_sample(annotations, hypothesis_load_fn, mask_shape, increment=5, shuffle=True):
        if shuffle:
            random.shuffle(annotations)
        for i in range(0, len(annotations) + 1, increment):
            yield hypothesis_load_fn(mask_shape, annotations[:i])

    def __repr__(self):
        return 'Hypothesis: (weight = %f)' % self.weight
