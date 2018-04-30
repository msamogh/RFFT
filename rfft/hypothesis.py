import random
import numpy as np


class Hypothesis(object):
    """Class representing a hypothesis for a dataset."""


    def __init__(self,
                 A,
                 weight=10,
                 per_annotation=True):
        """Initialize a hypothesis.

        Args:
            A: A numpy array representing the binary mask matrix. The array is of shape (Number of samples x Length of each sample vector).
            weight: Weight of the "right reasons" term in the objective function. The weight can either be specified as a constant for the entire dataset or on a "per annotation" basis. In the latter case, the weight is multiplied with the number of non-zero masks to arrive at the final weight.
            per_annotation: True if weight specified is "per annotation".
        """
        self.A = A
        if per_annotation:
            self.num_annotations = len(set(np.where(A != 0)[0]))
            self.weight = weight * self.num_annotations
        else:
            self.weight = weight
        self.per_annotation = per_annotation


    def __repr__(self):
        """String representation of hypothesis metadata."""
        if self.per_annotation:
            return 'Hypothesis: weight={}, per_annotation={}, num_annotations={}'.format(self.weight, 
                                                                                         self.per_annotation,
                                                                                         self.num_annotations)
        return 'Hypothesis: weight={}, per_annotation={}'.format(self.weight, self.per_annotation)


    @staticmethod
    def incrementally_sample(annotations,
                             hypothesis_load_fn,
                             mask_shape,
                             increment=5,
                             shuffle=True,
                             **hypothesis_params):
        if shuffle:
            random.shuffle(annotations)
        for i in range(0, len(annotations) + 1, increment):
            yield hypothesis_load_fn(mask_shape, annotations[:i], **hypothesis_params)
