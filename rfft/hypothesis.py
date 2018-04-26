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


    @staticmethod
    def incrementally_sample(annotations, hypothesis_load_fn, mask_shape, increment=5):
        for i in range(0, len(annotations), increment):
            yield hypothesis_load_fn(mask_shape, annotations[:i + 1])


    def __repr__(self):
        return 'Hypothesis: (weight = %f)' % self.weight
