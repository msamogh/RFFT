from autograd.misc import flatten

import numpy as np


def l2_norm(params):
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)
