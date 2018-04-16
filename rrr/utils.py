from autograd.misc import flatten

import autograd.numpy as np


def l2_norm(params):
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

