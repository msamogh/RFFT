import numpy as np


class Perceptron(object):

    def grad_explain(self, X, **kwargs):
        yhats = self.predict(X)
        coefs = self.input_gradients(X, **kwargs)
        return [LocalLinearExplanation(X[i], yhats[i], coefs[i]) for i in range(len(X))]

    def largest_gradient_mask(self, X, cutoff=0.67, **kwargs):
        grads = self.input_gradients(X, **kwargs)
        return np.array([np.abs(g) > cutoff * np.abs(g).max() for g in grads])


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=np.uint8)