import sys

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad, elementwise_grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam

from .utils import one_hot
from .hypothesis import Hypothesis
# from local_linear_explanation import LocalLinearExplanation

# Adapted from https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
# with modifications made such that we have a first-class MLP object
# and such that our loss function includes an explanation penalty.


def relu(inputs):
    return np.maximum(inputs, 0.)


def l2_norm(params):
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def feed_forward(params, inputs, nonlinearity=relu):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    # outputs log probabilities
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def input_gradients(params, y=None, scale='log'):
    # log probabilities or probabilities
    if scale is 'log':
        def p(x): return feed_forward(params, x)
    else:
        def p(x): return np.exp(feed_forward(params, x))

    # max, sum, or individual y
    if y is None:
        y = 'sum' if scale is 'log' else 'max'
    if y is 'sum':
        p2 = p
    elif y is 'max':
        def p2(x): return np.max(p(x), axis=1)
    else:
        def p2(x): return p(x)[:, y]

    return elementwise_grad(p2)


def l2_irrelevant_input_gradients(params, X, A, **kwargs):
    return l2_norm(input_gradients(params, **kwargs)(X)[A])


def init_random_params(scale, layer_sizes, rs=npr):
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


class MultilayerPerceptron:

    @classmethod
    def from_params(klass, params):
        mlp = klass()
        mlp.params = params
        return mlp

    def __init__(
        self,
        layers=(50, 30),
        l2_params=0.0001,
        l2_grads=0.0001,
        input_preprocessor=None
    ):
        self.l2_params = l2_params
        self.l2_grads = l2_grads
        self.layers = list(layers)
        self.input_preprocessor = input_preprocessor

    def predict_proba(self, inputs):
        if self.input_preprocessor:
            inputs = self.input_preprocessor(inputs)
        return np.exp(feed_forward(self.params, inputs))

    def predict(self, inputs):
        return np.argmax(feed_forward(self.params, inputs), axis=1)

    def score(self, inputs, targets):
        return np.mean(self.predict(inputs) == targets)

    def input_gradients(self, X, **kwargs):
        if 'scale' not in kwargs:
            kwargs['scale'] = None  # default to non-log probs
        return input_gradients(self.params, **kwargs)(X.astype(np.float32))

    def grad_explain(self, X, **kwargs):
        yhats = self.predict(X)
        coefs = self.input_gradients(X, **kwargs)
        return [LocalLinearExplanation(X[i], yhats[i], coefs[i]) for i in range(len(X))]

    def largest_gradient_mask(self, X, cutoff=0.67, **kwargs):
        grads = self.input_gradients(X, **kwargs)
        return np.array([np.abs(g) > cutoff * np.abs(g).max() for g in grads])

    def fit(
        self,
        inputs,
        targets,
        hypothesis=None,
        normalize=False,
        num_epochs=64,
        batch_size=256,
        step_size=0.001,
        rs=npr,
        always_include=None,
        nonlinearity=relu,
        verbose=False,
        callback=None,
        show_progress_every=None,
        **input_grad_kwargs
    ):
        X = inputs.astype(np.float32)
        y = one_hot(targets)
        params = init_random_params(
            0.1, [X.shape[1]] + self.layers + [y.shape[1]], rs=rs)

        batch_size = min(batch_size, X.shape[0])
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        input_grads = input_gradients(
            params,
            **input_grad_kwargs
        )(inputs)

        def batch_indices(iteration):
            idx = iteration % num_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)

        def objective(params, iteration):

            def update_progress_bar(i, num_iters):
                percent = i * 20 / num_iters
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" %
                                 ('=' * int(percent), 5 * percent))
                sys.stdout.flush()

            if show_progress_every is not None and iteration % show_progress_every == 0:
                update_progress_bar(iteration, num_epochs * num_batches)

            idx = batch_indices(iteration)
            Xi = X[idx]
            yi = y[idx]
            if hypothesis is not None:
                A = hypothesis.A
            else:
                A = np.zeros_like(inputs).astype(bool)
            Ai = A[idx]

            if always_include is not None:
                Ai = np.vstack((A[always_include], Ai))
                Xi = np.vstack((X[always_include], Xi))
                yi = np.vstack((y[always_include], yi))

            if normalize:
                lenX = max(1., float(len(Xi)))
            else:
                lenX = 1.

            crossentropy = - \
                np.sum(feed_forward(params, Xi, nonlinearity) * yi) / lenX
            if hypothesis is not None:
                rightreasons = hypothesis.weight * \
                    l2_norm(input_gradients(
                        params, **input_grad_kwargs)(Xi)[Ai])
            else:
                rightreasons = 0 * \
                    l2_norm(input_gradients(
                        params, **input_grad_kwargs)(Xi)[Ai])
            smallparams = self.l2_params * l2_norm(params)

            if iteration % show_progress_every == 0 and verbose:
                sys.stdout.write('Iteration={}, crossentropy={}, rightreasons={}'.format(
                    iteration, crossentropy._value, rightreasons._value))
                sys.stdout.flush()
            return crossentropy + rightreasons + smallparams

        self.params = adam(grad(objective), params,
                           step_size=step_size, num_iters=num_epochs * num_batches)
