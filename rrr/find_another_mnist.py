from time import sleep
import sys
sys.path.append('rrr')
import decoy_mnist
from multilayer_perceptron import *
import numpy as np
import pdb
import pickle

Xr, X, y, E, Xtr, Xt, yt, Et = decoy_mnist.generate_dataset()
hypothesis = decoy_mnist.load_hypothesis(X)
hypothesis.weight = 10000

def score_model(mlp):
    print('Train: {0}, Test: {1}'.format(mlp.score(X, y), mlp.score(Xt, yt)))


print('Training f0')
f0 = MultilayerPerceptron()
f0.fit(X, y, hypotheses=[hypothesis], num_epochs=2)
score_model(f0)

f0 = MultilayerPerceptron()
f0.fit(X, y, hypotheses=[], num_epochs=2)
score_model(f0)

for l2 in [1000, 10000, 100000]:
    print('l2 =', l2)
    hypothesis.weight = l2 * 100
    f1 = MultilayerPerceptron(l2_grads=l2)
    f2 = MultilayerPerceptron(l2_grads=l2)
    f3 = MultilayerPerceptron(l2_grads=l2)
    f4 = MultilayerPerceptron(l2_grads=l2)
    f5 = MultilayerPerceptron(l2_grads=l2)
    f6 = MultilayerPerceptron(l2_grads=l2)

    print('Training f1')
    M0 = f0.largest_gradient_mask(X)
    f1.fit(X, y, M0, hypotheses=[hypothesis])
    score_model(f1)

    print('Training f2')
    M1 = f1.largest_gradient_mask(X)
    f2.fit(X, y, M0 + M1, hypotheses=[hypothesis])
    score_model(f2)

    print('Training f3')
    M2 = f2.largest_gradient_mask(X)
    f3.fit(X, y, M0 + M1 + M2, hypotheses=[hypothesis])
    score_model(f3)

    # print('Training f4')
    # M3 = f3.largest_gradient_mask(X)
    # f4.fit(X, y, M0 + M1 + M2 + M3, hypotheses=[hypothesis])

    # print('Training f5')
    # M4 = f4.largest_gradient_mask(X)
    # f5.fit(X, y, M0 + M1 + M2 + M3 + M4, hypotheses=[hypothesis])

    # print('Training f6')
    # M5 = f5.largest_gradient_mask(X)
    # f6.fit(X, y, M0 + M1 + M2 + M3 + M4 + M5, hypotheses=[hypothesis])

    params = [f.params for f in [f0, f1, f2, f3, f4, f5, f6]]
    filename = 'data/decoy_mnist_fae_{}'.format(l2)
    pickle.dump(params, open(filename, 'wb'))
