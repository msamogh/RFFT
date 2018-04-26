from time import sleep
import sys
import os
sys.path.append('rrr')
import decoy_cifar
from rfft.multilayer_perceptron import *
from rfft.hypothesis import Hypothesis
import autograd.numpy as np
import pdb
import pickle
from parse import *


Xr, X, y, E, Xtr, Xt, yt, Et = decoy_cifar.generate_dataset()
indices, hypothesis = decoy_cifar.load_hypothesis(X)
hypothesis.weight = 800000


def score_model(mlp):
	print('Train: {0}, Test: {1}'.format(mlp.score(X, y), mlp.score(Xt, yt)))


print('Training f0')

if os.path.exists('models/1.pkl'):
	f0 = pickle.load(open('models/1.pkl', 'rb'))
else:
	f0 = MultilayerPerceptron()
	f0.fit(X, y, hypothesis=hypothesis,
		  num_epochs=16, always_include=indices)
	pickle.dump(f0, open('models/1.pkl', 'wb'))
score_model(f0)

if os.path.exists('models/2.pkl'):
	f0 = pickle.load(open('models/2.pkl', 'rb'))
else:
	f0 = MultilayerPerceptron()
	f0.fit(X, y, num_epochs=16, always_include=indices)
	pickle.dump(f0, open('models/2.pkl', 'wb'))
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
	f1.fit(X, y, hypothesis=Hypothesis(M0), num_epochs=16)
	score_model(f1)

	print('Training f2')
	M1 = f1.largest_gradient_mask(X)
	f2.fit(X, y, hypothesis=Hypothesis(M0 + M1), num_epochs=16)
	score_model(f2)

	print('Training f3')
	M2 = f2.largest_gradient_mask(X)
	f3.fit(X, y, hypothesis=Hypothesis(M0 + M1 + M2), num_epochs=16)
	score_model(f3)

	print('Training f4')
	M3 = f3.largest_gradient_mask(X)
	f4.fit(X, y, hypothesis=Hypothesis(M0 + M1 + M2 + M3), num_epochs=16)

	print('Training f5')
	M4 = f4.largest_gradient_mask(X)
	f5.fit(X, y, hypothesis=Hypothesis(M0 + M1 + M2 + M3 + M4), num_epochs=16)

	params = [f.params for f in [f0, f1, f2, f3, f4, f5, f6]]
	filename = 'data/decoy_cifar_fae_{}'.format(l2)
	pickle.dump(params, open(filename, 'wb'))
