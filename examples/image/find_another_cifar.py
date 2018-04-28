from time import sleep
import sys
import os
import decoy_cifar
from rfft.multilayer_perceptron import *
from rfft.hypothesis import Hypothesis
import autograd.numpy as np
import pdb
import pickle
from parse import *

#sys.path.append('../../rfft/')
from rfft.tensorflow_perceptron import *
from rfft.tensorflow_cnn import *

#Xr, X, y, E, Xtr, Xt, yt, Et = decoy_cifar.generate_dataset()
X,y,Xt,yt = decoy_cifar.get_data()
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*3)
Xt = Xt.reshape(Xt.shape[0],Xt.shape[1]*Xt.shape[2]*3)

#Xt = Xt[:2000]
#yt = yt[:2000]

indices, hypothesis = decoy_cifar.load_hypothesis(X)
hypothesis.weight = 800000


def score_model(mlp):
	print('Train: {0}, Test: {1}'.format(mlp.score(X, y), mlp.score(Xt, yt)))



"""
print('Training f0')

if os.path.exists('models/1.pkl'):
	f0 = pickle.load(open('models/1.pkl', 'rb'))
else:
	f0 = MultilayerPerceptron()
	f0.fit(X, y, hypothesis=hypothesis,
		  num_epochs=16, always_include=indices)
	#pickle.dump(f0, open('models/1.pkl', 'wb'))
score_model(f0)
"""

#f0 = TensorflowPerceptron()
f0 = TensorflowCNN()
f0.fit(X,y,hypothesis.A,always_include=indices)

