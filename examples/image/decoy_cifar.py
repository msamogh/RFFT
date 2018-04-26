import os
import cv2
import cPickle as cp
import numpy as np 
import random
from sklearn.model_selection import train_test_split
from PIL import Image
from rfft.hypothesis import Hypothesis
#from keras.datasets import cifar10
from parse import *
import sys
sys.path.append('../../rfft/')



def convert_to_pkl():
	files = os.listdir('data/decoy-cifar/images')
	labels = open('data/decoy-cifar/labels.txt').readlines()
	d = {}
	for i in range(len(labels)):
		line = labels[i].split()
		d[line[0]] = line[1]
	x = []
	y = []
	for i in range(len(files)):
		img = cv2.imread('data/decoy-cifar/images/'+files[i])
		x.append(img)
		y.append(d[files[i]])
	x = np.array(x)
	y = np.array(y)	
	y = y.astype(int)

	with open('data/decoy-cifar/data.pkl','wb') as ff:
		cp.dump([x,y],ff)



def Bern(p):
	return np.random.rand() < p


def augment(image, digit, randomize=False, mult=25, all_digits=range(10)):
	if randomize:
		return augment(image, np.random.choice(all_digits))

	img = image.copy()
	expl = np.zeros_like(img)

	fwd = [0, 1, 2, 3]
	rev = [-1, -2, -3, -4]
	dir1 = fwd if Bern(0.5) else rev
	dir2 = fwd if Bern(0.5) else rev
	for i in dir1:
		for j in dir2:
			img[i][j] = 255 - mult * digit
			expl[i][j] = 1

	return img.ravel(), expl.astype(bool).ravel()

def get_data():
	#convert_to_pkl()
	with open('data/decoy-cifar/data.pkl','rb') as ff:
		a = cp.load(ff)
	x = a[0]
	y = a[1]	

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
	x_train = x
	y_train = y

	with open('data/cifar-10-batches-py/test_batch','rb') as ff:
		a = cp.load(ff)
	x_test = a['data']
	y_test = np.array(a['labels'])

	x_test = x_test.reshape(x_test.shape[0],32,32,3)
	return x_train,y_train,x_test,y_test


def generate_dataset():

	X_raw, y, Xt_raw, yt = get_data()
	all_digits = list(set(y))

	X = []
	E = []
	Xt = []
	Et = []

	for image, digit in zip(X_raw, y):
		x, e = augment(image, digit, all_digits=all_digits)
		X.append(x)
		E.append(e)

	for image, digit in zip(Xt_raw, yt):
		x, e = augment(image, digit, all_digits=all_digits, randomize=True)
		Xt.append(x)
		Et.append(e)

	X = np.array(X)
	E = np.array(E)
	Xt = np.array(Xt)
	Et = np.array(Et)
	Xr = np.array([x.ravel() for x in X_raw])
	Xtr = np.array([x.ravel() for x in Xt_raw])

	return Xr, X, y, E, Xtr, Xt, yt, Et



def save_image_to_file(array, file_id, show=False):
	img = array.reshape((32, 32,3))
	img = Image.fromarray(img)
	img.save('tagging/decoy_mnist/' + str(file_id) + '.png')
	if show:
		img.show()


def generate_tagging_set(Xtr, size=20):
	indices = []
	for i in range(size):
		index = random.randint(0, len(Xtr))
		if index in indices:
			continue
		indices.append(index)
		save_image_to_file(Xtr[index], index, show=False)


def load_hypothesis(
	x,
	dirname='tagging/decoy_cifar',
	normalize=False
):
	xml_files = filter(
		lambda x: x.endswith('.xml'),
		os.listdir(dirname)
	)

	A = np.zeros_like(x).astype(bool)
	affected_indices = []
	for filename in xml_files:
		index = int(filename.split('.')[0])
		affected_indices.append(index)
		mask = get_image_mask(os.path.join(dirname, filename), (32,32,3))
		A[index] = mask.flatten()
	return (
		affected_indices,
		Hypothesis(
			A,
			normalize=normalize
		)
	)


if __name__ == '__main__':
	Xr, X, y, E, Xtr, Xt, yt, Et = generate_dataset()
	generate_tagging_set(X)
	print(load_hypothesis(X))
