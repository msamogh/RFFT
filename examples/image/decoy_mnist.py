from __future__ import absolute_import
from __future__ import print_function

from PIL import Image

import autograd.numpy.random as npr

import os
import gzip
import pickle
import struct
import array
import autograd.numpy as np
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlopen

from rfft.multilayer_perceptron import MultilayerPerceptron
from rfft.hypothesis import Hypothesis
from parse import get_image_mask


def download_mnist(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        if not os.path.exists(os.path.join(datadir, filename)):
            try:
                urlretrieve(base_url + filename,
                            os.path.join(datadir, filename))
            except:
                urlopen(base_url + filename, os.path.join(datadir, filename))

    train_images = parse_images(os.path.join(
        datadir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(os.path.join(
        datadir, 'train-labels-idx1-ubyte.gz'))
    test_images = parse_images(os.path.join(
        datadir, 't10k-images-idx3-ubyte.gz'))
    test_labels = parse_labels(os.path.join(
        datadir, 't10k-labels-idx1-ubyte.gz'))

    return train_images, train_labels, test_images, test_labels


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


def _generate_dataset(datadir):
    X_raw, y, Xt_raw, yt = download_mnist(datadir)
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


def generate_dataset(cachefile='data/decoy-mnist.npz'):
    if cachefile and os.path.exists(cachefile):
        cache = np.load(cachefile)
        data = tuple([cache[f] for f in sorted(cache.files)])
    else:
        data = _generate_dataset(os.path.dirname(cachefile))
        if cachefile:
            np.savez(cachefile, *data)
    return data


def save_image_to_file(array, file_id, show=False):
    img = array.reshape((28, 28))
    img = Image.fromarray(img)
    img.save('tagging/decoy_mnist/' + str(file_id) + '.png')
    if show:
        img.show()


def generate_tagging_set(Xtr, size=20):
    indices = []
    for i in range(size):
        index = npr.randint(0, len(Xtr))
        if index in indices:
            continue
        indices.append(index)
        save_image_to_file(Xtr[index], index)


def load_annotations(
    mask_shape,
    dirname='tagging/decoy_mnist',
    normalize=False
):
    xml_files = list(filter(
        lambda x: x.endswith('.xml'),
        os.listdir(dirname)
    ))
    xml_files = list(map(
        lambda x: os.path.join(dirname, x),
        xml_files
    ))
    return load_hypothesis(mask_shape, xml_files, normalize)


def load_hypothesis(
    mask_shape,
    xml_files,
    normalize=False
):
    A = np.zeros(mask_shape).astype(bool)
    affected_indices = []

    for filepath in xml_files:
        index = int(filepath.split('/')[-1].split('.')[0])
        mask = get_image_mask(filepath, (28, 28)).flatten()

        affected_indices.append(index)
        A[index] = mask
    return (
        affected_indices,
        Hypothesis(
            A,
            normalize=normalize
        )
    )


if __name__ == '__main__':
    Xr, X, y, E, Xtr, Xt, yt, Et = generate_dataset()
    dirname = 'tagging/decoy_mnist'
    xml_files = list(filter(
        lambda x: x.endswith('.xml'),
        os.listdir(dirname)
    ))
    xml_files = list(map(
        lambda x: os.path.join(dirname, x),
        xml_files
    ))
    indices, hypothesis = load_hypothesis(X.shape, xml_files[:30])
    hypothesis.weight = 10000000

    def score_model(mlp):
        print('Train: {0}, Test: {1}'.format(
            mlp.score(X, y), mlp.score(Xt, yt)))

    print('Training with annotations')
    if False and os.path.exists('models/1.pkl'):
        f0 = pickle.load(open('models/1.pkl', 'rb'))
    else:
        f0 = MultilayerPerceptron()
        f0.fit(X, y, hypothesis=hypothesis,
               num_epochs=16, always_include=indices, verbose=True)
        pickle.dump(f0, open('models/1.pkl', 'wb'))
    score_model(f0)

    print('Training without annotations')
    if os.path.exists('models/2.pkl'):
        f0 = pickle.load(open('models/2.pkl', 'rb'))
    else:
        f0 = MultilayerPerceptron()
        f0.fit(X, y, num_epochs=16, always_include=indices)
        pickle.dump(f0, open('models/2.pkl', 'wb'))
    score_model(f0)
