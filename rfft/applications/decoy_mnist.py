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

# from lime import lime_image
# from skimage.segmentation import mark_boundaries

from rfft.experiment import Experiment
from rfft.experiment import ExperimentStatus
from rfft.experiment import ExperimentType
from rfft.experiment import Dataset

from rfft.multilayer_perceptron import MultilayerPerceptron
from rfft.hypothesis import Hypothesis
from rfft.applications.parse import get_image_mask_from_xml


ANNOTATIONS_DIR = 'tagging/decoy_mnist'


class DecoyMNIST(Experiment):


    def get_status(self):
        return self.status


    def domain(self):
        return ExperimentType.IMAGE


    def generate_dataset(self, cachefile='data/decoy-mnist.npz'):
        if cachefile and os.path.exists(cachefile):
            cache = np.load(cachefile)
            data = tuple([cache[f] for f in sorted(cache.files)])
        else:
            data = self._generate_dataset(os.path.dirname(cachefile))
            if cachefile:
                np.savez(cachefile, *data)
        self.Xr, self.X, self.y, self.E, self.Xtr, self.Xt, self.yt, self.Et = data
        self.status.dataset_generated = True


    def get_sample(self, dataset, idx):
        if not self.status.dataset_generated:
            raise AttributeError('Generate dataset before fetching samples.')
        if dataset == Dataset.TRAIN:
            return Xr[idx]
        elif dataset == Dataset.TEST:
            return Xtr[idx]


    def load_annotations(self, **hypothesis_params):
        annotation_files = [os.path.join(ANNOTATIONS_DIR, x)
                            for x in os.listdir(ANNOTATIONS_DIR)
                            if x.endswith('.npy')]

        A = np.zeros(self.X.shape).astype(bool)
        affected_indices = []
        for f in annotation_files:
            index = int(f.split('/')[-1].split('.')[0])
            mask = np.load(f)
            affected_indices.append(index)
            A[index] = mask

        self.affected_indices = affected_indices
        self.hypothesis = Hypothesis(A, **hypothesis_params)
        self.status.annotations_loaded = True


    def unload_annotations(self):
        self.hypothesis = None
        self.status.annotations_loaded = False


    def set_annotation(self, idx, annotation_json):
        value = annotation_json['value']
        indices = annotation_json['indices']
        length = annotation_json['length']

        initializer = {0: np.ones, 1: np.zeros}
        mask = initializer[value](length, dtype='uint8')
        mask[indices] = int(not value)
        np.save(os.path.join(ANNOTATIONS_DIR, str(idx)), mask)


    def get_annotation(self, idx):
        annotation_path = os.path.join(ANNOTATIONS_DIR, str(idx) + '.npy')
        if os.path.exists(annotation_path):
            return np.load(annotation_path)
        return None


    def delete_annotation(self, idx):
        annotation_path = os.path.join(ANNOTATIONS_DIR, str(idx) + '.npy')
        try:
            os.remove(annotation_path)
        except FileNotFoundError:
            pass

    
    def train(self, num_epochs=6):
        self.model = MultilayerPerceptron()
        self.model.fit(self.X,
                       self.y,
                       hypothesis=self.hypothesis,
                       num_epochs=num_epochs,
                       always_include=self.affected_indices)
        self.status.trained = True


    def explain(self, sample, **explanation_params):
        if not self.status.trained:
            raise AttributeError('You must have trained the model to be able to generate explanations.')
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image,
                                                 self.model,
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)
        temp, mask = explanation.get_image_and_mask(240, positive_only=True, num_features=5, hide_rest=True)
        masked_image = mark_boundaries(temp / 2 + 0.5, mask)
        return masked_image

    
    def score_model(self):
        return (
            self.model.score(self.X, self.y),
            self.model.score(self.Xt, self.yt)
        )
    

    def download_mnist(self, datadir):
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
                return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows,
                                                                                     cols)
        
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


    def Bern(self, p):
        return np.random.rand() < p


    def augment(self, image, digit, randomize=False, mult=25, all_digits=range(10)):
        if randomize:
            return self.augment(image, np.random.choice(all_digits))
        
        img = image.copy()
        expl = np.zeros_like(img)
        
        fwd = [0, 1, 2, 3]
        rev = [-1, -2, -3, -4]
        dir1 = fwd if self.Bern(0.5) else rev
        dir2 = fwd if self.Bern(0.5) else rev
        for i in dir1:
            for j in dir2:
                img[i][j] = 255 - mult * digit
                expl[i][j] = 1
        
        return img.ravel(), expl.astype(bool).ravel()


    def _generate_dataset(self, datadir):
        X_raw, y, Xt_raw, yt = self.download_mnist(datadir)
        all_digits = list(set(y))
        
        X = []
        E = []
        Xt = []
        Et = []
        
        for image, digit in zip(X_raw, y):
            x, e = self.augment(image, digit, all_digits=all_digits)
            X.append(x)
            E.append(e)
        
        for image, digit in zip(Xt_raw, yt):
            x, e = self.augment(image, digit, all_digits=all_digits, randomize=True)
            Xt.append(x)
            Et.append(e)
        
        X = np.array(X)
        E = np.array(E)
        Xt = np.array(Xt)
        Et = np.array(Et)
        Xr = np.array([x.ravel() for x in X_raw])
        Xtr = np.array([x.ravel() for x in Xt_raw])
        
        return Xr, X, y, E, Xtr, Xt, yt, Et


    def save_image_to_file(self, array, file_id, show=False):
        img = array.reshape((28, 28))
        img = Image.fromarray(img)
        img.save('tagging/decoy_mnist/' + str(file_id) + '.png')
        if show:
            img.show()


    def generate_tagging_set(self, Xtr, size=20):
        indices = []
        for i in range(size):
            index = npr.randint(0, len(Xtr))
            if index in indices:
                continue
            indices.append(index)
            self.save_image_to_file(Xtr[index], index)


if __name__ == '__main__':
    print('Training with annotations')
    decoy_mnist = DecoyMNIST()
    decoy_mnist.generate_dataset()
    decoy_mnist.load_annotations(weight=10, per_annotation=True)
    decoy_mnist.train(num_epochs=1)
    print(decoy_mnist.score_model())

    print('Training without annotations')
    decoy_mnist.unload_annotations()
    decoy_mnist.train(num_epochs=2)
    print(decoy_mnist.score_model())
