from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from rfft.experiment import Experiment
from rfft.experiment import ExperimentStatus
from rfft.experiment import ExperimentType
from rfft.experiment import Dataset

from rfft.multilayer_perceptron import MultilayerPerceptron
from rfft.hypothesis import Hypothesis

from parse import get_text_mask

import json
import os
import sys
import random

import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics


class NewsGroup(Experiment):
    
    def domain():
        return ExperimentType.TEXT
    
    def generate_dataset(self):
        ATHEISM = 'alt.atheism'
        CHRISTIANITY = 'soc.religion.christian'
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(
            subset='train', categories=categories)
        newsgroups_test = fetch_20newsgroups(
            subset='test', categories=categories)
        class_names = ['atheism', 'christian']
        
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=False)
        train_vectors = vectorizer.fit_transform(
            newsgroups_train.data).toarray()
        test_vectors = vectorizer.transform(newsgroups_test.data).toarray()
        self.newsgroups_train = newsgroups_train
        self.newsgroups_test = newsgroups_test
        self.vectorizer = vectorizer
        self.X, self.y, self.Xt, self.yt = train_vectors, newsgroups_train.target, test_vectors, \
                                           newsgroups_test.target
        self.status.dataset_generated = True
    
    def get_sample(dataset, idx):
        if not self.status.dataset_generated:
            raise AttributeError('Generate dataset before fetching samples.')
        if dataset == Dataset.TRAIN:
            return self.newsgroups_train.data[idx]
        elif dataset == Dataset.TEST:
            return self.newsgroups_test.data[idx]
    
    def load_annotations(self, dirname='tagging/newsgroup', **hypothesis_params):
        txt_files = [os.path.join(dirname, x)
                     for x in os.listdir(dirname) if x.endswith('.txt')]
        
        A = np.zeros(self.X.shape).astype(bool)
        affected_indices = []
        
        for filepath in txt_files:
            index = int(filepath.split('/')[-1].split('.')[0])
            file_content = open(filepath, 'rb').read()
            original_file_content = self.newsgroups_train.data[index]
            
            original_feature = self.X[index]
            file_feature = self.vectorizer.transform([file_content]).toarray()
            file_feature = np.squeeze(file_feature)
            mask_indices = []
            mask = np.ones(self.X.shape[1], dtype='uint8')
            for i in range(len(original_feature)):
                if file_feature[i] != original_feature[i]:
                    mask_indices.append(i)  # words to be ignored
            mask[mask_indices] = 0
            A[index] = mask
            affected_indices.append(index)
        
        # Mitigate tf-idf effects
        
        # for i in range(self.X.shape[0]):
        #     for j in range(self.X.shape[1]):
        #         if j in mask_indices:
        #             A[i][j] = 0
        
        self.affected_indices = affected_indices
        self.hypothesis = Hypothesis(A, **hypothesis_params)
        self.status.annotations_loaded = True
    
    def unload_annotations(self):
        self.hypothesis = None
        self.status.annotations_loaded = False
    
    def delete_annotation(self, idx):
        pass
    
    def get_annotation(self, idx):
        pass
    
    def set_annotation(self, idx):
        pass
    
    def get_status(self):
        return self.status
    
    def train(self, num_epochs=6):
        self.model = MultilayerPerceptron()
        self.model.fit(self.X,
                       self.y,
                       hypothesis=self.hypothesis,
                       num_epochs=num_epochs,
                       always_include=self.affected_indices,
                       show_progress_every=5)
    
    def explain(self, sample):
        pass
    
    def score_model(self):
        print('Train: {0}, Test: {1}'.format(
            self.model.score(self.X, self.y), self.model.score(self.Xt, self.yt)))
    
    def save_to_text_file(self, file_id):
        if not os.path.exists('tagging'):
            os.mkdir('tagging')
        if not os.path.exists('tagging/newsgroup'):
            os.mkdir('tagging/newsgroup')
        
        with open('tagging/newsgroup/' + str(file_id) + '.txt', 'w') as fout:
            fout.write(self.newsgroups_train.data[file_id])
    
    def generate_tagging_set(self, size=20):
        indices = []
        for i in range(size):
            index = random.randint(0, len(self.X))
            if index in indices:
                continue
            indices.append(index)
            self.save_to_text_file(index)


if __name__ == '__main__':
    print('Training with annotations')
    news = NewsGroup()
    news.generate_dataset()
    # news.generate_tagging_set()
    news.load_annotations(weight=10, per_annotation=True)
    news.train(num_epochs=6)
    print(news.score_model())
    
    print("Without hypothesis")
    news.hypothesis = None
    news.train(num_epochs=6)
    print(news.score_model())

"""

A = load_hypothesis(vectorizer)
mlp = MultilayerPerceptron(input_preprocessor=lambda x: x.toarray())
mlp.fit(train_vectors, newsgroups_train.target, hypothesis=A, num_epochs=20)

# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(train_vectors, newsgroups_train.target)
pred = mlp.predict(test_vectors)
# pred = rf.predict(test_vectors)
print(sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary'))


c = make_pipeline(vectorizer, mlp)
explainer = LimeTextExplainer(class_names=newsgroups_train.target_names)
"""
