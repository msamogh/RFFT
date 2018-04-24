import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups

import json
import os

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from multilayer_perceptron import *
from parse import get_text_mask

ATHEISM = 'alt.atheism'
CHRISTIANITY = 'soc.religion.christian'
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data).toarray()
test_vectors = vectorizer.transform(newsgroups_test.data).toarray()


def load_hypothesis(
    vectorizer,
    annotations_map_file='annotations',
    text_files_base_dir='../text_tagger/Newsgroup-20'
):
    annotations_map = json.load(open(annotations_map_file, 'r'))
    text_files = [x for x in os.listdir(
        text_files_base_dir) if x.endswith('.txt')]
    A = np.zeros_like(len(vectorizer.vocabulary_)).astype(bool)
    for file in text_files:
        if file not in annotations_map:
            continue
        target = ATHEISM if ATHEISM in file else CHRISTIANITY
        index = int(file[file.index(target) + len(target):file.rindex('.')])
        A[index] = get_text_mask(
            vectorizer=vectorizer,
            annotations_map=annotations_map,
            text_files_base_dir=text_files_base_dir,
            text_filename=file
        )
    return A


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
