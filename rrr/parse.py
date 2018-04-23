from sklearn.feature_extraction.text import TfidfVectorizer
from xml.etree import ElementTree as elem

import json
import os
import autograd.numpy as np


def get_image_mask(bbox_path, image_size, valid_class_names=[]):
    masked_img = np.zeros(image_size, dtype='uint8')

    root = elem.parse(bbox_path).getroot()
    for obj in root.findall('object'):
        if valid_class_names:
            if obj.find('name').text not in valid_class_names:
                continue
        bbox = obj.find('bndbox')

        def get_coord(name):
            return int(bbox.find(name).text)

        masked_img[
            get_coord('ymin'):get_coord('ymax'),
            get_coord('xmin'):get_coord('xmax')
        ] = 1
    return masked_img


def parent_word(word_endings, i):
    word_endings = [-1] + word_endings
    for x in range(len(word_endings)):
        if i > word_endings[x] and i <= word_endings[x + 1]:
            return x


def get_text_mask(
    n_words,
    annotations_map,
    text_files_base_dir,
    text_filename
):
    text_file_path = os.path.join(text_files_base_dir, text_filename)
    text = open(text_file_path, 'r').read()

    word_endings = [i for i in range(len(text)) if text[i] == ' ']

    ignore_words = np.zeros(n_words, dtype='uint8')
    annotations = annotations_map[text_filename]
    for ann in annotations:
        start, end = ann.split(',')
        start_word = parent_word(word_endings, int(start))
        end_word = parent_word(word_endings, int(end))
        ignore_words[start_word:end_word + 1] = 1

    return ignore_words
