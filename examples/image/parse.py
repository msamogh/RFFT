from sklearn.feature_extraction.text import TfidfVectorizer
from xml.etree import ElementTree as elem

from PIL import Image

import json
import os
import autograd.numpy as np


def get_image_mask_from_xml(bbox_path, image_size, valid_class_names=[]):
    """Return mask vector from LabelImg annotation.

    Args:
        bbox_path: Path to XML file of annotation.
        image_size: (width, height) tuple of image dimensions.
        valid_class_names: If empty, considers all class names of annotations. Otherwise, only those that are part of the list.

    Returns:
        1D Numpy array of the masked image.
    """
    masked_img = np.ones(image_size, dtype='uint8')

    root = elem.parse(bbox_path).getroot()
    annotations = root.findall('object')
    if valid_class_names:
        annotations = filter(lambda x: x.find('name').text in valid_class_names, annotations)

    for obj in annotations:
        bbox = obj.find('bndbox')
        get_coord = lambda name: int(bbox.find(name).text)
        masked_img[
            get_coord('ymin'):get_coord('ymax'),
            get_coord('xmin'):get_coord('xmax')
        ] = 0
    return masked_img


def get_image_mask_from_ui(image_path, image_size, mask_color, n_channels=3):
    """Return mask vector from WHITEBox Studio annotator tool.

    Args:
        image_path: Path to annotated image.
        image_size: (width, height) tuple of image dimensions.
        mask_color: Tuple of color of the mask used while coloring.
        n_channels: Number of channels in the image.

    Returns:
        1D Numpy array of the masked image.
    """
    masked_img = np.zeros(image_size, dtype='uint8')

    img = Image.open(image_path).load()
    img = np.asarray(img, dtype='int32')
    img = img.reshape(-1, n_channels)
    ignore = np.where(img == mask_color)

    masked_img[ignore] = 1
    return masked_img


def parent_word(word_endings, i):
    print(word_endings)
    print(i)
    word_endings = [-1] + word_endings
    for x in range(len(word_endings) - 1):
        if i > word_endings[x] and i <= word_endings[x + 1]:
            return x


def get_text_mask(
    vectorizer,
    annotations_map,
    text_files_base_dir,
    text_filename
):
    text_file_path = os.path.join(text_files_base_dir, text_filename)
    text = open(text_file_path, 'r').read()

    word_endings = [i for i in range(len(text)) if text[i] == ' ']

    ignore_words = np.zeros(len(vectorizer.vocabulary_), dtype='uint8')
    annotations = annotations_map[text_filename]
    for ann in annotations:
        start, end = ann.split(',')
        start_word = parent_word(word_endings, int(start))
        end_word = parent_word(word_endings, int(end))
        words = text.split()[start_word:end_word + 1]
        indices = []
        for w in words:
            if w in vectorizer.vocabulary_:
                indices.append(vectorizer.vocabulary_[w])
        ignore_words[indices] = 1
    return ignore_words
