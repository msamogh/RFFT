from sklearn.feature_extraction.text import TfidfVectorizer
from xml.etree import ElementTree as elem

import os
import autograd.numpy as np


def get_image_mask(bbox_path, image_size, valid_class_names=[]):
    masked_img = np.zeros(image_size, dtype="uint8")

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


def get_text_mask(text_files_base_dir, annotations):
    for file in annotations:
        text = open(os.path.join(text_files_base_dir, file), 'r').read()
        for a in annotations:
            start, end = a.split(',')
            