from xml.etree import ElementTree as elem

import numpy as np


def get_mask(bbox_path, image_size):
    masked_img = np.ones(image_size, dtype="uint8")

    root = elem.parse(bbox_path).getroot()
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')

        def get_coord(name):
            return int(bbox.find(name).text)

        masked_img[
            get_coord('xmin'):get_coord('xmax'),
            get_coord('ymin'):get_coord('ymax')
        ] = 0
    return masked_img
