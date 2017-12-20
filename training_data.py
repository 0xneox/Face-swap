import numpy
from image_augmentation import random_transform
from image_augmentation import random_warp

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }
