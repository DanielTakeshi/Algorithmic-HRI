"""
Several utilities. I guess put random methods here.
"""

import errno
import numpy as np
import os

def make_path_check_exists(path):
    """ From StackOverflow, use this to create directories. """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def rgb2gray(rgb):
    """ Also from StackOverflow. I'm not totally comfortable with
    multidimensional np.dot operations, but this seems to work well. Assumes
    that rgb.shape = (210,160,3) but really it sums over the (210,160)
    components, weighted according to these tuned values.
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
