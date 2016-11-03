"""
Several utilities. I guess put random methods here.
"""

import errno
import os

def make_sure_path_exists(path):
    """ From Stack Overflow, to be safe on creating directories. """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
