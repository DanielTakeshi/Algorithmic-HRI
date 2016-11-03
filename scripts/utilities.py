"""
Several utilities. I guess put random methods here.
"""

import errno
import os

def make_path_check_exists(path):
    """ From Stack Overflow, use this to create directories. """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
