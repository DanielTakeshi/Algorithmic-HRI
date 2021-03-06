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
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def analyze_model_files(model_file_path):
    """ Analyze model files. I'll put this here for now. """
    directories = [x for x in os.listdir(model_file_path) if x[:5]=='stats']
    train_loss_accs = []
    valid_loss_accs = []
    test_loss_accs = []

    for d in directories:
        array = np.load(model_file_path + d)
        train_loss_accs.append((d, array['train_losses'][-1], array['train_accs'][-1]))
        valid_loss_accs.append((d, array['valid_losses'][-1], array['valid_accs'][-1]))
        test_loss_accs.append((d, array['test_loss_acc'][0], array['test_loss_acc'][1]))

    sorted_scores = sorted(test_loss_accs, key=lambda x:x[2])
    print("Sorted by test scores:")
    for item in sorted_scores:
        print(item)
    sorted_valids = sorted(valid_loss_accs, key=lambda x:x[2])
    print("Sorted by valid scores:")
    for item in sorted_valids:
        print(item)
    sorted_trains = sorted(train_loss_accs, key=lambda x:x[2])
    print("Sorted by train scores:")
    for item in sorted_trains:
        print(item)


if __name__ == "__main__":
    analyze_model_files('qnet_outputs/qnet_breakout_nature/')
