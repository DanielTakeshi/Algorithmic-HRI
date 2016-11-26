"""
Given all of the output files as determined by the run.py and human_player.py
scripts, we run this to transform it entirely into a script for deep learning
software. For now just call it from the normal directory of the project. That
is, after running:

    python scripts/run.py roms/breakout.bin

From the home directory (doesn't have to be Breakout), we simply call:

    python scripts/process_data.py

If I want to delete the data, just delete whatever this file creates (don't
delete the raw data files from the games!).  I am probably going to use
TensorFlow, so this should be a good learning experience.
"""

import cv2
import glob
import logging
import numpy as np
import os
from PIL import Image
import random
import scipy
from scipy import misc
import sys
import utilities
np.set_printoptions(edgeitems=20)

# -----------------------------------------#
# A bunch of options for logging.          #
# I'm not sure the best place to put this? #
# -----------------------------------------#
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------------- #
# Other global variables #
# ---------------------- #
DIGITS = 6
PAD_T = 7 # One more than screenshots
PHI_LENGTH = 4
RESIZED_HEIGHT = 84
RESIZED_WIDTH = 84
CROP_OFFSET = 8 # Breakout


def frame_skipping(num_frames, openai_style=False):
    """ This subsamples indices which we later use for frame skipping.
    
    There are two options. One is to use OpenAI-style and skip 1, 2, or 3 frames
    at random (thus the actions are repeated for 2, 3, or 4 frames). The second,
    default way, is to use what DQN did originally, with actions repeated 4
    times. We still need the previous index to stop flickering, but we don't do
    that here (it's done elsewhere in the code).
    
    Args:
        num_frames: The total number of frames in a particular game.

    Returns:
        An array of subsampled indices. These are the indices that we want to
        use (and therefore not skip). Note that for DQN, we only return every
        4th index.
    """
    nonskipped = []
    frame_skip = 4
    last_index = frame_skip
    assert last_index < num_frames

    while last_index < num_frames:
        nonskipped.append(last_index)
        skip = frame_skip
        if openai_style:
            skip = random.randint(2,4)
        last_index += skip
    return np.array(nonskipped)


def downsample_single(game_name, frame_raw_color):
    """ Downsamples a single game frame.
    
    Make sure this is the same procedure as it is done with the code I use for
    DQN (and potentially other algorithms). This is also game-dependent since
    different games take different components of the screen. In general, this
    defaults to reducing (210,160,3)-dimensional numpy arrays into grayscales
    with shape (84,84).  For now we will use spragnur's method with cropping;
    his scaling method is probably going to be worse.

    Args:
        game_name: The game we are using. Supported games: 'breakout'.
        frame_raw: The raw frame, should be (210,160,3).
    """
    if (game_name != "breakout"):
        raise ValueError("game_name \'{}\' is not supported".format(game_name))
    assert frame_raw_color.shape == (210,160,3)
    frame_raw = utilities.rgb2gray(frame_raw_color)

    resize_height = int(round(210. * RESIZED_WIDTH/160.))
    resized = cv2.resize(frame_raw, 
                         (RESIZED_WIDTH,resize_height),
                         interpolation=cv2.INTER_LINEAR)

    crop_y_cutoff = resize_height - CROP_OFFSET - RESIZED_HEIGHT
    cropped = resized[crop_y_cutoff:(crop_y_cutoff+RESIZED_HEIGHT):]
    return cropped


def downsample_all(game_name, output_dir, raw_data_dir):
    """ Downsample all images and save images/actions in files.

    Specifically, this will create output files:
        - data_raw/game_name/phis/{}.npy // One per 'phi', i.e. four frames.
        - data_raw/game_name/actions_target.txt // Text file, the action per line.

    Args:
        game_name: The game we are using. Supported games: 'breakout'.
        output_dir: File path that ends at the game name.
        raw_data_dir: Where the downsampled files are stored! It should be
            data_raw/game_name/.
    """
    current_games = glob.glob(output_dir+ "/game_*")
    current_games.sort()

    # We now save in one directory.
    t = 0 
    target_actions = []
    utilities.make_path_check_exists(raw_data_dir+ "/phis/")

    for game_dir in current_games:
        start = t
        with open(game_dir+'/actions.txt', 'r') as f_actions, \
            open(game_dir+'/rewards.txt', 'r') as f_rewards:

            # First, get number of frames and subsample from them, skipping
            # intervals. Depends on whether we use OpenAI-style or DQN.
            actions_raw = np.array(f_actions.readlines())
            rewards_raw = np.array(f_rewards.readlines())
            assert len(actions_raw) == len(rewards_raw)
            num_frames = len(actions_raw)
            logger.info("Processing game_dir = {} with num_frames = {}".format(
                game_dir, num_frames))
            indices_to_use = frame_skipping(num_frames, openai_style=False)
            assert len(indices_to_use) > PHI_LENGTH

            # Our (PHI_LENGTH,84,84)-shape image for CNN input. phi[0] is the
            # oldest frame, and phi[PHI_LENGTH-1] is the most recent frame.
            phi = np.zeros((PHI_LENGTH,RESIZED_HEIGHT,RESIZED_WIDTH))

            # First few frames. These get ignored since we don't actually save
            # until later, but humans don't react fast enough so it works out.
            for (index,scr_index) in enumerate(indices_to_use[:PHI_LENGTH]):
                index_prev = str(scr_index-1).zfill(DIGITS)
                index_curr = str(scr_index).zfill(DIGITS)
                frame_raw_prev = scipy.misc.imread(
                    game_dir+ '/screenshots/frame_' +index_prev+ '.png')
                frame_raw_curr = scipy.misc.imread(
                    game_dir+ '/screenshots/frame_' +index_curr+ '.png')
                frame_raw = np.maximum(frame_raw_prev, frame_raw_curr)
                frame = downsample_single(game_name, frame_raw)
                phi[index] = frame

            # Save phis and discard oldest frame each time. We don't use
            # enumerate as we don't need the index of scr_index anymore.
            for scr_index in indices_to_use[PHI_LENGTH:]:
                index_prev = str(scr_index-1).zfill(DIGITS)
                index_curr = str(scr_index).zfill(DIGITS)
                frame_raw_prev = scipy.misc.imread(
                    game_dir+ '/screenshots/frame_' +index_prev+ '.png')
                frame_raw_curr = scipy.misc.imread(
                    game_dir+ '/screenshots/frame_' +index_curr+ '.png')
                frame_raw = np.maximum(frame_raw_prev, frame_raw_curr)
                frame = downsample_single(game_name, frame_raw)

                # THANK GOD FOR NUMPY!!! (The irony: I am an atheist.)
                phi = np.roll(phi, shift=-1, axis=0) # Shift frames 'backward'.
                phi[PHI_LENGTH-1] = frame # Update the most recent frame.

                padded_t = str(t).zfill(PAD_T)
                np.save(raw_data_dir+ "/phis/phi_" +padded_t, phi)
                target_actions.append(actions_raw[scr_index])
                t += 1
        logger.info("Finished processing game, number of phis/actions = {}.".format(t-start))

    np.savetxt(raw_data_dir+ "/actions_target.txt", 
               np.array(target_actions).astype('uint8'),
               fmt='%d')
    logger.info("Finished all games. Number of phis/actions = {}.".format(t))


def sample_indices(game_name, raw_data_dir):
    """
    Given the phis and targets (actions), sample them to balance data.

    This assumes Breakout! We assume we only care about 0, 3, and 4 actions.
    We'll have to figure out someting about that later.

    Args:
        game_name: The game we are using. Supported games: 'breakout'.
        raw_data_dir: Where the downsampled files are stored.

    Returns:
        The (shuffled) indices for the actual data we use for Deep Learning.
    """
    if (game_name != "breakout"):
        raise ValueError("game_name \'{}\' is not supported".format(game_name))

    actions = np.loadtxt(raw_data_dir+ "/actions_target.txt")
    (unique_a, counts_a) = np.unique(actions, return_counts=True)
    logger.info("\nUnique actions: {},\ncorresponding counts: {}".format(unique_a,counts_a))

    # For Breakout, but noop, left, and right are consistent among games.
    indices_noop_all = np.where(actions == 0)[0]
    indices_left = np.where(actions == 4)[0]
    indices_right = np.where(actions == 3)[0]
    num_noop_touse = (len(indices_left)+len(indices_right))/2
    logger.info("num_noop_touse = {}".format(num_noop_touse))
    indices_noop = np.random.choice(indices_noop_all, 
                                    size=num_noop_touse,
                                    replace= False)

    # This is the final set of indices to use for train/valid/test data.
    indices_all = np.concatenate((indices_noop, indices_left, indices_right))
    np.random.shuffle(indices_all)
    assert len(indices_all) <= len(actions)
    return indices_all


def create_test_valid_train(game_name, indices, ratio, raw_data_dir, final_data_dir):
    """ Given shuffled indices, create train/valid/test data/label files.

    Don't forget to use (N,depth,height,width) as the shape, and to have actions
    be 0,1,2,... instead of 0,4,3,...

    Args:
        game_name: The name of the game (only breakout supported).
        indices: The indices within the actual data (i.e. phi index) which
            we use for training, validation, and test. It is ALREADY
            shuffled so we should just take the corresponding proportions.
        ratio: An array with desired proportions of train/valid/test data.
        raw_data_dir: Where we can access the phis and actions.
        final_data_dir: The "final" data directory where we store the data in
            numpy arrays, to be used as input to theano/lasagne.
    """
    if np.sum(ratio) != 1:
        logger.info("Warning, np.sum(ratio)!=1, currently re-normalizing ...")
        ratio = ratio / np.sum(ratio)
    N = len(indices)
    actions = np.loadtxt(raw_data_dir+ "/actions_target.txt")

    # Maps non-consecutive actions into consecutive numerical range (0,1,2,...).
    a_map = {}
    if (game_name == "breakout"):
        a_map = {0:0, 4:1, 3:2}
    else:
        raise ValueError("game_name not supported")

    # We need the padded versions, with PAD_T, to refer to the phis.
    indices_padded = [str(t).zfill(PAD_T) for t in indices]
    train_indices = indices_padded[ : int(N*ratio[0])]
    valid_indices = indices_padded[int(N*ratio[0]) : int(N*(ratio[0]+ratio[1]))]
    test_indices  = indices_padded[int(N*(ratio[0]+ratio[1])) : ]
    assert len(valid_indices) < len(test_indices) < len(train_indices)

    # We now know dimensions of the datasets to use. HEIGHT BEFORE WIDTH!
    train_data = np.zeros((len(train_indices),PHI_LENGTH,RESIZED_HEIGHT,RESIZED_WIDTH))
    valid_data = np.zeros((len(valid_indices),PHI_LENGTH,RESIZED_HEIGHT,RESIZED_WIDTH))
    test_data  = np.zeros((len(test_indices) ,PHI_LENGTH,RESIZED_HEIGHT,RESIZED_WIDTH))
    train_labels = np.zeros(len(train_indices))
    valid_labels = np.zeros(len(valid_indices))
    test_labels  = np.zeros(len(test_indices))

    for (i,v) in enumerate(train_indices):
        train_data[i] = np.load(raw_data_dir+ "/phis/phi_" +v+ ".npy")
        train_labels[i] = a_map[ actions[int(v)] ]

    for (i,v) in enumerate(valid_indices):
        valid_data[i] = np.load(raw_data_dir+ "/phis/phi_" +v+ ".npy")
        valid_labels[i] = a_map[ actions[int(v)] ]

    for (i,v) in enumerate(test_indices):
        test_data[i] = np.load(raw_data_dir+ "/phis/phi_" +v+ ".npy")
        test_labels[i] = a_map[ actions[int(v)] ]

    logger.info("Now saved information in {train,valid,test}.{data,labels} ...")
    logger.info("train_data,labels.shape: {} and {}".format(train_data.shape,train_labels.shape))
    logger.info("valid_data,labels.shape: {} and {}".format(valid_data.shape,valid_labels.shape))
    logger.info("test_data,labels.shape: {} and {}".format(test_data.shape,test_labels.shape))

    utilities.make_path_check_exists(final_data_dir)
    np.save(final_data_dir+ "/train.data", train_data)
    np.save(final_data_dir+ "/valid.data", valid_data)
    np.save(final_data_dir+ "/test.data", test_data)
    np.save(final_data_dir+ "/train.labels", train_labels)
    np.save(final_data_dir+ "/valid.labels", valid_labels)
    np.save(final_data_dir+ "/test.labels", test_labels)


if __name__ == "__main__":
    """ 
    A sequence of calls to get the data into a form usable by TensorFlow.
    First, we go through all the games and save phis and actions. Second, we get
    a set of balanced indices 
    """

    game_name = "breakout"
    output_dir = "output/" +game_name
    raw_data_dir = "data_raw/" +game_name
    final_data_dir = "final_data/" +game_name

    # First: go through games I played (for game_name), save phis and actions.
    downsample_all(game_name, output_dir, raw_data_dir) 

    # Second: subsample these to balance dataset and save into new arrays.
    indices = sample_indices(game_name, raw_data_dir)
    ratio = np.array([0.76, 0.04, 0.2])
    create_test_valid_train(game_name, indices, ratio, raw_data_dir, final_data_dir)

    # A quick reminder at the end to save the log.
    logger.info("All done. Rename log.txt if you want to save the log." \
                " If you want to run this again, I suggest removing data_raw")
