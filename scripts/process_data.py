"""
Given all of the output files as determined by the run.py and human_player.py
scripts, we run this to transform it entirely into a script for deep learning
software. For now just call it from the normal directory of the project. That
is, after running:

    python scripts/run.py roms/breakout.bin

From the home directory (doesn't have to be Breakout), we simply call:

    python scripts/process_data.py

I am probably going to use TensorFlow.
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
DIGITS_SCREENSHOTS = 6
PHI_LENGTH = 4
RESIZED_HEIGHT = 84
RESIZED_WIDTH = 84
CROP_OFFSET = 8 # Breakout


def frame_skipping(num_frames):
    """ Frame skips, done in a stochastic manner (2,3,4) just like OpenAI. AFTER
    this is done, we then use history (e.g. phi) based only on these indices.
    
    Args:
        num_frames: The total number of frames in a particular game.
    Returns:
        An array of subsampled indices. These are the indices that we want to
        use (and therefore not skip). The first few frames are stored but the
        actions will not be chosen until the 4th one (or whatever value
        corresponds to PHI_LENGTH).
    """
    nonskipped = []
    last_index = 0
    while last_index < num_frames:
        nonskipped.append(last_index)
        skip = random.randint(2,4)
        last_index += skip
    return np.array(nonskipped)


def downsample_single(game_name, frame_raw_color):
    """ Downsamples a single game frame.
    
    Make sure this is the same procedure as it is done with the code I use for
    DQN (and potentially other algorithms). This is also game-dependent since
    different games take different components of the screen. In general, this
    defaults to reducing (210,160,3)-dimensional numpy arrays into grayscales
    with shape (84,84).

    For now we will use spragnur's method with cropping; his scaling method is
    probably going to be worse.

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


def downsample_all(game_name, output_dir):
    """ Downsample all images and save images/actions in files.

    Specifically, this will create output files:
        - data/game_name/phis/{}.npy // One per 'phi'
        - data/game_name/targets.txt // Text file, the action per line.

    Args:
        game_name: The game we are using. Supported games: 'breakout'.
        output_dir: File path that ends at the game name.
    """
    current_games = glob.glob(output_dir+ "/game_*")
    current_games.sort()

    # We now save in one directory.
    t = 0 
    pad_t = DIGITS_SCREENSHOTS+1
    target_actions = []
    head_dir = "data_raw/" +game_name+ "/"
    utilities.make_path_check_exists(head_dir+ "phis/")

    for game_dir in current_games:
        start = t
        with open(game_dir+'/actions.txt', 'r') as f_actions, \
            open(game_dir+'/rewards.txt', 'r') as f_rewards:

            # First, get number of frames and subsample from them, skipping
            # intervals of 2, 3, and 4 (at random). Exclude starting few frames.
            actions_raw = np.array(f_actions.readlines())
            rewards_raw = np.array(f_rewards.readlines())
            assert len(actions_raw) == len(rewards_raw)
            num_frames = len(actions_raw)
            logger.info("Processing game_dir = {} with num_frames = {}".format(
                game_dir, num_frames))
            indices_to_use = frame_skipping(num_frames)

            # Our (PHI_LENGTH,84,84)-shape image for CNN input. phi[0] is the
            # oldest frame, and phi[PHI_LENGTH-1] is the most recent frame.
            phi = np.zeros((PHI_LENGTH,RESIZED_HEIGHT,RESIZED_WIDTH))

            # First few frames. These get ignored since we don't actually save
            # until later, but humans don't react fast enough so it works out.
            for (index,scr_index) in enumerate(indices_to_use[:PHI_LENGTH]):
                padded_index = str(scr_index).zfill(DIGITS_SCREENSHOTS)
                frame_raw = scipy.misc.imread(game_dir+ '/screenshots/frame_' +padded_index+ '.png')
                frame = downsample_single(game_name, frame_raw)
                phi[index] = frame

            # Now this will start saving 'phi'. Discard oldest image each time.
            for scr_index in indices_to_use:
                padded_index = str(scr_index).zfill(DIGITS_SCREENSHOTS)
                frame_raw = scipy.misc.imread(game_dir+ '/screenshots/frame_' +padded_index+ '.png')
                frame = downsample_single(game_name, frame_raw)

                # THANK GOD FOR NUMPY!!! (The irony: I am an atheist.)
                phi = np.roll(phi, shift=-1, axis=0) # Shift frames 'backward'.
                phi[PHI_LENGTH-1] = frame # Update the most recent frame.

                padded_t = str(t).zfill(pad_t)
                np.save(head_dir+ "phis/phi_" +padded_t, phi)
                target_actions.append(actions_raw[scr_index])
                t += 1
        logger.info("Finished processing game, number of phis/actions = {}.".format(t-start))

    np.savetxt(head_dir+ "actions_target.txt", 
               np.array(target_actions).astype('uint8'),
               fmt='%d')
    logger.info("Finished all games. Number of phis/actions = {}.".format(t))


def sample_indices(game_name, distribution):
    """
    Given the phis and targets (actions), sample them to balance data.

    This assumes Breakout! We assume we only care about 0,3, and 4 actions.
    We'll have to figure out someting about that later.

    Args:
        game_name: The game we are using. Supported games: 'breakout'.
        distribution: The desired class distribution. For Breakout, with three
            main actions, this means we ideally have [1/3,1/3,1/3]. Other
            distributions may be more desirable depending on the circumstances.
            NOTE! I do not actually use this right now ... for the sake of time
            I just want to get things running now so will assume balance.
    Returns:
        The set of indices for the actual data that we use for Deep Learning.
    """
    if (game_name != "breakout"):
        raise ValueError("game_name \'{}\' is not supported".format(game_name))
    head_dir = "data_raw/" +game_name+ "/"

    actions = np.loadtxt(head_dir+ "actions_target.txt")
    (unique_a, counts_a) = np.unique(actions, return_counts=True)
    logger.info("\nUnique actions: {},\ncorresponding counts: {}".format(unique_a,counts_a))

    # For Breakout, but noop, left, and right are consistent among games.
    indices_noop_all = np.where(actions == 0)[0]
    indices_left = np.where(actions == 4)[0]
    indices_right = np.where(actions == 3)[0]
    num_noop_touse = (len(indices_left)+len(indices_right))/2.
    indices_noop = np.random.choice(indices_noop_all, 
                                    size=num_noop_touse,
                                    replace= False)

    # This is the final set of indices to use for train/valid/test data.
    indices_all = np.concatenate((indices_noop, indices_left, indices_right))
    np.random.shuffle(indices_all)
    assert len(indices_all) <= len(actions)
    return indices_all


if __name__ == "__main__":
    """ A sequence of calls to get the data into a form usable by caffe. """

    game_name = "breakout"
    output_dir = "output/" +game_name

    # First: go through games I played (for game_name), save phis and actions.
    #downsample_all(game_name, output_dir) 

    # Second: subsample these to balance dataset.
    distribution = np.array([0.33,0.33,0.33])
    distribution /= np.sum(distribution)
    indices = sample_indices(game_name, distribution)

    # Third: rearrange them into stuff that can be used by TensorFlow.
    # TODO

    # A quick reminder at the end to save the log.
    logger.info("All done. Rename log.txt if you want to save the log." \
                " If you want to run this again, I suggest removing data_raw")
