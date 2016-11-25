# Algorithmic-HRI

My final project for CS 294-115. The main idea is to use DQN but boost it with human data. Specifically, when the Q-Learning policy asks us to take a random action, we will usually (but not always, more on that later) take an action that is instead chosen by a previously trained classifier which maps from sequences of game frames (i.e. states) to actions. The classifier was trained on HUMAN data from my own gameplay, which should presumably be better than selecting random actions. The revised DQN code is in my other repository, which was forked from spragnur's version: https://github.com/DanielTakeshi/deep_q_rl.

Make sure to follow these steps carefully!!! There's a lot of places in the pipeline where it's easy to make a mistake. The high level idea of this code is as follows: first the human plays Atari games to gather data in the form of (game frames, action), where the game frames are really four "consecutive", non-skipped frames. Then we train a neural network to classify from game frames to actions. Then we finally plug it inside a DQN learner, so that during the exploration phase, instead of playing random actions, we play the action that the classifier tells us to play based on the current game state. The hope is that this will boost early performance.

More detailed documentation now follows.

# Gathering the Data and Training the Classifier

(1) Play games and gather data using `scripts/human_player.py`. Each time that script is called, the human can play one "round" of an Atari game (e.g. for Breakout this would be one game with 5 lives). Then after the script is done, the saved game frames, actions, and rewards are saved in appropriate files within directories named after the game itself and after the game ID (i.e. from 0 to however many games one plays). This will save ALL game frames in full RGB format, so make sure there's enough memory for this.

(2) Next, run the preprocessing script `scripts/process_data.py` to generate the dataset. We perform frame-skipping to skip every 2, 3, or 4 frames, which is used to increase stochasticity and also to "spread out" the influence of game frames, as four frames is a bit too few for there to be detectable motion. Once we subsample, we will get frames that may correspond to time steps t, t+3, t+6, t+8, t+12, t+16, t+19, etc. We take sequences of four (or `PHI_LENGTH`) game frames, i.e. [t, t+3, t+6, t+8], then [t+3, t+6, t+8, t+12], etc., with *overlapping* game frames. These four frames are downsampled to (84,84) each and assembled into a (4,84,84)-dimensional numpy array with the most recent frame at index 3. The corresponding action chosen is the one that was played on the last frame of the four. That is the target. In the code, I refer to the sequences of `PHI_LENGTH` game frames as `phi` or `states`.

(3) After saving all of these phis and actions, we then sub-sample the data to balance it out among actions. For instance, in Breakout and most other games, the 0 action (NO-OP) will be by far the most common one. It is more desirable to have a balanced data. In theory it would be better to keep all the data and simply subsample or have a loss function which weighs things appropriately. However, for coding reasons, it is easier to just subsample the NO-OP actions to make them even with LEFT and RIGHT actions in terms of frequency. We also map actions to be in the range (0,1,2,...), so they are numbered consecutively. This is all still in the `scripts/process_data.py` scripts, and upon conclusion, we should get data that looks like this:

```
$ls -lh final_data/breakout/
total 7.1G
-rw-rw-r-- 1 daniel daniel 1.5G Nov 19 13:15 test.data.npy
-rw-rw-r-- 1 daniel daniel  53K Nov 19 13:15 test.labels.npy
-rw-rw-r-- 1 daniel daniel 5.4G Nov 19 13:15 train.data.npy
-rw-rw-r-- 1 daniel daniel 200K Nov 19 13:15 train.labels.npy
-rw-rw-r-- 1 daniel daniel 289M Nov 19 13:15 valid.data.npy
-rw-rw-r-- 1 daniel daniel  11K Nov 19 13:15 valid.labels.npy
```

All of the data files above have shape (N,4,84,84), where height comes before width, but usually the distinction shouldn't matter as I plan to keep the pixels square. We can now finally use this data as input to a neural network.

(4) We use theano/lasagne as the neural network library. In `scripts/q_network_actions.py`, we insert lasagne code (based on spragnur's DQN code) to replicate the network style from the DQN papers, except that we add in the softmax. We then train the network based on code from the lasagne documentation. It takes some time to understand how this works but it's not too bad once one gets used to theano/lasagne. I use this script and different hyperparameter settings to decide on the best network to save. Save the network which performs best on the validation set. This is what should get loaded inside the DQN code. Also, use this script to investigate the mapping from phis to actions, as that would provide a lot of intuition to people.

The TL;DR for the above is that:

- Use `scripts/human_data.py` (and `scripts/run.py`) to play the game.
- Use `scripts/process_data.py` to create the dataset.
- Use `scripts/q_network_actions.py` to train and save a classifier.

The other script in this directory is `scripts/utilities.py` but that's for containing random supporting methods.

# Incorporating the Classifier into DQN

IN PROGRESS
