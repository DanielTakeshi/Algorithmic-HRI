# Algorithmic-HRI

My final project for CS 294-115. The main idea is to use DQN but boost it with human data. Specifically, when the Q-Learning policy asks us to take a random action, we will usually (but not always, more on that later) take an action that is instead chosen by a previously trained classifier which maps from sequences of game frames (i.e. states) to actions. The classifier was trained on HUMAN data from my own gameplay, which should presumably be better than selecting random actions. The revised DQN code is in my other repository, which was forked from [spragnur's version](https://github.com/DanielTakeshi/deep_q_rl).

Make sure to follow these steps carefully!!! There's a lot of places in the pipeline where it's easy to make a mistake. The high level idea of this code is as follows: first the human plays Atari games to gather data in the form of (game frames, action), where the game frames are really four "consecutive", non-skipped frames. Then we train a neural network to classify from game frames to actions. Then we finally plug it inside a DQN learner, so that during the exploration phase, instead of playing random actions, we play the action that the classifier tells us to play based on the current game state. The hope is that this will boost early performance.

More detailed documentation now follows.

# Gathering the Data and Training the Classifier

(1) Play games and gather data using `scripts/human_player.py`. Each time that script is called, the human can play one "round" of an Atari game (e.g. for Breakout this would be one game with 5 lives). Then after the script is done, the saved game frames, actions, and rewards are saved in appropriate files within directories named after the game itself and after the game ID (i.e. from 0 to however many games one plays). This will save ALL game frames in full RGB format, so make sure there's enough memory for this.

(2) Next, run the preprocessing script `scripts/process_data.py` to generate the dataset. We perform frame-skipping to skip every 2, 3, or 4 frames, which is used to increase stochasticity and also to "spread out" the influence of game frames, as four frames is a bit too few for there to be detectable motion. Once we subsample, we will get frames that may correspond to time steps t, t+3, t+6, t+8, t+12, t+16, t+19, etc. We take sequences of four (or `PHI_LENGTH`) game frames, i.e. [t, t+3, t+6, t+8], then [t+3, t+6, t+8, t+12], etc., with *overlapping* game frames. These four frames are downsampled to (84,84) each and assembled into a (4,84,84)-dimensional numpy array with the most recent frame at index 3. The corresponding action chosen is the one that was played on the last frame of the four. That is the target. In the code, I refer to the sequences of `PHI_LENGTH` game frames as `phi` or `states`.

**Update**: I changed this code so that it only skips by 4. The reason is that spragnur's code will always skip by 4, so I want to ensure that the neural network classifier I train is going to have inputs that are consistent with each other. Note also that due to (1), we will never be in a case where we see a phi which may have two frames from the end of an episode and two frames from the next episode. spragnur needed to have a special case for that, but here we don't!

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

This documentation is [for the code here](https://github.com/DanielTakeshi/deep_q_rl), but I am putting things here to keep it centralized.

(1) The revised DQN code is heavily based on spragnur's code, with the following changes:

- The `run_nips.py`, `run_nature.py`, and `launcher.py` scripts now contain settings for my new human net. In particular, `launcher.py` contains a call to the `HumanQNetwork` class to contain this new network trained on human data (saved in `.npz` format). This network will, of course, not be modified in any form during the training run. Also, the launcher contains a dictionary which will map from consecutive integer numbers to another set of integers, which *then* get mapped to the appropriate actions as determined from `ale.getMinimalActionSet()`. For details on what action indices correspond to what, see [here](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/java-agent/code/src/ale/io/Actions.java) or the ALE documentation manual. In particular, RIGHT=3 and LEFT=4, but in my neural network code, I put in RIGHT=2 and LEFT=1. Watch out! Also, spragnur's code handles the action mapping by using the list from `ale.getMinimalActionSet()` which peforms the mapping, e.g. for Breakout it's [0 1 3 4]. Anyway, this "Human Q-Network" gets passed into the NeuralAgent class so it can call it as necessary.
- `make_net.py` is a new script which creates the different neural network variants. I did this because ideally the human nets and the Q-network nets have the same structure.
- `human_q_net.py` is another new script which creates the human network class and contains code for initializing the weights as soon as it is created.
- The `q_network.py` code has been simplified to put the neural network code construction inside `make_net.py`.
- The human network gets incorporated in the DQN code via `ale_agent.py`, specifically inside the crucial `_choose_action` method. There's documentation there which should make it clear. In particular, one important point is that I am using epsilon = 0.1 as the probability for a random action throughout the ENTIRE training run. For the other "90 percent", that value will degrade linearly from all human network actions to all Q-network actions. I keep the random action percentage at 0.1 so that actions such as FIRE, which only occur rarely (in Breakout, it should happen exactly 5 times per game unless one is pressing FIRE for no reason) can be played in practice, preventing such situations from having to be hard-coded.

The above *should* list exhaustively the changes I made. I hope I didn't forget anything.

(2) The default code from spragnur produces a `results.csv` file, of which the first five lines may look like:

```
epoch,num_episodes,total_reward,reward_per_epoch,mean_q
1,63,62,0.984126984127,0.084294614559
2,34,0,0.0,0.0975618274463
3,62,67,1.08064516129,0.128791995049
4,58,84,1.44827586207,0.113572772667
```

In each row, the first number is the epoch. In supervised learning, an epoch is one full pass over the dataset, but in reinforcement learning, an epoch is ambiguous. In his code, spragnur defined it as 50000 "steps" (in the NIPS version), so we'll just refer to an epoch as consisting of some total number of steps. The `ale_experiment.py` code gets called to run the experiment using the `run` method. That method contains a call to a `run_epoch` method (followed by a testing epoch) with a while loop that executes until we've exceeded the number of steps. The while loop calls its `run_episode` method with two parameters: the number of steps left, and whether it's testing/training, and the `run_episode` method initializes the episode and then keeps playing it (calls its own `_step` method) until the max steps has exceeded (or more commonly, if the ALE indicates it's game over).

(3) Experiment protocol: for now, just use `results.csv` from spragnur's code. However, I'll want to better understand it and see if there are other ways I can plot results. Ideally I will use the NATURE code, not the NIPS code, but for time constraints I may have to use the NIPS code.

Reminders:

- Keep separate `deep_q_rl` and `deel_q_rl_old` directories. The former is my current version, forked from spragnur's code. The latter is spragnur's code, untouched from his most recent version (though he hasn't updated in a long time). 
