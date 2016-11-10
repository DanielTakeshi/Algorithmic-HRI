"""
This is code to let humans play Atari games, via the Arcade Learning Environment
and the pygame library. This is based on code originally written by Ben
Goodrich, who modifed ale_python_test_pygame.py to provide an interactive
experience to allow humans to play.

I can also display RAM contents, current action, and reward if needed.

The keys are: 
    arrow keys -> up/down/left/right, 
    z -> fire button.

Right now, the human plays one Atari game for one episode. Some games can take a
while so I'll probably just call these as needed. Also, after every game, I can
double check if the is enough RAM to store all the screenshots. Store them on my
work station!

Games tested/verified/understood with notes (see other files for details):

1. Brekout
"""

import os
import errno
import re
import glob
import sys
import string
from ale_python_interface import ALEInterface
import pygame
import scipy.misc
from PIL import Image
import time
import numpy as np
import utilities
np.set_printoptions(edgeitems=20)

# For now keep this global and intact. It's ugly, but works.
key_action_tform_table = (
    0, #00000 none
    2, #00001 up
    5, #00010 down
    2, #00011 up/down (invalid)
    4, #00100 left
    7, #00101 up/left
    9, #00110 down/left
    7, #00111 up/down/left (invalid)
    3, #01000 right
    6, #01001 up/right
    8, #01010 down/right
    6, #01011 up/down/right (invalid)
    3, #01100 left/right (invalid)
    6, #01101 left/right/up (invalid)
    8, #01110 left/right/down (invalid)
    6, #01111 up/down/left/right (invalid)
    1, #10000 fire
    10, #10001 fire up
    13, #10010 fire down
    10, #10011 fire up/down (invalid)
    12, #10100 fire left
    15, #10101 fire up/left
    17, #10110 fire down/left
    15, #10111 fire up/down/left (invalid)
    11, #11000 fire right
    14, #11001 fire up/right
    16, #11010 fire down/right
    14, #11011 fire up/down/right (invalid)
    11, #11100 fire left/right (invalid)
    14, #11101 fire left/right/up (invalid)
    16, #11110 fire left/right/down (invalid)
    14  #11111 fire up/down/left/right (invalid)
)

class HumanPlayer(object):
    """ Represents a human playing one Atari game (for one episode). """

    def __init__(self, game="breakout.bin", rand_seed=1, output_dir="output/"):
        self.ale = ALEInterface()
        self.game = game
        self.actions = []
        self.rewards = []
        self.screenshots = []
        self.output_dir = output_dir

        # Set values of any flags (must call loadROM afterwards).
        self.ale.setInt("random_seed", rand_seed)
        self.ale.loadROM(self.game)

        # Other (perhaps interesting) stuff.
        self.random_seed = self.ale.getInt("random_seed")
        self.legal_actions = self.ale.getMinimalActionSet()
        print("Some info:\n-random_seed: {}\n-legal_actions: {}\n".format(
            self.random_seed, self.legal_actions))


    def play_and_save(self):
        """ The human player now plays!

        After playing it calls a post-processing step.
        """

        # TODO Figure out how to save checkpoints!

        # Both screen and game_surface are: <type 'pygame.Surface'>.
        pygame.init()
        screen = pygame.display.set_mode((320,420))
        game_surface = pygame.Surface(self.ale.getScreenDims()) # (160,210)
        pygame.display.flip()

        # Clock and some various statistics.
        clock = pygame.time.Clock()
        total_reward = 0.0 
        time_steps = 0
        start_time = time.time()

        # Iterate through game turns.
        while not self.ale.game_over():
            time_steps += 1

            # Get the keys human pressed.
            keys = 0
            pressed = pygame.key.get_pressed()
            keys |= pressed[pygame.K_UP]
            keys |= pressed[pygame.K_DOWN]  <<1
            keys |= pressed[pygame.K_LEFT]  <<2
            keys |= pressed[pygame.K_RIGHT] <<3
            keys |= pressed[pygame.K_z] <<4
            action = key_action_tform_table[keys]
            reward = self.ale.act(action);
            total_reward += reward

            # Clear screen and get pixels from atari on the surface via blit.
            screen.fill((0,0,0))
            rgb_image = self.ale.getScreenRGB().transpose(1,0,2) # Transposed!
            surface = pygame.surfarray.make_surface(rgb_image)
            screen.blit(pygame.transform.scale2x(surface), (0,0))
            pygame.display.flip() # Updates screen.

            # Process pygame event queue.
            exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit = True
                    break;
            if pressed[pygame.K_q]:
                exit = True
            if exit:
                break
            clock.tick(60.) # Higher number means game moves faster.

            self.rewards.append(reward)
            self.actions.append(action)
            self.screenshots.append(rgb_image)

        # Collect statistics and start post-processing.
        episode_frame_number = self.ale.getEpisodeFrameNumber()
        assert episode_frame_number == time_steps, \
            "episode_frame_number = {}, time_steps = {}".format(episode_frame_number, time_steps)
        end_time = time.time() - start_time
        print("Number of frames: {}".format(episode_frame_number))
        print("Game lasted {:.4f} seconds.".format(end_time))
        print("Total reward: {}.".format(total_reward))
        self.post_process()


    def post_process(self):
        """ Now save my game frames, actions, rewards, and checkpoints. 
        
        This creates three sets of files in self.output_dir/game_name:
        -> screenshots: contains screenshots (one screenshot per file)
        -> rewards: contains rewards (one file, one reward per line)
        -> actions: contains actions (one file, one action per line)
        -> checkpoints: contains checkpoints (one file, checkpoints)

        These are nested within self.output_dir/game_name according to the game
        ID, in files 'game_ID', which we determine here by looking at the
        numbers that exist.  The IDs should be listed in order as I increment
        them each time. Also, I pad the numbers to make it easier to read later
        once more games are added.
        """

        # A bit clumsy but it works if I call in correct directory.
        game_name = (self.game.split("/")[1]).split(".")[0]
        padding_digits = 4
        current_games = glob.glob(self.output_dir+ "/" +game_name+ "/game*")
        current_games.sort()
        next_index = 0

        if len(current_games) != 0:
            previous_index = re.findall('\d+', current_games[-1])[0]
            next_index = int(previous_index)+1

        game_id = string.zfill(next_index, padding_digits)
        head_dir = self.output_dir+ "/" +game_name+ "/game_" +game_id
        utilities.make_path_check_exists(head_dir)

        # TODO Check if rewards ever have floats but most are integers, I think.
        np.savetxt(head_dir+ "/actions.txt", self.actions, fmt="%d")
        np.savetxt(head_dir+ "/rewards.txt", self.rewards, fmt="%d")

        # TODO Check if I can extract max value automatically.
        utilities.make_path_check_exists(head_dir+ "/screenshots/")
        padding_digits_fr = 6 
        print("Now saving images ...")
        for (frame_index, scr) in enumerate(self.screenshots):
            if frame_index % 1000 == 0:
                print("Saved {} frames so far.".format(frame_index))
            # Be sure to transpose, otherwise it's "sideways." Assumes RGB.
            im = Image.fromarray(scr.transpose(1,0,2)) 
            pad_frame_num = string.zfill(frame_index, padding_digits_fr)
            im.save(head_dir+ "/screenshots/frame_" +pad_frame_num+ ".png")

        print("Done with post processing.")
