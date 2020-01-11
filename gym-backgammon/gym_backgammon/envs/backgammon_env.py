import gym
import numpy as np
import time
from gym import error, spaces, utils

from gym_backgammon.game.game import Game, all_actions


class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, opponent, continuous=False):
        # Action and observation spaces.
        lower_bound = np.array([1, ] * 2 + [0, ] * 52)
        upper_bound = np.array([6, ] * 2 + [15, ] * 4 + [
            item for sublist in [[2, 15], ] * 24 for item in sublist])
        # The observation space is a 54 dimensional vector:
        # [dice1, dice2, white hit, black hit, white borne off, black borne off,
        #  point1 color, point1 count, point2 color, point2 count ... point24 color, point24 count]
        self.observation_space = spaces.Box(low=lower_bound, high=upper_bound,
                                            dtype=np.float32)

        # The action space is either continuous or discrete, ranging from 0 to 1728 for each possible
        # action of this tuple: (Type, Source, Target)
        if continuous:
            self.action_space = spaces.Box(low=np.array([-int((len(all_actions()) / 2) - 1)]),
                                           high=np.array(
                                               [int((len(all_actions()) / 2) - 1)]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(all_actions()))

        # Debug info.
        self.invalid_actions_taken = 0
        self.time_elapsed = 0

        # Game initialization.
        self.opponent = opponent
        self.game = Game(None, opponent)

    def step(self, action_index):
        if isinstance(self.action_space, spaces.Box):
            action_index += int(len(all_actions()) / 2)
            action_index = int(action_index)

        reward = self.game.player_turn(action_index)
        observation = self.game.get_observation()
        if reward == -10:
            self.invalid_actions_taken += 1
        if self.game.game_over() == 'white':
            done = True
            reward = 1
        elif self.game.game_over() == 'black':
            done = True
            reward = -1
        else:
            done = False
            reward = 0
        info = self.get_info()
        return observation, reward, done, info

    def reset(self):
        self.game = Game(self, self.opponent)
        self.invalid_actions_taken = 0
        self.time_elapsed = 0
        return self.game.get_observation()

    def render(self, mode='human'):
        if mode == 'human':
            self.game.print_game()

    def get_info(self):
        return {'time elapsed': time.time() - self.time_elapsed,
                'invalid actions taken': self.invalid_actions_taken}
