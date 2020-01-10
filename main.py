#
# # This allows to force running on cpu for measurement
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# #This removes  tf warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from gym_backgammon.agents.random import RandomAgent
from gym_backgammon.game.game import all_actions
import gym

env = gym.make("gym_backgammon:backgammon-v0", opponent=RandomAgent(action_space=gym.spaces.Discrete(len(all_actions()))))
while True:
    observation, reward, done, info = env.step(env.action_space.sample())
