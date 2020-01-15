import os
import gym
import numpy as np
import keras
from gym_backgammon.game.game import all_actions


def continuous_space():
    return gym.spaces.Box(low=np.array([-int((len(all_actions()) / 2) - 1)]),
                          high=np.array(
                              [int((len(all_actions()) / 2) - 1)]),
                          dtype=np.float32)


def discrete_space():
    return gym.spaces.Discrete(len(all_actions()))


def normalize_action(action):
    return action / len(all_actions())


def denormalize_action(action):
    return int(action * len(all_actions()))


def get_latest_model(path='./models'):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    path = max(paths, key=os.path.getctime)
    print("Loading model ", path)
    return keras.models.load_model(path)