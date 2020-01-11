#
# # This allows to force running on cpu for measurement
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
import os
#This removes  tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from gym_backgammon.agents.random import RandomAgent
from gym_backgammon.game.game import all_actions
import gym

# import matplotlib.pyplot as plt

# import timeit
# print(timeit.timeit(run, number=1))


env = gym.make("gym_backgammon:backgammon-v0",
               opponent=RandomAgent(action_space=gym.spaces.Discrete(len(all_actions()))))
score_requirement = -550
initial_games = 100000


def model_data_preparation():
    training_data = []
    accepted_scores = []
    scores = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break
        scores.append(score)
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], data[1]])
        env.reset()
    print(accepted_scores)
    print('Average Score:', sum(scores) / len(scores))
    return training_data


def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='relu'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model = build_model(input_size=len(X[0]), output_size=1)
    model.fit(X, y, epochs=10)
    return model


def play_trained(trained_model):
    scores = []
    choices = []
    for each_game in range(100):
        previous_observation = []
        done = False
        score = 0
        while not done:
            if len(previous_observation) == 0:
                action = env.action_space.sample()
            else:
                action = round(np.array(trained_model.predict(np.array([previous_observation]))).item(0))
            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            previous_observation = new_observation
            score += reward
            if done:
                break
        env.reset()
        scores.append(score)

    print(scores)
    print('Average Score:', sum(scores) / len(scores))


training_data = model_data_preparation()
trained_model = train_model(training_data=training_data)
play_trained(trained_model)
