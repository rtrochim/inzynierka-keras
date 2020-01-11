import gym
import keras
import numpy as np
import os
from gym_backgammon.agents.random import RandomAgent
from gym_backgammon.game.game import all_actions
from keras.activations import relu
from keras import *
# We want only won games
from keras.layers import Activation, Lambda

score_requirement = 1


def build_env(opponent):
    env = gym.make("gym_backgammon:backgammon-v0",
                   opponent=opponent)
    return env


def play_training(env, training_games, model):
    training_data = []
    accepted_scores = []
    scores = []
    for game_index in range(training_games):
        score = 0
        game_memory = []
        previous_observation = env.reset()
        done = False
        while not done:
            action = round(np.array(model.predict(np.array([previous_observation]))).item(0))
            observation, reward, done, info = env.step(action)
            game_memory.append([previous_observation, action])
            previous_observation = observation
            score += reward
            if done:
                break
        scores.append(score)
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], data[1], env.get_info()['invalid actions taken']])

    print('Training Win rate', sum(score for score in scores if score == 1) / len(scores))
    return training_data


def build_model(input_size, output_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=input_size, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(output_size))
    model.add(Activation(lambda x: relu(x, max_value=1727)))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(training_data, model, epochs):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model.fit(X, y, epochs=epochs, verbose=0)
    return model


def play_test(model, env, test_games):
    scores = []
    for each_game in range(test_games):
        previous_observation = env.reset()
        done = False
        score = 0
        while not done:
            action = round(np.array(model.predict(np.array([previous_observation]))).item(0))
            new_observation, reward, done, info = env.step(action)
            previous_observation = new_observation
            score += reward
            if done:
                break
        scores.append(score)
    print('Test Win rate', sum(score for score in scores if score == 1) / len(scores))


def get_latest_model(path='./models'):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    path = max(paths, key=os.path.getctime)
    print("Loading model ", path)
    return keras.models.load_model(path)


def train_against_random(training_games, test_games, model, epochs):
    env = build_env(opponent=RandomAgent(action_space=gym.spaces.Discrete(len(all_actions()))))
    training_data = play_training(env=env, training_games=training_games, model=model)
    trained_model = train_model(training_data=training_data, model=model, epochs=epochs)
    # play_test(model=trained_model, env=env, test_games=test_games)
    return trained_model


def validate_against_random(trained_model, test_games):
    env = build_env(opponent=RandomAgent(action_space=gym.spaces.Discrete(len(all_actions()))))
    play_test(model=trained_model, env=env, test_games=test_games)


# def play_random_games(training_games, env):
#     training_data = []
#     accepted_scores = []
#     scores = []
#     for game_index in range(training_games):
#         score = 0
#         game_memory = []
#         previous_observation = env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             game_memory.append([previous_observation, action])
#             previous_observation = observation
#             score += reward
#             if done:
#                 break
#         scores.append(score)
#         if score >= score_requirement:
#             accepted_scores.append(score)
#             for data in game_memory:
#                 training_data.append([data[0], data[1], env.get_info()['invalid actions taken']])
#     print('Initial random Win rate', sum(score for score in scores if score == 1) / len(scores))
#     return training_data

