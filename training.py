import statistics

import gym
import keras
import numpy as np
import os
from gym_backgammon.agents.random import RandomAgent
from gym_backgammon.game.game import all_actions
from keras import *
from keras.layers.advanced_activations import ReLU

# We want only won games
from keras.optimizers import Adam

score_requirement = 0


def build_env(opponent):
    env = gym.make("gym_backgammon:backgammon-v0",
                   opponent=opponent)
    return env


def play_training(env, training_games, model):
    scores = []
    mistakes = []
    accepted_games = []
    for game_index in range(training_games):
        score = 0
        game_memory = []
        previous_observation = env.reset()
        done = False
        while not done:
            action = model.predict(np.array([previous_observation]))
            observation, reward, done, info = env.step(action)
            game_memory.append([previous_observation, action])
            previous_observation = observation
            score += reward
            if done:
                break
        scores.append(score)
        mistakes.append(info['invalid actions taken']/len(game_memory))
        if score > score_requirement:
            accepted_games.append([game_memory, info['invalid actions taken']])
    # Sort by how many mistakes were made during the game
    accepted_games.sort(key=lambda x: x[1])
    # Remove x% with the most mistakes
    accepted_games = accepted_games[0:round(len(accepted_games)*0.7)]
    # Unzip into list of observation->decision
    training_data = [play for game in accepted_games for play in game[0]]
    print('Training win rate', sum(score for score in scores if score == 1) / len(scores))
    print('Training avg mistake rate', round(statistics.mean(mistakes), 2))
    return training_data


def build_model(input_size, output_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_dim=input_size))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Dense(output_size))
    model.add(keras.layers.ReLU(max_value=863, negative_slope=0.0, threshold=-863.0))
    model.compile(loss='msle', optimizer=keras.optimizers.Adam(lr=0.01))
    return model


def train_model(training_data, model, epochs):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model.fit(X, y, epochs=epochs, verbose=2, batch_size=8)
    return model


def play_test(model, env, test_games):
    scores = []
    mistakes = []
    games = []
    for each_game in range(test_games):
        game_memory = []
        previous_observation = env.reset()
        done = False
        score = 0
        while not done:
            action = model.predict(np.array([previous_observation]))
            game_memory.append([previous_observation, action])
            new_observation, reward, done, info = env.step(action)
            previous_observation = new_observation
            score += reward
            if done:
                break
        scores.append(score)
        mistakes.append(info['invalid actions taken']/len(game_memory))
        games.append(game_memory)
    print('Test win rate', sum(score for score in scores if score == 1) / len(scores))
    print('Test avg mistake rate', round(statistics.mean(mistakes), 2))


def get_latest_model(path='./models'):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    path = max(paths, key=os.path.getctime)
    print("Loading model ", path)
    return keras.models.load_model(path)


def train_against_random(training_games, test_games, model, epochs):
    env = build_env(opponent=RandomAgent(action_space=continuous_space()))
    training_data = play_training(env=env, training_games=training_games, model=model)
    trained_model = train_model(training_data=training_data, model=model, epochs=epochs)
    # play_test(model=trained_model, env=env, test_games=test_games)
    return trained_model


def validate_against_random(trained_model, test_games):
    env = build_env(opponent=RandomAgent(action_space=continuous_space()))
    play_test(model=trained_model, env=env, test_games=test_games)


def play_random_games(training_games, env):
    training_data = []
    accepted_scores = []
    scores = []
    mistakes = []
    for game_index in range(training_games):
        score = 0
        game_memory = []
        previous_observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            game_memory.append([previous_observation, action])
            previous_observation = observation
            score += reward
            if done:
                break
        scores.append(score)
        mistakes.append(info['invalid actions taken']/len(game_memory))
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], data[1]])
    print('random win rate', sum(score for score in scores if score == 1) / len(scores))
    print('avg mistake rate', round(statistics.mean(mistakes), 2))

    return training_data


def continuous_space():
    return gym.spaces.Box(low=np.array([-int((len(all_actions()) / 2) - 1)]),
                   high=np.array(
                       [int((len(all_actions()) / 2) - 1)]),
                   dtype=np.float32)