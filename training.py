import statistics
from utilities import *
from copy import deepcopy
from gym_backgammon.agents.random import RandomAgent

score_requirement = 0


def build_env(opponent):
    env = gym.make("gym_backgammon:backgammon-v0",
                   opponent=opponent)
    return env


def play_training(env, training_games, model):
    scores = []
    valid_actions = 0
    training_data = []
    mistakes = [0] * training_games
    for game_index in range(training_games):
        score = 0
        previous_observation = env.reset()
        done = False
        game_length = 0
        while not done:
            game_length += 1
            action = model.predict(np.array([previous_observation]))
            observation, reward, done, info = env.step(denormalize_action(action))
            if reward > 0:
                training_data.append([deepcopy(previous_observation), deepcopy(action)])
                valid_actions += 1
                score += reward
            else:
                mistakes[game_index] += 1
            previous_observation = observation
            if done:
                mistakes[game_index] /= game_length
                break
        scores.append(score)
    # games.sort(key=lambda x: x[1], reverse=True)
    # games = games[0:round(0.8 * len(games))]
    # training_data = [item for game, score in games for item in game]
    if len(mistakes) and len(scores):
        print('Training avg score', statistics.mean(scores))
        print('Training avg mistake rate', statistics.mean(mistakes))
    print('Training valid actions', valid_actions)
    print('Training data length', len(training_data))
    return training_data


def build_model(input_size, output_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=input_size, activation='relu'))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(output_size, activation='sigmoid'))
    model.compile(loss='mean_absolute_percentage_error', optimizer=keras.optimizers.Adam(lr=0.001))
    return model


def train_model(training_data, model, epochs):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model.fit(X, y, epochs=epochs, verbose=2, batch_size=64)
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
        game_length = 0
        while not done:
            game_length += 1
            action = model.predict(np.array([previous_observation]))
            game_memory.append([deepcopy(previous_observation), deepcopy(action)])
            new_observation, reward, done, info = env.step(denormalize_action(action))
            previous_observation = new_observation
            score += reward
            if done:
                break
        scores.append(score)
        mistakes.append(info['invalid actions taken'] / game_length)
        games.append(game_memory)
    print('Test avg score', statistics.mean(scores))
    print('Test avg mistake rate', statistics.mean(mistakes))


def train_against_random(training_games, model, epochs):
    env = build_env(opponent=RandomAgent(action_space=discrete_space()))
    training_data = play_training(env=env, training_games=training_games, model=model)
    # play_test(model=trained_model, env=env, test_games=test_games)
    return training_data


def validate_against_random(model, test_games):
    env = build_env(opponent=RandomAgent(action_space=discrete_space()))
    play_test(model=model, env=env, test_games=test_games)


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
        game_length = 0
        while not done:
            game_length += 1
            action = env.action_space.sample()
            game_memory.append([deepcopy(previous_observation), deepcopy(action)])
            observation, reward, done, info = env.step(action)
            previous_observation = observation
            score += reward
            if done:
                break
        scores.append(score)
        mistakes.append(info['invalid actions taken'] / game_length)
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], data[1]])
    print('Random avg score', statistics.mean(scores))
    print('Random avg mistake rate', statistics.mean(mistakes))
    return training_data
