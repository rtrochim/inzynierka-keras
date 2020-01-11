# This allows to force running on cpu for measurement
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import matplotlib.pyplot as plt

import os

# This removes  tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from training import *
import time
from keras.models import load_model
from gym_backgammon.agents.trained import TrainedAgent

training_games = 100
test_games = 100
runs = 5
epochs = 3


try:
    initial_model = get_latest_model()
except:
    print("Could not load trained model")
    print('========= INITIAL TRAIN AGAINST RANDOM =========')
    initial_model = train_against_random(training_games=training_games, test_games=test_games,
                                         model=build_model(input_size=54, output_size=1), epochs=epochs)


model = initial_model
# What we begin with
print('========= INITIAL TEST AGAINST RANDOM  =========')
validate_against_random(trained_model=model, test_games=test_games)

# Main training loop
for index in range(runs):
    # Build environment using latest model as opponent
    env = build_env(opponent=TrainedAgent(model=model))
    print('========= TRAIN AGAINST ITSELF ' + str(index) + ' =========')
    training_data = play_training(env=env, training_games=training_games, model=model)
    trained_model = train_model(training_data=training_data, model=model, epochs=epochs)
    print('========= TEST AGAINST ITSELF ' + str(index) + ' =========')
    play_test(model=trained_model, env=env, test_games=test_games)
    print('========= TEST AGAINST RANDOM ' + str(index) + ' =========')
    validate_against_random(trained_model=trained_model, test_games=test_games)
    model = trained_model
model.save('./models/trained-' + str(round(time.time())) + '.h5')

# What we end with
print('========= FINAL TESTING AGAINST RANDOM  =========')
validate_against_random(trained_model=model, test_games=test_games)