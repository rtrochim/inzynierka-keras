from training import *
import time
from keras.models import load_model
from gym_backgammon.agents.trained import TrainedAgent
# This removes excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_games = 1000
test_games = 100
runs = 1
epochs = 1

try:
    model = get_latest_model()
except:
    print("Could not load trained model")
    model = build_model(input_size=54, output_size=1)
    # print('========= INITIAL RANDOM GAMES =========')
    # env = build_env(opponent=RandomAgent(action_space=continuous_space()))
    # training_data = play_random_games(training_games=training_games, env=env)
    # model = train_model(training_data=training_data, model=model, epochs=1)

# What we begin with
print('========= INITIAL TEST vs RANDOM  =========')
validate_against_random(trained_model=model, test_games=test_games)

for index in range(runs):
    # Build environment using latest model as opponent
    env = build_env(opponent=TrainedAgent(model=model))
    print('========= TRAIN vs SELF ' + str(index) + ' =========')
    training_data = play_training(env=env, training_games=training_games, model=model)
    model = train_model(training_data=training_data, model=model, epochs=epochs)
    # print('========= TEST vs SELF ' + str(index) + ' =========')
    # play_test(model=trained_model, env=env, test_games=test_games)
    # print('========= TRAIN vs RANDOM ' + str(index) + ' =========')
    # env = build_env(opponent=RandomAgent(action_space=continuous_space()))
    # training_data = play_random_games(training_games=training_games, env=env)
    # model = train_model(training_data=training_data, model=model, epochs=epochs)
    # if index % 10 == 9:
    #     print('========= TEST vs RANDOM ' + str(index) + ' =========')
    #     validate_against_random(trained_model=model, test_games=test_games)
model.save('./models/trained-' + str(round(time.time())) + '.h5')

# What we end with
print('========= FINAL TEST vs RANDOM  =========')
validate_against_random(trained_model=model, test_games=test_games)


