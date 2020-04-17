FROM tensorflow/tensorflow:latest-gpu-py3
RUN pip install gym
COPY ./gym-backgammon/ /tmp/gym-backgammon
RUN pip install /tmp/gym-backgammon
VOLUME ./models