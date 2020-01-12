import numpy as np


class TrainedAgent:
    def __init__(self, model):
        self.model = model

    def make_decision(self, observation):
        return self.model.predict(np.array([observation]))
