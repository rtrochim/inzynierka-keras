import numpy as np

class TrainedAgent:
    def __init__(self, model):
        self.model = model

    def make_decision(self, observation):
        return round(np.array(self.model.predict(np.array([observation]))).item(0))
