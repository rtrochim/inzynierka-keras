class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def make_decision(self, _):
        return self.action_space.sample()
