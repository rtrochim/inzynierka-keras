import random
import numpy as np
from cachetools.func import lru_cache

from gym_backgammon.game.board import Board
from itertools import chain


def roll_dice():
    dice = [np.random.randint(1, 6), np.random.randint(1, 6)]
    # If dice are equal, double the moves available
    dice = dice * 2 if dice[0] == dice[1] else dice
    return dice


# Get all possible actions
@lru_cache(maxsize=1)
def all_actions():
    actions = []
    for source in range(0, 24):
        for target in range(0, 24):
            if (target - source) <= 6:
                actions += [('move', source, target), ('move', target, source),
                            ('move_and_hit', source, target), ('move_and_hit', target, source)]
    for home_checker in chain(range(0, 6), range(18, 24)):
        actions += [('reenter', home_checker), ('reenter_and_hit', home_checker), ('bear_off', home_checker)]
    return actions


def get_random_action(valid_actions):
    first_choice = random.choice(valid_actions)
    while not first_choice:
        first_choice = random.choice(valid_actions)
    return random.choice(first_choice)


class Game:
    def __init__(self, player, opponent):
        self.board = Board()
        self.white_hit = 0
        self.black_hit = 0
        self.white_borne_off = 0
        self.black_borne_off = 0
        self.white_can_bear_off = False
        self.black_can_bear_off = False
        # Player one is our (white) perspective
        self.opponent = opponent
        self.dice = []

        # Higher dice roll starts
        white_roll = roll_dice()
        black_roll = roll_dice()
        while sum(white_roll) == sum(black_roll):
            white_roll = roll_dice()
            black_roll = roll_dice()

        if sum(white_roll) > sum(black_roll):
            self.turn = 'w'
        else:
            self.turn = 'b'
            self.opponent_turn()

    # Play the turn
    def player_turn(self, action_index):
        assert self.turn == 'w', 'White playing in Black turn'
        # No valid actions available
        valid_actions, rewards = self.get_valid_actions()
        if not any(valid_actions):
            self.turn = 'b'
            self.opponent_turn()
            return 0  # 0 reward

        assert len(valid_actions) == len(self.dice)

        # Valid action chosen, act until moves remain, then return achieved reward
        action = all_actions()[action_index]
        for index, valid_action in enumerate(valid_actions):
            if action in valid_action:
                self.act(action)
                self.dice.pop(index)
                if not self.dice:
                    self.turn = 'b'
                    self.opponent_turn()
                return rewards[index][valid_action.index(action)]

        # Invalid action chosen, play random valid, return negative reward
        action = get_random_action(valid_actions)
        for index, valid_action in enumerate(valid_actions):
            if action in valid_action:
                self.act(action)
                self.dice.pop(index)
                if not self.dice:
                    self.turn = 'b'
                    self.opponent_turn()
                return -10

        # This should never be reached
        assert False, 'No action was taken!'

    # Start opponent turn
    def opponent_turn(self):
        self.dice = roll_dice()  # Roll dice for yourself(black)
        while self.dice:
            if self.game_over():
                break
            self.play_opponent()
        self.turn = 'w'
        self.dice = roll_dice()  # Roll dice for your opponent(white)

    # Play single opponent action
    def play_opponent(self):
        assert self.turn == 'b', 'Black playing in White turn'
        # No valid actions
        valid_actions, _ = self.get_valid_actions()
        if not any(valid_actions):
            self.turn = 'w'
            self.dice = []
            return

        assert len(valid_actions) == len(self.dice)

        action = all_actions()[self.opponent.make_decision(self.get_observation())]

        # Valid action chosen
        for index, action_set in enumerate(valid_actions):
            if action in action_set:
                self.act(action)
                self.dice.pop(index)
                return

        # Invalid action chosen
        action = get_random_action(valid_actions)
        for index, action_set in enumerate(valid_actions):
            if action in action_set:
                self.act(action)
                self.dice.pop(index)
                return

    def get_valid_actions(self):
        """Returns two NUMPY array of NUMPY arrays as such:
        For actions:
        [
            [Valid action 1, Valid action 2, ...], # Dice 1 valid actions
            [Valid action 1, Valid action 2, ...], # Dice 2 valid actions
            .
            .
        ]
        For rewards:
        [
            [Reward of valid action 1, Reward of valid action 2, ...],
            [Reward of valid action 1, Reward of valid action 2, ...],
            .
            .
        ]
        """
        action_array = []
        reward_array = []
        points = self.board.points
        # Figure out valid actions for each roll
        for roll in self.dice:
            white_indices = []
            black_indices = []
            empty_indices = []
            # Count colored points
            for point_index, point in enumerate(points):
                if point.color == 'w':
                    white_indices.append(point_index)
                elif point.color == 'b':
                    black_indices.append(point_index)
                else:
                    empty_indices.append(point_index)

            assert sum(point.count for point in self.board.points if
                       point.color == 'w') + self.white_borne_off + self.white_hit == 15
            assert sum(point.count for point in self.board.points if
                       point.color == 'b') + self.black_borne_off + self.black_hit == 15

            # Check whether anyone can bear off
            if white_indices and max(white_indices) < 6 and (self.white_hit == 0):
                self.white_can_bear_off = True
            if black_indices and min(black_indices) > 17 and (self.black_hit == 0):
                self.black_can_bear_off = True

            actions = []
            rewards = []

            # White turn
            if self.turn == 'w':
                if self.white_hit > 0:  # If anything is hit only reenters are valid
                    if (24 - roll) in (empty_indices + white_indices):
                        actions.append(('reenter', 24 - roll))
                        rewards.append(roll)
                    elif ((24 - roll) in black_indices) and ((points[24 - roll]).count < 2):
                        actions.append(('reenter_and_hit', 24 - roll))
                        rewards.append(24)

                else:  # Check for any possible moves/hits/bear_offs using own checkers
                    for point_index in white_indices:
                        if (point_index - roll) in (white_indices + empty_indices):
                            actions.append(('move', point_index, point_index - roll))
                            rewards.append(roll)
                        if ((point_index - roll) in black_indices) and ((points[point_index - roll]).count < 2):
                            actions.append(('move_and_hit', point_index, point_index - roll))
                            rewards.append(point_index)
                        if self.white_can_bear_off and (point_index < roll):
                            actions.append(('bear_off', point_index))
                            rewards.append(roll)
            # Black turn
            if self.turn == 'b':
                if self.black_hit > 0:  # If anything is hit only reenters are valid
                    if (roll - 1) in (empty_indices + black_indices):
                        actions.append(('reenter', roll - 1))
                        rewards.append(roll)
                    elif ((roll - 1) in white_indices) and ((points[roll - 1]).count < 2):
                        actions.append(('reenter_and_hit', roll - 1))
                        rewards.append(24)

                else:  # Check for any possible moves/hits/bear_offs using own checkers
                    for point_index in black_indices:
                        if (point_index + roll) in (black_indices + empty_indices):
                            actions.append(('move', point_index, point_index + roll))
                            rewards.append(roll)
                        if ((point_index + roll) in white_indices) and ((points[point_index + roll]).count < 2):
                            actions.append(('move_and_hit', point_index, point_index + roll))
                            rewards.append(24 - point_index)
                        if self.black_can_bear_off and ((23 - point_index) < roll):
                            actions.append(('bear_off', point_index))
                            rewards.append(roll)

            action_array.append(actions)
            reward_array.append(rewards)

        return action_array, reward_array

    # Perform the given action and update the board
    def act(self, action):
        # White turn
        if self.turn == 'w':
            if self.white_hit > 0:
                if action[0] == "reenter":
                    self.white_hit -= 1
                    self.board.reenter("w", action[1])
                elif action[0] == "reenter_and_hit":
                    self.white_hit -= 1
                    self.black_hit += 1
                    self.board.reenter_and_hit("w", action[1])
            else:
                if action[0] == "move":
                    self.board.move("w", action[1], action[2])
                elif action[0] == "move_and_hit":
                    self.black_hit += 1
                    self.board.move_and_hit("w", action[1], action[2])
                elif action[0] == "bear_off":
                    self.board.bear_off("w", action[1])
                    self.white_borne_off = self.board.borne_off['w']
        # Black turn
        if self.turn == 'b':
            if self.black_hit > 0:
                if action[0] == "reenter":
                    self.black_hit -= 1
                    self.board.reenter("b", action[1])
                elif action[0] == "reenter_and_hit":
                    self.white_hit += 1
                    self.black_hit -= 1
                    self.board.reenter_and_hit("b", action[1])
            else:
                if action[0] == "move":
                    self.board.move("b", action[1], action[2])
                elif action[0] == "move_and_hit":
                    self.white_hit += 1
                    self.board.move_and_hit("b", action[1], action[2])
                elif action[0] == "bear_off":
                    self.board.bear_off("b", action[1])
                    self.black_borne_off = self.board.borne_off['b']

    # Returns a list with information about current game state
    def get_observation(self):
        state_vector = []
        if len(self.dice) < 1:
            state_vector += [0, 0]
        elif len(self.dice) < 2:
            state_vector += [self.dice[0], 0]
        else:
            state_vector += [self.dice[0], self.dice[1]]
        state_vector += [self.white_hit, self.black_hit, self.white_borne_off, self.black_borne_off]

        for point in self.board.points:
            if point.color == 'w':
                state_vector += [1, point.count]
            elif point.color == 'b':
                state_vector += [2, point.count]
            else:
                state_vector += [0, 0]
        return state_vector

    # Game ends when any player has 15 borne off checkers
    def game_over(self):
        if self.black_borne_off == 15 or self.white_borne_off == 15:
            return True
        else:
            return False

    # Prints current game state to console
    def print_game(self):
        state = self.get_observation()
        max_pieces_point = max(point.count for point in self.board.points)
        state = zip(state[::2], state[1::2])
        state = list(state)
        up_side = list()
        bottom_side = list()
        for i in range(3, int(3 + (len(state) - 3) / 2)):
            bottom_side.append(state[i])
        for i in range(int(3 + (len(state) - 3) / 2), len(state)):
            up_side.append(state[i])
        bottom_side.reverse()
        output = list()
        output.append(
            '  --------------------------------------------------------------- ')
        output.append(
            ' | 12  13  14  15   16   17   |    | 18   19   20   21   22   23 | ')
        output.append(
            ' |---------------------------------------------------------------|')

        for i in range(0, max_pieces_point):
            point = list()
            for item in up_side:
                if i < item[1]:
                    point.append('w' if item[0] == 1 else 'b')
                else:
                    point.append(' ')
            output.append(
                ' | {}   {}   {}   {}    {}    {}    |    | {}    {}    {}    {}    {}    {}  | '.format(*point))
        output.append(
            '  --------------------------------------------------------------- ')

        for i in range(0, max_pieces_point):
            point = list()
            for item in bottom_side:
                if max_pieces_point - item[1] <= i:
                    point.append('w' if item[0] == 1 else 'b')
                else:
                    point.append(' ')
            output.append(
                ' |{}   {}   {}    {}    {}    {}    |    | {}    {}    {}    {}    {}    {}  | '.format(*point))
        output.append(
            ' |---------------------------------------------------------------|')
        output.append(
            ' |11  10   9   8    7    6    |    | 5    4    3    2    1    0  | ')
        output.append(
            '  --------------------------------------------------------------- ')
        output.append('Dice 1: {}'.format(state[0][0]))
        output.append('Dice 2: {}'.format(state[0][1]))
        output.append('White hit: {}'.format(state[1][0]))
        output.append('Black hit: {}'.format(state[1][1]))
        output.append('White borne off: {}'.format(state[2][0]))
        output.append('Black borne off: {}'.format(state[2][1]))

        for line in output:
            print(line)
