import random
import numpy as np

from gym_backgammon.game.board import Board
from itertools import chain


def roll_dice():
    dice = [np.random.randint(1, 6), np.random.randint(1, 6)]
    # If dice are equal, double moves available
    dice = dice * 2 if dice[0] == dice[1] else dice
    return dice


# Get all possible actions
def all_actions():
    actions = []
    for source in range(0, 24):
        for target in range(0, 24):
            if (target - source) <= 6:
                actions += [('move', source, target), ('move', target, source),
                            ('move_with_hit', source, target), ('move_with_hit', target, source)]
    for home_checker in chain(range(0, 6), range(18, 24)):
        actions += [('reenter', home_checker), ('reenter_with_hit', home_checker), ('bear_off', home_checker)]
    return actions


def get_action(action_index):
    return all_actions()[action_index]


def get_random_action(valid_actions):
    first_choice = random.choice(valid_actions)
    while not first_choice:
        first_choice = random.choice(valid_actions)
    return random.choice(first_choice)


class Game:

    def __init__(self, player, opponent):
        self.board = Board()
        self.w_hit = 0
        self.b_hit = 0
        self.w_borne_off = 0
        self.b_borne_off = 0
        self.w_can_bear_off = False
        self.b_can_bear_off = False
        # Player one is our (white) perspective
        self.opponent = opponent
        self.dice = []

        # Higher dice roll starts
        w_toss = roll_dice()
        b_toss = roll_dice()
        while sum(w_toss) == sum(b_toss):
            w_toss = roll_dice()
            b_toss = roll_dice()

        if sum(w_toss) > sum(b_toss):
            print("White begin \n")
            self.turn = 1
        else:
            print("Black begin \n")
            self.turn = 2
            self.opponent_turn()

    # Play the turn
    def player_turn(self, action_index):
        # No valid actions available
        valid_actions, rewards = self.get_valid_actions()
        if not any(valid_actions):
            self.turn = 2
            self.opponent_turn()
            return 0  # 0 reward

        # Valid action chosen, act until moves remain, then return achieved reward
        action = get_action(action_index)
        for index, actions in enumerate(valid_actions):
            if action in actions:
                self.act(action)
                del self.dice[index]
                if not self.dice:
                    self.turn = 2
                    self.opponent_turn()
                return rewards[index][actions.index(action)]

        # Invalid action chosen, play random valid, return negative reward
        action = get_random_action(valid_actions)
        for index, action_set in enumerate(valid_actions):
            if action in action_set:
                self.act(action)
                del self.dice[index]
                if not self.dice:
                    self.turn = 2
                    self.opponent_turn()
                return -10

    # Start opponent turn
    def opponent_turn(self):
        self.dice = roll_dice()  # Roll dice for yourself(black)
        while self.dice:
            if self.game_over():
                break
            self.play_opponent()
        self.turn = 1
        self.dice = roll_dice()  # Roll dice for your opponent(white)

    # Play single opponent action
    def play_opponent(self):
        # No valid actions
        valid_actions, _ = self.get_valid_actions()
        if not any(valid_actions):
            self.turn = 1
            self.dice = []
            return

        action = get_action(self.opponent.make_decision(self.get_observation()))

        # Valid action chosen
        for index, action_set in enumerate(valid_actions):
            if action in action_set:
                self.act(action)
                del self.dice[index]
                return

        # Invalid action chosen
        action = get_random_action(valid_actions)
        for index, action_set in enumerate(valid_actions):
            if action in action_set:
                self.act(action)
                del self.dice[index]
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
        points = self.board.board
        # Figure out valid actions for each roll
        for roll in self.dice:
            w_indices = []
            b_indices = []
            empty_indices = []
            # Count colored points
            for point_index, point in enumerate(points):
                if point.color == 'w':
                    w_indices.append(point_index)
                elif point.color == 'b':
                    b_indices.append(point_index)
                else:
                    empty_indices.append(point_index)

            # Check whether anyone can bear off
            if max(w_indices) < 6 and (self.w_hit == 0):
                self.w_can_bear_off = True
            if min(b_indices) > 17 and (self.b_hit == 0):
                self.b_can_bear_off = True

            actions = []
            rewards = []

            # White turn
            if self.turn == 1:
                if self.w_hit > 0:  # If anything is hit only reenters are valid
                    if (24 - roll) in (empty_indices + w_indices):
                        actions.append(('reenter', 24 - roll))
                        rewards.append(roll)
                    elif ((24 - roll) in b_indices) and ((points[24 - roll]).count < 2):
                        actions.append(('reenter_with_hit', 24 - roll))
                        rewards.append(24)

                else:  # Check for any possible moves/hits/bear_offs using own checkers
                    for point_index in w_indices:
                        if (point_index - roll) in (w_indices + empty_indices):
                            actions.append(('move', point_index, point_index - roll))
                            rewards.append(roll)
                        if ((point_index - roll) in b_indices) and ((points[point_index - roll]).count < 2):
                            actions.append(('move_with_hit', point_index, point_index - roll))
                            rewards.append(point_index)
                        if self.w_can_bear_off and (point_index < roll):
                            actions.append(('bear_off', point_index))
                            rewards.append(roll)
            # Black turn
            if self.turn == 2:
                if self.b_hit > 0:  # If anything is hit only reenters are valid
                    if (roll - 1) in (empty_indices + b_indices):
                        actions.append(('reenter', roll - 1))
                        rewards.append(roll)
                    elif ((roll - 1) in w_indices) and ((points[roll - 1]).count < 2):
                        actions.append(('reenter_with_hit', roll - 1))
                        rewards.append(24)

                else:  # Check for any possible moves/hits/bear_offs using own checkers
                    for point_index in b_indices:
                        if (point_index + roll) in (b_indices + empty_indices):
                            actions.append(('move', point_index, point_index + roll))
                            rewards.append(roll)
                        if ((point_index + roll) in w_indices) and ((points[point_index + roll]).count < 2):
                            actions.append(('hit', point_index, point_index + roll))
                            rewards.append(24 - point_index)
                        if self.b_can_bear_off and ((23 - point_index) < roll):
                            actions.append(('bear_off', point_index))
                            rewards.append(roll)

            action_array.append(actions)
            reward_array.append(rewards)

        return action_array, reward_array

    # Perform the given action and update the board
    def act(self, action):
        # White turn
        if self.turn == 1:
            if self.w_hit > 0:
                if action[0] == "reenter":
                    self.w_hit -= 1
                    self.board.reenter("w", action[1])
                if action[0] == "reenter_with_hit":
                    self.w_hit -= 1
                    self.b_hit += 1
                    self.board.reenter_with_hit("w", action[1])
            else:
                if action[0] == "move":
                    self.board.move("w", action[1], action[2])
                if action[0] == "move_with_hit":
                    self.b_hit += 1
                    self.board.move_with_hit("w", action[1], action[2])
                if action[0] == "bear_off":
                    self.board.bear_off("w", action[1])
        # Black turn
        if self.turn == 2:
            if self.b_hit > 0:
                if action[0] == "reenter":
                    self.b_hit -= 1
                    self.board.reenter("b", action[1])
                elif action[0] == "reenter_with_hit":
                    self.w_hit += 1
                    self.b_hit -= 1
                    self.board.reenter_with_hit("b", action[1])
            else:
                if action[0] == "move":
                    self.board.move("b", action[1], action[2])
                elif action[0] == "move_with_hit":
                    self.w_hit += 1
                    self.board.move_with_hit("b", action[1], action[2])
                elif action[0] == "bear_off":
                    self.board.bear_off("b", action[1])

        self.w_borne_off = self.board.borne_off['w']
        self.w_borne_off = self.board.borne_off['b']

    # Returns a list with information about current game state
    def get_observation(self):
        state_vector = []
        if len(self.dice) < 1:
            state_vector += [0, 0]
        elif len(self.dice) < 2:
            state_vector += [self.dice[0], 0]
        else:
            state_vector += [self.dice[0], self.dice[1]]
        state_vector += [self.w_hit, self.b_hit, self.w_borne_off, self.b_borne_off]

        for point in self.board.board:
            if point.color == 'w':
                state_vector += [1, point.count]
            elif point.color == 'b':
                state_vector += [2, point.count]
            else:
                state_vector += [0, 0]
        return state_vector

    # Returns true if any player has 0 checkers
    def game_over(self):
        points = self.board.board
        for color in ['w', 'b']:
            i = 0
            for point in points:
                checkers = point.count
                if point.color == color:
                    i += checkers
            if i < 1:
                return True
        return False

    # Prints current game state to console
    def print_game(self):
        state = self.get_observation()
        max_pieces_point = max(state)
        state = zip(state[::2], state[1::2])
        state = list(state)
        up_side = list()
        bottom_side = list()
        for i in range(3, int(3 + (len(state) - 3) / 2)):
            bottom_side.append(state[i])
        for i in range(int(3 + (len(state) - 3) / 2), len(state)):
            up_side.append(state[i])
        bottom_side.reverse()
        board = list()
        board.append(
            '  -------------------------------------------------------------- ')
        board.append(
            ' |12  13  14  15   16   17   |    | 18   19   20   21   22   23  | ')
        board.append(
            ' |---------------------------------------------------------------|')

        for i in range(0, max_pieces_point):
            point = list()
            for item in up_side:
                if i < item[1]:
                    point.append('w' if item[0] == 1 else 'b')
                else:
                    point.append(' ')
            board.append(
                ' |{}   {}   {}   {}    {}    {}    |    | {}    {}    {}    {}    {}    {}   | '.format(*point))
        board.append(
            '  --------------------------------------------------------------- ')

        for i in range(0, max_pieces_point):
            point = list()
            for item in bottom_side:
                if max_pieces_point - item[1] <= i:
                    point.append('w' if item[0] == 1 else 'b')
                else:
                    point.append(' ')
            board.append(
                ' |{}   {}   {}    {}    {}    {}    |    | {}    {}    {}    {}    {}    {}  | '.format(*point))
        board.append(
            ' |---------------------------------------------------------------|')
        board.append(
            ' |11  10   9   8    7    6    |    | 5    4    3    2    1    0  | ')
        board.append(
            '  --------------------------------------------------------------- ')
        board.append('Dice 1: {}'.format(state[0][0]))
        board.append('Dice 2: {}'.format(state[0][1]))
        board.append('White hit: {}'.format(state[1][0]))
        board.append('Black hit: {}'.format(state[1][1]))
        board.append('White borne off: {}'.format(state[2][0]))
        board.append('Black borne off: {}'.format(state[2][1]))

        for line in board:
            print(line)
