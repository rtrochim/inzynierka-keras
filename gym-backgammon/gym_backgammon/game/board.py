# This class here implements the game board.
#
# The data structure is a list of 24 points.

from gym_backgammon.game.point import Point


class Board:

    def __init__(self):
        # Initial board state
        self.points = [Point('b', 2), Point(None, 0), Point(None, 0),
                       Point(None, 0), Point(None, 0), Point('w', 5),
                       Point(None, 0), Point('w', 3), Point(None, 0),
                       Point(None, 0), Point(None, 0), Point('b', 5),
                       Point('w', 5), Point(None, 0), Point(None, 0),
                       Point(None, 0), Point('b', 3), Point(None, 0),
                       Point('b', 5), Point(None, 0), Point(None, 0),
                       Point(None, 0), Point(None, 0), Point('w', 2)]
        # Hit counter
        self.hit = {'w': 0, 'b': 0}
        # Borne off counter
        self.borne_off = {'w': 0, 'b': 0}

    # Move checker from one point to another. Check whether target is empty
    # or occupied by the player moving.
    def move(self, color, source_point, target_point):
        source = self.points[source_point]
        target = self.points[target_point]
        target.add_first_checker(color) if target.count == 0 else target.add_checker()
        source.remove_checker()

    # Hit opposing checker. Check whether target checker is the only one on point
    def move_with_hit(self, color, source_point, target_point):
        source = self.points[source_point]
        target = self.points[target_point]
        source.remove_checker()
        target.add_first_checker(color)
        opponent = 'b' if color == 'w' else 'w'
        self.hit[opponent] += 1

    # Bears off a checker from the board
    def bear_off(self, color, source_point):
        source = self.points[source_point]
        source.remove_checker()
        self.borne_off[color] += 1

    # Moves hit checker back to point
    def reenter(self, color, target_point):
        target = self.points[target_point]
        target.add_first_checker(color) if target.count == 0 else target.add_checker()
        self.hit[color] -= 1

    # Moves hit checker back to point hitting opponent
    def reenter_with_hit(self, color, target_point):
        target = self.points[target_point]
        target.add_first_checker(color)
        opponent = 'b' if color == 'w' else 'w'
        self.hit[opponent] += 1
        self.hit[color] -= 1
