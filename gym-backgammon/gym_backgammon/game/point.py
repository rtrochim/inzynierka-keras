# A single elements of the board.
# Point can be 'w' - white, 'b' - black or None - empty
# Point may contain between 0 and 15 checkers


class Point:

    def __init__(self, color, count):
        self.count = count
        self.color = color

    def add_checker(self):
        self.count += 1

    def add_first_checker(self, color):
        # Color can be changed only if checker is the first one added
        self.color = color
        self.count = 1

    def remove_checker(self):
        self.count -= 1
        # Removing last checker removes the color
        if self.count == 0:
            self.color = None
