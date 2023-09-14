import numpy as np


class PlayData(object):

    def __init__(self):
        self.board = None
        self.state = np.zeros(9, 10, 9)
        self.winner = 0
