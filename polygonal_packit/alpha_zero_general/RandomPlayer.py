import numpy as np
from .TriangleGame.TriangleGame import TriangleGame as tg
from .HexGame.HexGame import HexGame as hg


class RandomPlayer:

    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

        if mode == 'triangular':
            self.game = tg(size)
        else:
            self.game = hg(size)

    def get_action_for_arena(self, board, turn):
        valids = self.game.getValidMoves(board, 1, turn)
        valid_ids = np.where(valids == 1)[0]
        move = np.random.randint(len(valid_ids))
        return valid_ids[move]