import numpy as np


class RandomPlayer:

    def __init__(self, game):
        self.game = game

    def play(self, board, turn):
        valids = self.game.getValidMoves(board, 1, turn)
        valid_ids = np.where(valids == 1)[0]
        move = np.random.randint(len(valid_ids))

        return valid_ids[move]
