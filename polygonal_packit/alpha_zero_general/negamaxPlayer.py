from easyAI.TwoPlayerGame import TwoPlayerGame
from HexGame.HexGame import HexGame
from TriangleGame.TriangleGame import TriangleGame
import numpy as np
from PackitAIPlayer import AIPlayer
from easyAI import Negamax


class easyAIPackit(TwoPlayerGame):
    def __init__(self, players, size, mode='triangular'):
        self.players = players
        self.game = TriangleGame(size) if mode == 'triangular' else HexGame(size)
        self.board = self.game.getInitBoard()
        self.turn = 1
        self.current_player = 1

    def possible_moves(self):
        valids = self.game.getValidMoves(self.board, 1, self.turn)
        return np.where(valids == 1)[0].tolist()

    def make_move(self, move, turn=None):
        if turn:
            self.turn = turn
        self.board = self.board + self.game.action_space[move]
        self.turn += 1

    def is_over(self):
        return bool(self.game.getGameEnded(self.board, 1, self.turn))

    def scoring(self):
        if not self.is_over():
            return 0
        return -100

    def show(self):
        print(self.board)


class NegamaxPlayer:
    def __init__(self, size, mode, depth=3):
        self.negamax_game = easyAIPackit([], size, mode)
        self.negamax = Negamax(depth)

    def get_action_for_arena(self, board, turn):
        self.negamax_game.board = board
        self.negamax_game.turn = turn
        return self.negamax(self.negamax_game)