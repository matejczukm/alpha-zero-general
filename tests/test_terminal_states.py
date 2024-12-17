import sys
import os
import numpy as np
import logging

from tqdm import tqdm

log = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from PackitAIPlayer import AIPlayer
from TriangleGame.TrianglePlayers import RandomPlayer
# from HexGame.HexGame import HexGame
from TriangleGame.TriangleGame import TriangleGame

n = 500
mcts_wins = 0
nn_wins = 0
size = 4
mode = 'triangular'
ai_player = AIPlayer(size, mode)

boards = []

for i in tqdm(range(n)):
    game = TriangleGame(size)

    players = [RandomPlayer(game).play, None, RandomPlayer(game).play]
    curPlayer = 1
    board = game.getInitBoard()
    it = 0
    for player in players[0], players[2]:
        if hasattr(player, "startGame"):
            player.startGame()

    is_game_over = game.getGameEnded(board, curPlayer, 1)
    prev_board = np.copy(board)
    while is_game_over == 0:
        prev_board = np.copy(board)
        it += 1

        action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer), it)
        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1, it)

        if valids[action] == 0:
            log.error(f'Action {action} is not valid!')
            log.debug(f'valids = {valids}')
            assert valids[action] > 0

        # Notifying the opponent for the move
        opponent = players[-curPlayer + 1]
        if hasattr(opponent, "notify"):
            opponent.notify(board, action)
        board, curPlayer = game.getNextState(board, curPlayer, action)

        is_game_over = game.getGameEnded(board, curPlayer, it + 1)

    move1 = ai_player.mcts_get_action(prev_board, it)
    move2 = ai_player.nnet_get_action(prev_board, it)

    nn_winning_move = game.getGameEnded(prev_board+move2, -curPlayer, it + 1)
    mcts_winning_move = game.getGameEnded(prev_board+move1, -curPlayer, it + 1)

    # print('NN did' + ' not' if nn_winning_move else '' + ' find winning move.')
    # print('MCTS did' + ' not' if mcts_winning_move else '' + ' find winning move.')
    nn_wins -= nn_winning_move
    mcts_wins -= mcts_winning_move
    if not (nn_winning_move or mcts_winning_move):
        # print(prev_board)
        boards.append(prev_board)

print(f'mode: {mode}, size:{size}')
print(f'NN found {nn_wins/n * 100}% of winning moves.')
print(f'MCTS found {mcts_wins/n * 100}% of winning moves.')
for b in boards:
    print(b)


