import json
from . import data_conversions as dc
from .game_logic import *


def start_game(board_side):
    board_np = get_board(board_side)  # np.ndarray
    possible_moves = get_possible_moves(board_np, 1)

    return {
        'board': dc.convert_numpy_board_to_list(board_np),
        'moves': [
            json.dumps(dc.convert_numpy_board_to_list(move), separators=(',', ': ')) for move in possible_moves
        ]
    }


def perform_move(board, move, turn):
    board_np = dc.convert_list_board_to_numpy(board, 1)
    move_np = dc.convert_list_board_to_numpy(move)

    board_np = place_polygon(board_np, move_np)
    possible_moves = get_possible_moves(board_np, turn)

    return {
        'board': dc.convert_numpy_board_to_list(board_np),
        'moves': [
            json.dumps(dc.convert_numpy_board_to_list(move), separators=(',', ': ')) for move in
            possible_moves
        ]
    }


if __name__ == '__main__':
    pass

