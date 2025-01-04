from .game_logic import place_polygon, get_possible_moves, get_board
import json
from . import data_conversions as dc


def start_game(board_size):
    """

    Args:
        board_size (int):

    """
    board = get_board(board_size)
    possible_moves = get_possible_moves(board, 1)
    return {
        'board': dc.convert_numpy_array_to_triangle(board),
        'moves': [
            json.dumps(dc.convert_numpy_array_to_triangle(move), separators=(',', ': ')) for move in possible_moves
        ]
    }


def perform_move(board, move, turn):
    """

    Args:
        board (list): Matrix-like representation of the board.
        move (list): Matrix-like representation of the polygon to be placed.
        turn (int): The current turn.

    Returns:
        dict:

    """

    board_np = dc.convert_triangle_to_numpy_array(board)
    move_np = dc.convert_triangle_to_numpy_array(move)

    # board_np = board_np.astype(bool).astype(int)
    # move_np = move_np.astype(bool).astype(int)

    board_np = place_polygon(board_np, move_np)
    possible_moves = get_possible_moves(board_np, turn)

    return {
        'board': dc.convert_numpy_array_to_triangle(board_np),
        'moves': [
            json.dumps(dc.convert_numpy_array_to_triangle(move ), separators=(',', ': ')) for move in
            possible_moves
        ]
    }


if __name__ == '__main__':
    print(start_game(3))
