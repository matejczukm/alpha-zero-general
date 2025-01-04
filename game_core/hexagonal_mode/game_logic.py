import numpy as np
from . import data_conversions as dc
from . import polygon_creation as pc
from .display import print_board


def get_board(board_side):
    """
    Creates board of size board_side.

    Args:
        board_side (int): The size of the board.

    Returns:
        numpy.ndarray: A NumPy array representing the initial state of the board.
    """
    board_size = board_side * 2 - 1
    board = np.zeros([board_size, board_size])
    for row in range(board_side):
        for column in range(board_side + row, board_size):
            board[row, column] = 1
    for row in range(board_side):
        for column in range(row):
            board[row + board_side - 1, column] = 1
    return board


def get_possible_placements_of_polygon(polygon_matrix, board):
    """
    Finds all valid placements for a given polygon on the board.

    Args:
        polygon_matrix (numpy.ndarray): A 2D NumPy array representing the polygon.
        board (numpy.ndarray): The current state of the game board.

    Returns:
        list: A list of valid placements for the polygon.
    """
    board_size = len(board)
    horizontal_offset = board_size - len(polygon_matrix[0]) + 1
    vertical_offset = board_size - len(polygon_matrix) + 1
    possible_placements = [None] * (horizontal_offset * vertical_offset)
    count = 0
    for top in range(vertical_offset):
        for left in range(horizontal_offset):

            expanded_polygon = dc.expand_to_hex_board(polygon_matrix, board_size, left, top)
            if not np.any(board + expanded_polygon == 2):
                possible_placements[count] = expanded_polygon
                count += 1
    return possible_placements[:count]


def get_possible_placements(board, k):
    """
    Generates all possible placements of polygons of size k on the board.

    Args:
        board (numpy.ndarray): The current state of the game board.
        k (int): The size of the polygon to place.

    Returns:
        list: A list of all valid placements of polygons of size k.
    """
    board_size = len(board)
    valid_vectors = pc.get_polygon_vectors(k, board_size)
    polygon_matrices = []
    for vector in valid_vectors:
        polygon_matrices.extend(pc.convert_polygon_vector_to_matrices(vector))
    possible_placements = []
    for polygon_matrix in polygon_matrices:
        possible_placements.extend(get_possible_placements_of_polygon(polygon_matrix, board))
    return possible_placements


def get_possible_moves(board, turn):
    """
    Generates all possible moves for the current turn.

    Args:
        board (numpy.ndarray): The current state of the game board.
        turn (int): The current turn number, determining the polygon size.

    Returns:
        list: A list of all valid moves for the current turn.
    """
    res = get_possible_placements(board, turn)
    res.extend(
        get_possible_placements(board, turn + 1)
    )
    return res


def place_polygon(board, polygon):
    """
    Places a polygon on the board.

    Args:
        board (numpy.ndarray): The current state of the game board.
        polygon (numpy.ndarray): The polygon to be placed.

    Returns:
        numpy.ndarray: The updated game board after placing the polygon.
    """
    return board + polygon


def validate_placements(placements, board):
    """
    Validates an array of placements.

    Args:
        placements (numpy.ndarray): A NumPy array of placements of polygons.
        board (numpy.ndarray): A NumPy array representing the current state of the board.

    Returns:
        numpy.ndarray: Binary vector indicating which placement is valid.
    """

    return np.array(
        [int(not np.any(board + placement == 2)) for placement in placements]
    )


def play(board_size):
    """
    Initiates a terminal-based gameplay session for a hexagonal board game.

    Args:
        board_size (int): The side length of the hexagonal game board.

    Returns:
        None
    """
    board = get_board(board_size)
    turn = 1
    moves = get_possible_moves(board, turn)
    while moves:
        moves_dict = {i: move for i, move in enumerate(moves)}
        for i, move in moves_dict.items():
            print(f'Move {i}:')
            print_board(move)
        print('Board: ')
        print_board(board)
        chosen_move = int(input('Choose move number: '))
        board = place_polygon(board, moves_dict[chosen_move])
        turn += 1
        moves = get_possible_moves(board, turn)
        # print('Board: ')
        # print(board)
    print(f'Player {1 + turn % 2} wins after {turn - 1} turns')
    print('Board: ')
    print_board(board)
    return


if __name__ == '__main__':
    play(5)
