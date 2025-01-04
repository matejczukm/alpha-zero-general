import numpy as np


def convert_numpy_board_to_list(board_np):
    """
    Converts a NumPy representation of a hexagonal board to a list-based representation.

    Args:
        board_np (numpy.ndarray): A NumPy array representing the hexagonal board.

    Returns:
        list: A list-based representation of the board.
    """
    board = []
    board_np = board_np.astype(int)
    n = board_np.shape[0]
    offset = int(
        np.floor(n / 2)
    )
    for i in range(offset + 1):
        board.append(board_np[i].tolist()[:n - offset + i])
    for i in range(offset):
        board.append(board_np[offset + 1 + i].tolist()[1 + i:])
    return board


def convert_list_board_to_numpy(board, value=0):
    """
    Converts a list-based representation of a hexagonal board to a NumPy array.

    Args:
        board (list): A list-based representation of the hexagonal board.
        value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        numpy.ndarray: A NumPy array representing the board.
    """
    n = len(board)
    offset = int(
        np.floor(n / 2)
    )
    for i in range(offset):
        board[i].extend([value] * (offset - i))
    for i in range(offset):
        board[offset + 1 + i] = [value] * (i + 1) + board[offset + 1 + i]

    return np.array(board)


def expand_to_hex_board(polygon, board_size, left, top):
    """
    Expands a polygon to fit within a hexagonal board at a specified position.

    Args:
        polygon (numpy.ndarray): A 2D NumPy array representing the polygon.
        board_size (int): The size of the hexagonal board.
        left (int): The horizontal offset for placing the polygon.
        top (int): The vertical offset for placing the polygon.

    Returns:
        numpy.ndarray: A NumPy array of the expanded polygon on the hexagonal board.

    Raises:
        AssertionError: If the polygon does not fit within the board dimensions based on the offsets.
    """
    polygon_height = len(polygon)
    polygon_width = len(polygon[0])

    assert polygon_height + top <= board_size, 'Invalid vertical offset'
    assert polygon_width + left <= board_size, 'Invalid horizontal offset'

    left_offset = np.zeros([polygon_height, left])
    right_offset = np.zeros([polygon_height, board_size - left - polygon_width])
    top_offset = np.zeros([top, board_size])
    bottom_offset = np.zeros([board_size - top - polygon_height, board_size])

    expanded_polygon = np.hstack((left_offset, polygon, right_offset))
    expanded_polygon = np.vstack((top_offset, expanded_polygon, bottom_offset))

    return expanded_polygon
