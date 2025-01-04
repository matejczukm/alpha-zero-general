import numpy as np


def validate_polygon(polygon_vector):
    """
    Validates whether a polygon vector represents a valid polygon.

    Args:
        polygon_vector (numpy.ndarray): The vector representation of the polygon.

    Returns:
        bool: True if the vector is valid, False otherwise.
    """
    vector_length = len(polygon_vector)
    if vector_length == 1:
        return True

    tmp_polygon_vector = np.array(polygon_vector)
    tmp_polygon_vector = np.append(tmp_polygon_vector, -1)

    i = 1
    last_element = polygon_vector[0]
    current_element = tmp_polygon_vector[i]

    while i < vector_length and last_element == current_element - 1:
        last_element = current_element
        i += 1
        current_element = tmp_polygon_vector[i]

    while i < vector_length and last_element == current_element:
        last_element = current_element
        i += 1
        current_element = tmp_polygon_vector[i]

    while i < vector_length and last_element == current_element + 1:
        last_element = current_element
        i += 1
        current_element = tmp_polygon_vector[i]

    return True if current_element == -1 else False


def convert_polygon_vector_to_matrices(polygon_vector):
    """
    Converts a polygon vector into one or two matrix representations.

    The method generates right-leaning and optionally left-leaning matrix
    representations depending on the vector.

    Args:
        polygon_vector (numpy.ndarray): The vector representation of the polygon.

    Returns:
        list: A list of NumPy arrays representing the polygon in matrix form.
    """
    height = len(polygon_vector)
    width = np.max(polygon_vector)

    right_leaning_matrix = np.zeros([height, width])
    # matrix for symmetrical and left-leaning polygon (always exists for validated vector)
    max_val_counter = 0
    offset = 0
    for row in range(height):

        if polygon_vector[row] == width:
            max_val_counter += 1  # only max value can duplicate

        if row != 0 and polygon_vector[row] < polygon_vector[row - 1]:
            offset += 1

        for column in range(offset, offset + polygon_vector[row]):
            right_leaning_matrix[row][column] = 1

    if max_val_counter > 1:
        first_max_row = np.argmax(polygon_vector)
        left_leaning_matrix = np.hstack((np.array(right_leaning_matrix), np.zeros([height, max_val_counter - 1])))
        left_offset = 0
        for row in range(first_max_row + 1, height):
            if polygon_vector[row] == width:
                left_offset += 1
            left_leaning_matrix[row] = np.roll(left_leaning_matrix[row], left_offset)

        return [right_leaning_matrix, left_leaning_matrix]

    return [right_leaning_matrix]


def check_if_vector_fits_on_board(vector, board_size):
    """
    Checks if a validated polygon vector can fit on a board of a given size.

    Args:
        vector (numpy.ndarray): A validated polygon vector.
        board_size (int): The size of the board.

    Returns:
        bool: True if the vector fits, False otherwise.
    """
    max_row_length = np.max(vector)
    max_row_count = np.count_nonzero(vector == max_row_length)
    # check if polygon fits horizontally
    if max_row_length + max_row_count - 1 > board_size:
        return False
    # check if polygon fits vertically
    min_row_length = np.min(vector)
    row_length_diff = max_row_length - min_row_length

    return max_row_count + row_length_diff - (board_size - max_row_length) <= (board_size + 1) / 2


def get_polygon_vectors(polygon_size, board_size):
    """
    Generates all possible polygon vectors of a given size that fit on a board.

    Args:
        polygon_size (int): The size of the polygon (sum of vector elements).
        board_size (int): The size of the board.

    Returns:
        list: A list of NumPy arrays representing all valid polygon vectors.
    """
    def helper(remaining, current):
        if len(current) > board_size:
            return
        if remaining == 0 and validate_polygon(current) and check_if_vector_fits_on_board(current, board_size):
            polygon_vectors.append(np.array(current))
            return
        for i in range(1, remaining + 1):
            helper(remaining - i, current + [i])

    polygon_vectors = []
    helper(polygon_size, [])

    return polygon_vectors
