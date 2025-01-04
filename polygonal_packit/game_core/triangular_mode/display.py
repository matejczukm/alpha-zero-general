from . import data_conversions as dc


def print_triangle(triangle):
    """
    Prints a matrix-like representation of a triangle.

    Args:
        triangle (list[list[int]]): A matrix-like representation of a triangle.
    """
    assert isinstance(triangle, list) , f'Invalid data types. board: {type(triangle)}'


    for row in triangle:
        s = (len(triangle[-1]) - len(row))//2
        print('   '*s + str(row))


def print_numpy_triangle(np_triangle):
    """
    Prints a NumPy array representation of a triangle.

    Args:
        np_triangle (numpy.ndarray): A matrix-like representation of a triangle.
    """
    print_triangle(
        dc.convert_numpy_array_to_triangle(np_triangle)
    )


def print_board(board):
    num_cells = 1
    max_num_cells = len(board) * 2 +1
    separators = ('/', '\\')
    for row in board:
        row_repr = ' ' * (max_num_cells - num_cells)
        row_repr += separators[0]
        for i in range(num_cells):
            row_repr += str(row[i]) 
            row_repr += separators[(i+1)%2]
        num_cells += 2
        # row_repr += separators[1]
        print(row_repr)