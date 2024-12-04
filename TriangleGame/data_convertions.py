import copy
import numpy as np
from . import polygon_creation as pc


def decode_solution(i, a, b, c):
    """
    Converts the encoded representation of a polygon into a matrix-like representation of the polygon.

    Args:
        i (int): The size of base triangle.
        a (int): The size of the triangle to be cut from the top.
        b (int): The size of the triangle to be cut from the bottom left
        c (int): The size of the triangle to be cut from the bottom right

    Returns:
        list[list[int]]: A list representation of polygon.
    """
    base_triangle = pc.get_triangle_matrix(i, 1)

    for row in range(a):
        for col in range(2 * row + 1):
            base_triangle[row][col] = 0

    for row in range(b):
        for col in range(2 * row + 1):
            base_triangle[-(b - row)][col] = 0

    for row in range(c):
        for col in range(2 * row + 1):
            base_triangle[-(c - row)][-(1 + col)] = 0

    return base_triangle


def convert_triangle_to_matrix(triangle, value=0):
    """
    Converts a triangle in matrix-like representation to a full matrix.
    Populates missing elements with a specified value.

    Args:
        triangle (list[list[int]]): A matrix-like representation of a triangle.
        value (Union[int, float], optional): The value to place in positions above the main diagonal.
                                             Can be `0` or `np.inf`. Defaults to `0`.

    Returns:
        list[list[int]]: A full matrix representation of the triangle.
    """
    assert isinstance(triangle, list), f'Invalid data types. triangle: {type(triangle)}'

    triangle = copy.deepcopy(triangle)
    for i in range(1, len(triangle)):
        triangle[-(i + 1)].extend([value, value] * i)
    return triangle


def convert_triangle_to_numpy_array(triangle, value=0):
    """
    Converts a triangle in matrix-like representation to a full numpy matrix.
    Populates missing elements with a specified value.

    Args:
        triangle (list[list[int]]): A matrix-like representation of a triangle.
        value (Union[int, float], optional): The value to place in positions above the main diagonal.
                                             Can be `0` or `np.inf`. Defaults to `0`.

    Returns:
        numpy.ndarray: A full matrix representation of the triangle.
    """
    assert isinstance(triangle, list), f'Invalid data types. triangle: {type(triangle)}'

    return np.array(
        convert_triangle_to_matrix(triangle, value)
    )


def convert_numpy_array_to_triangle(np_triangle):
    """
    Converts NumPy array representation of triangle to matrix-like representation.

    Args:
        np_triangle (numpy.ndarray): A NumPy array representation of triangle.

    Returns:
        list[list[int]]: A matrix-like representation of a triangle.
    """
    assert isinstance(np_triangle, np.ndarray), f'Invalid data types. np_triangle: {type(np_triangle)}'

    triangle = []
    for i, row in enumerate(np_triangle):
        triangle.append(row.tolist()[:(1 + 2 * i)])
    return triangle
