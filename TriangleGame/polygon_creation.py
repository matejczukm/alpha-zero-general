import numpy as np
from itertools import permutations


def starting_value(k):
    """
    Calculates the size of the smallest triangle from which k-element polygon can be made.

    Args:
        k (int): The size of polygon to be made.

    Returns:
        int: The size of the smallest triangle, from which k-element polygon can be made.
    """
    return int(
        np.ceil(
            np.sqrt(k)
        )
    )


def end_value(k):
    """
    Calculates the size of the biggest triangle from which k-element polygon can be made, unequivocally.

    Args:
        k (int): The size of polygon to be made.

    Returns:
        int: The size of the biggest triangle, from which k-element polygon can be made, unequivocally.
    """
    return int(
        max(np.floor(np.sqrt(k) * 2), np.ceil(k/2)+1)
        )


def get_range(k):
    """
    Calculates the range of triangle sizes from which k-element polygon can be made, unequivocally.

    Args:
        k (int): The size of polygon to be made.

    Returns:
        range: The range of triangle sizes from which a k-element polygon can be made, unequivocally.
    """
    return range(
        starting_value(k),
        end_value(k) + 1,
        1
    )


def get_unique_solutions(k):
    """
    Generates all unique encoded shapes of polygons of size k.

    Args:
        k (int): The size of polygon to be made.

    Returns:
        list[tuple[int]]: A list of tuples, where each tuple represents the encoding of suitable polygon.
    """
    solutions = []
    # 4 nested loops might look scary, but since k won't be (much) greater than 10 (I think), it should be ok,
    # at least for now.
    for i in get_range(k):
        for a in range(i):
            for b in range(a + 1):
                for c in range(b + 1):
                    if i ** 2 - a ** 2 - b ** 2 - c ** 2 == k:
                        if a + b <= i:
                            solutions.append((i, a, b, c))
    return solutions


def get_all_solutions(k):
    """
    Generates all encoded polygons of size k, including all rotations.

    Args:
        k (int): The size of polygon to be made.

    Returns:
        list[tuple[int]]: A list of tuples, where each tuple represents the encoding of suitable polygon.
    """
    solutions = []
    # 5 nested loops might look even worse, but it's faster solution then the second approach (in
    # alternative_functions.ipynb).
    for i in get_range(k):
        for a in range(i):
            for b in range(a + 1):
                for c in range(b + 1):
                    if i ** 2 - a ** 2 - b ** 2 - c ** 2 == k and a + b <= i:
                        # if a + b <= i:
                        solutions.extend(
                            [(i,) + perm for perm in set(permutations((a, b, c)))]
                        )
    return solutions


def get_triangle_matrix(k, value=0):
    """
    Generates a matrix-like representation of a triangle of size k.

    Args:
        k (int): The size of triangle to be made.
        value (int, optional) = 0: 0 or 1 indicating whether triangle should be full or empty.

    Returns:
        list[list[int]]: A list of lists, filled with zeros or ones.
    """
    return [
        [value for i in range(2 * j + 1)]
        for j in range(k)
    ]
