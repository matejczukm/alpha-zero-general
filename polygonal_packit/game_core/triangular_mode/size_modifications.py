from . import data_conversions as dc
from . import polygon_creation as pc


def expand_downward(triangle, k):
    """
    Expands given triangle of size n, in matrix-like representation, to a triangle of size n+k
    by adding k rows at the bottom.

    Args:
        triangle (list[list[int]]): A matrix-like representation of the initial triangle to expand.
        k (int): The number of rows to add at the bottom of the triangle.

    Returns:
        list[list[int]]: A matrix-like representation of the expanded triangle.
    """
    # triangle = copy.deepcopy(triangle)
    base = len(triangle[-1])
    for i in range(1, k + 1):
        triangle.append([
            0 for _ in range(base + 2 * i)
        ])
    return triangle


def expand_to_left(triangle, k):
    """
    Expands given triangle of size n, in matrix-like representation, to a triangle of size n+k
    by adding 2*k cells at the beginning of each row, extending the base length to n+k, and
    adding a new triangle of size k at the top to maintain the triangle shape.

    Args:
        triangle (list[list[int]]): A matrix-like representation of the initial triangle to expand.
        k (int): The number of units to add to the left side of each row.

    Returns:
        list[list[int]]: A matrix-like representation of expanded triangle.
    """
    for i in range(len(triangle)):
        # triangle[i] = [0, 0] * k + triangle[i]
        triangle[i] = [0 for _ in range(2*k)] + triangle[i]

    top_part = pc.get_triangle_matrix(k)
    top_part.extend(triangle)
    return top_part


def expand_to_right(triangle, k):
    """
    Expands given triangle of size n, in matrix-like representation, to a triangle of size n+k
    by adding 2*k cells to the end of each row, extending the base length to n+k, and
    adding a new triangle of size k at the top to maintain the triangle shape.

    Args:
        triangle (list[list[int]]): A matrix-like representation of the initial triangle to expand.
        k (int): The number of units to add to the right side of each row.

    Returns:
        list[list[int]]: A matrix-like representation of expanded triangle.
    """
    for i in range(len(triangle)):
        # triangle[i] = triangle[i] + [0, 0] * k
        triangle[i] = triangle[i] + [0 for _ in range(2*k)] 
    top_part = pc.get_triangle_matrix(k)
    top_part.extend(triangle)
    return top_part


def expand_triangle(triangle, expand_left, expand_right, expand_down):
    """
    Expands a given triangle of size n, represented as a matrix, to a triangle of size
    n + expand_left + expand_right + expand_down. The triangle is expanded by adding
    units on the left and right sides of each row and additional rows at the bottom.

    Args:
        triangle (list[list[int]]): A matrix-like representation of the initial triangle to expand.
        expand_left (int): The number of units to add to the left side of each row.
        expand_right (int): The number of units to add to the right side of each row.
        expand_down (int): The number of rows to add at the bottom of the triangle.

    Returns:
        list[list[int]]: A matrix-like representation of the expanded triangle.
    """
    assert isinstance(triangle, list), f'Invalid data types. triangle: {type(triangle)}'

    triangle = expand_to_left(triangle, expand_left)
    triangle = expand_to_right(triangle, expand_right)
    return expand_downward(triangle, expand_down)


def expand_polygon(i, a, b, c, n):
    """
    Converts the encoded representation of a polygon into a all possible matrix-like representations of the polygon inside a triangle of size n.
    Each triangle of size i can be expanded to size n by adding 'j' triangles to the right,
    'k' triangles to the left, and 'l' triangles downward, where j + k + l = n - i.
    We need to find all possible combinations of j, k, and l, and expand the triangle accordingly.

    Args:
        i (int): The size of base triangle.
        a (int): The size of the triangle to be cut from the top.
        b (int): The size of the triangle to be cut from the bottom left
        c (int): The size of the triangle to be cut from the bottom right
        n (int): The desired size of the triangle, in which polygon will be placed.

    Returns:
        list[list[list[int]]]: A list of matrix-like representations of all expanded polygons.
    """
    base_polygon = dc.decode_solution(i, a, b, c)
    if i == n:
        return [base_polygon]

    dif = n - i
    result = []
    for j in range(dif + 1):
        for k in range(dif + 1):
            for l in range(dif + 1):  
                if j + k + l == dif:
                    result.append(expand_triangle(base_polygon.copy(), j, k, l))
    return result
