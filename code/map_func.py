from typing import Tuple
from grid import GridMap, Grid


# =========================== MAP FUNCTIONS ========================== #
def grid_index_map_func(g: Grid, grid_map: GridMap):
    """
    Map a grid to its index: (i, j) => int
    return: i*|column|+j
    """
    i, j = g.index
    return i * len(grid_map.map[0]) + j


def pair_grid_index_map_func(grid_pair: Tuple[Grid, Grid], grid_map: GridMap):
    """
    Map a pair of grid to index: (g1, g2) => (i1, i2) => int
    Firstly map (g1, g2) to a matrix of [N x N], where N is
    the total number of grids
    return: i1 * N + i2
    """
    g1, g2 = grid_pair
    index1 = grid_index_map_func(g1, grid_map)
    index2 = grid_index_map_func(g2, grid_map)

    return index1 * grid_map.size + index2


def adjacent_pair_grid_map_func(grid_pair: Tuple[Grid, Grid], grid_map: GridMap):
    """
    Map a pair of adjacent grid to index: (g1, g2) => (j1, j2) => int
    Firstly map (g1, g2) to a matrix of [N x 8], where N is
    the total number of grids
    |0|1|2|
    |3|-|4|
    |5|6|7|
    return: j1 * 8 + j2
    """
    g1, g2 = grid_pair
    if not grid_map.is_adjacent_grids(g1, g2):
        return -1

    index1 = grid_index_map_func(g1, grid_map)
    i1, j1 = g1.index
    i2, j2 = g2.index

    if j2 == j1 + 1:
        index2 = i2 - i1 + 1
    elif j2 == j1:
        index2 = 3 if i2 == i1 - 1 else 4
    else:
        index2 = i2 - i1 + 6

    return index1 * 8 + index2


def grid_index_inv_func(index: int, grid_map: GridMap):
    """
    Inverse function of grid_index_map_func
    """
    i = index // len(grid_map.map[0])
    j = index % len(grid_map.map[0])
    return grid_map.map[i][j]


def pair_grid_index_inv_func(index: int, grid_map: GridMap):
    """
    Inverse function of pair_grid_index_map_func
    """
    index1 = index // grid_map.size
    index2 = index % grid_map.size
    return grid_index_inv_func(index1, grid_map), grid_index_inv_func(index2, grid_map)


def adjacent_pair_grid_inv_func(index: int, grid_map: GridMap):
    """
    Inverse function of adjacent_pair_grid_map_func
    """
    index1 = index // 8
    g1 = grid_index_inv_func(index1, grid_map)
    i1, j1 = g1.index
    index2 = index % 8

    if 0 <= index2 <= 2:
        j2 = j1 + 1
        i2 = index2 + i1 - 1
    elif 3 <= index2 <= 4:
        j2 = j1
        i2 = i1 - 1 if index2 == 3 else i1 + 1
    else:
        j2 = j1 - 1
        i2 = index2 + i1 - 6

    # Out of bound
    if not (0 <= i2 < len(grid_map.map) and 0 <= j2 < len(grid_map.map[0])):
        return g1, None
    g2 = grid_map.map[i2][j2]
    return g1, g2



