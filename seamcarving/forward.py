"""Forward-energy DP seam (Rubinstein, Shamir & Avidan, 2008).

Backward energy (backward.py) scores a pixel by how different it is from its
neighbours *now*. Forward energy instead scores a seam by the new differences
it *creates*: after removing pixel (i, j), pixels that were not adjacent become
neighbours. The cost of a step depends on which way the seam moves:

    cost_if_from_up        = |I[i, j+1] - I[i, j-1]|              (new left/right pair)
    cost_if_from_up_left   = cost_if_from_up + |I[i-1, j] - I[i, j-1]|
    cost_if_from_up_right  = cost_if_from_up + |I[i-1, j] - I[i, j+1]|

    M[0, j] = cost_if_from_up(0, j)
    M[i, j] = min(M[i-1, j-1] + cost_if_from_up_left,
                  M[i-1, j]   + cost_if_from_up,
                  M[i-1, j+1] + cost_if_from_up_right)

Names in the code:  M = cost_table,  backpointers = came_from_column.

Same O(H*W) DP as backward.py, but it needs the image itself, not just an energy
map, so its signature is img -> seam rather than energy -> seam.
"""
import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]              # BGRA/RGBA: drop the alpha (transparency) channel
    grayscale = img.astype(np.float64)
    return grayscale.mean(axis=2) if grayscale.ndim == 3 else grayscale


def forward_costs(grayscale: np.ndarray):
    # Return (cost_if_from_up, cost_if_from_up_left, cost_if_from_up_right),
    # each (height, width): the cost of the seam entering pixel (row, col).
    pixel_left = np.concatenate([grayscale[:, :1], grayscale[:, :-1]], axis=1)    # I[i, j-1], clamped at the edge
    pixel_right = np.concatenate([grayscale[:, 1:], grayscale[:, -1:]], axis=1)   # I[i, j+1], clamped at the edge
    pixel_above = np.vstack([grayscale[:1], grayscale[:-1]])                      # I[i-1, j] (row 0 unused)

    cost_if_from_up = np.abs(pixel_right - pixel_left)
    cost_if_from_up_left = cost_if_from_up + np.abs(pixel_above - pixel_left)
    cost_if_from_up_right = cost_if_from_up + np.abs(pixel_above - pixel_right)
    return cost_if_from_up, cost_if_from_up_left, cost_if_from_up_right


def find_vertical_seam_forward(img: np.ndarray) -> np.ndarray:
    # Minimum forward-energy vertical seam. Returns int array of length height.
    grayscale = _to_grayscale(img)
    height, width = grayscale.shape
    cost_if_from_up, cost_if_from_up_left, cost_if_from_up_right = forward_costs(grayscale)

    cost_table = np.empty((height, width), dtype=np.float64)
    cost_table[0] = cost_if_from_up[0]       # base case: first row has no row above
    came_from_column = np.zeros((height, width), dtype=np.int64)
    all_columns = np.arange(width)
    infinity = np.array([np.inf])            # "off the edge" so it is never chosen

    for row in range(1, height):
        previous_row_cost = cost_table[row - 1]
        # Total seam cost arriving from each of the three parents (same layout as backward.py)
        cost_from_each_parent = np.stack([
            np.concatenate([infinity, previous_row_cost[:-1]]) + cost_if_from_up_left[row],
            previous_row_cost + cost_if_from_up[row],
            np.concatenate([previous_row_cost[1:], infinity]) + cost_if_from_up_right[row],
        ])
        cheapest_parent = cost_from_each_parent.argmin(axis=0)   # 0 = up-left, 1 = up, 2 = up-right
        came_from_column[row] = all_columns + (cheapest_parent - 1)
        cost_table[row] = cost_from_each_parent[cheapest_parent, all_columns]

    seam = np.empty(height, dtype=np.int64)
    seam[-1] = cost_table[-1].argmin()       # bottom pixel: column with the cheapest total
    for row in range(height - 1, 0, -1):     # walk up, following the "came from" pointers
        seam[row - 1] = came_from_column[row, seam[row]]
    return seam


def forward_seam_cost(img: np.ndarray, seam: np.ndarray) -> float:
    # Forward-energy cost of a given vertical seam (for tests/comparisons).
    cost_if_from_up, cost_if_from_up_left, cost_if_from_up_right = forward_costs(_to_grayscale(img))
    cost_for_each_shift = {   # keyed by seam[row] - seam[row-1]
        1: cost_if_from_up_left,     # moved right: came from up-left
        0: cost_if_from_up,          # same column: came from up
        -1: cost_if_from_up_right,   # moved left: came from up-right
    }
    total_cost = cost_if_from_up[0, seam[0]]
    for row in range(1, len(seam)):
        column_shift = seam[row] - seam[row - 1]
        total_cost += cost_for_each_shift[column_shift][row, seam[row]]
    return float(total_cost)
