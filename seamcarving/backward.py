"""Backward-energy DP seam (Avidan & Shamir, 2007).

Step (b): dynamic programming minimum-energy vertical seam. Each pixel is scored
by its own energy (energy.py), i.e. how different it is from its neighbours *now*.

Subproblem: M[i, j] = cost of the cheapest seam that starts anywhere in row 0
and ends at pixel (i, j).

    M[0, j] = e[0, j]
    M[i, j] = e[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])   (out-of-range = inf)

Answer = min_j M[H-1, j]; follow the backpointers up to recover the seam.
Time O(H*W), space O(H*W) (the table + backpointers).

Names in the code:  M = cost_table,  e = energy,  backpointers = came_from_column.

Seam format used by the whole project: int array of length H, seam[i] = column
of the pixel removed in row i, with |seam[i] - seam[i-1]| <= 1.
Horizontal seams are found by running the same code on the transposed energy.
"""
import numpy as np


def cumulative_energy(energy: np.ndarray):
    # Fill the DP table one row at a time (NumPy handles every column of a row at
    # once). Returns (cost_table, came_from_column):
    #   cost_table[row][col]         = cheapest seam ending at pixel (row, col)
    #   came_from_column[row][col]   = column in the row above that seam came from
    height, width = energy.shape
    cost_table = energy.astype(np.float64)   # copy; row 0 is already correct (base case)
    came_from_column = np.zeros((height, width), dtype=np.int64)
    all_columns = np.arange(width)           # [0, 1, ..., width-1]

    for row in range(1, height):
        previous_row_cost = cost_table[row - 1]   # finished costs of the row above

        # For every column: the seam cost arriving from its up-left, up and up-right
        # neighbour. np.inf stands in for "off the edge" so it is never chosen.
        cost_from_up_left = np.concatenate([[np.inf], previous_row_cost[:-1]])
        cost_from_up = previous_row_cost
        cost_from_up_right = np.concatenate([previous_row_cost[1:], [np.inf]])
        cost_from_each_parent = np.stack(
            [cost_from_up_left, cost_from_up, cost_from_up_right]
        )   # shape (3, width)

        # Per column, which parent is cheapest: 0 = up-left, 1 = up, 2 = up-right
        # (the leftmost one wins ties).
        cheapest_parent = cost_from_each_parent.argmin(axis=0)
        column_shift = cheapest_parent - 1   # -1, 0 or +1 columns
        came_from_column[row] = all_columns + column_shift
        cost_table[row] += cost_from_each_parent[cheapest_parent, all_columns]
    return cost_table, came_from_column


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    # Minimum-energy vertical seam via DP. Returns int array of length height.
    height, _ = energy.shape
    cost_table, came_from_column = cumulative_energy(energy)
    seam = np.empty(height, dtype=np.int64)
    seam[-1] = cost_table[-1].argmin()       # bottom pixel: column with the cheapest total
    for row in range(height - 1, 0, -1):     # walk up, following the "came from" pointers
        seam[row - 1] = came_from_column[row, seam[row]]
    return seam


def find_horizontal_seam(energy: np.ndarray) -> np.ndarray:
    # Minimum-energy horizontal seam: seam[col] = row removed in column col.
    return find_vertical_seam(energy.T) # transpose the energy so the DP code can be reused
