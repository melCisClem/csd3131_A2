"""Slow but obviously-correct reference ("oracle") shared by the tests.

The real code finds the best seam cleverly (DP, max-flow). To prove it is right
we compare it with the dumbest possible method: list EVERY valid seam and take
the cheapest. That is exponential, so it is only used on tiny images.
"""
import numpy as np


def all_vertical_seams(height: int, width: int):
    # Yield every valid vertical seam as a tuple of columns, one per row:
    # start in any column, then move to the same or an adjacent column each row.
    def extend(seam):
        if len(seam) == height:
            yield tuple(seam)
            return
        for column in (seam[-1] - 1, seam[-1], seam[-1] + 1):
            if 0 <= column < width:
                yield from extend(seam + [column])

    for first_column in range(width):
        yield from extend([first_column])


def brute_force_min_cost(energy) -> float:
    # Cost of the cheapest seam, found by trying all of them.
    energy = np.asarray(energy, dtype=float)
    height, width = energy.shape
    rows = np.arange(height)
    return min(float(energy[rows, list(seam)].sum()) for seam in all_vertical_seams(height, width))


def count_seams(height: int, width: int) -> int:
    # Number of valid seams, counted with a small DP instead of listing them:
    # ways[j] = number of seams that end in column j of the current row.
    ways = [1] * width
    for _ in range(height - 1):
        ways = [sum(ways[max(j - 1, 0):j + 2]) for j in range(width)]
    return sum(ways)
