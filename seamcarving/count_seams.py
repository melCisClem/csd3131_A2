"""Q2a(i): count how many vertical seams an image of a given size has.

ways[j] = number of partial seams that start anywhere in the top row and end in
column j of the current row. Each row, a column can be reached from up to three
columns above it (j-1, j, j+1), the same moves the DP seam finder uses.

    python -m seamcarving.count_seams 229 341     # castle.png: height, width

The brute-force check lists every column sequence and keeps the ones that
is_valid_seam (carve.py) accepts, so the count uses the project's own seam rule.
"""
import itertools
from decimal import Decimal
import sys

import numpy as np

from .carve import is_valid_seam


def count_seams(height: int, width: int) -> int:
    ways = [1] * width                       # row 0: every pixel starts one seam
    for _ in range(height - 1):
        ways = [sum(ways[k] for k in (j - 1, j, j + 1) if 0 <= k < width)
                for j in range(width)]
    return sum(ways)                         # Python ints never overflow


def count_seams_brute_force(height: int, width: int) -> int:
    # Only for tiny images: tries all width**height column sequences.
    return sum(is_valid_seam(np.array(cols), height, width)
               for cols in itertools.product(range(width), repeat=height))


if __name__ == "__main__":
    h, w = int(sys.argv[1]), int(sys.argv[2])
    total = count_seams(h, w)
    lower = w * 2 ** (h - 1)                 # bound proved in the report

    def sci(x: int) -> str:                  # big int -> "1.96E+111" (Decimal avoids float overflow)
        return str(x) if x < 10**6 else format(Decimal(x), ".2E")

    print(f"{h} x {w} image: {sci(total)} seams (proved lower bound n*2^(m-1) = {sci(lower)})")
