"""Greedy seam (assignment item 3): an alternative to the DP in backward.py.

Instead of looking at the whole image, the seam is built one row at a time by
a purely local decision:

    seam[0] = argmin_j e[0, j]                  (cheapest pixel of the top row)
    seam[i] = argmin over j in {seam[i-1]-1, seam[i-1], seam[i-1]+1} of e[i, j]

From the pixel chosen in the row above, step down to whichever of its
(up to) three neighbours below has the lowest energy. A decision is never
revisited, so a cheap pixel now can lead the seam into an expensive region
later -- the DP avoids this because M[i, j] already accounts for every row above.

Time O(W + H) per seam for the search itself (DP: O(H*W)), space O(H) for the
seam only (DP: an H x W table plus backpointers). In carve() the energy map is
still recomputed every seam, which is O(H*W) for both methods.

Same signature and seam format as backward.find_vertical_seam, so it plugs
straight into carve(img, w, h, seam_finder=greedy.find_vertical_seam).
"""

import numpy as np


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    # Greedy vertical seam
    # Returns int array of length height
    height, width = energy.shape
    seam = np.empty(height, dtype=np.int64)
    seam[0] = energy[0].argmin()                 # Start at the cheapest pixel of the top row
    for row in range(1, height):
        previous_column = seam[row - 1]

        # The three pixels below the previous one, clipped at the image edges
        first_candidate = max(previous_column - 1, 0)
        last_candidate = min(previous_column + 1, width - 1)
        candidates = energy[row, first_candidate:last_candidate + 1]

        # Cheapest of them (the leftmost one wins ties, same as the DP)
        seam[row] = first_candidate + candidates.argmin()
    return seam


def find_horizontal_seam(energy: np.ndarray) -> np.ndarray:
    # Greedy horizontal seam: seam[col] = row removed in column col
    return find_vertical_seam(energy.T)  # Transpose the energy so the same code can be reused
