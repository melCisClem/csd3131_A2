"""One-seam-at-a-time carving with undo, for the interactive app (app.py).

carve() in carve.py runs from start to finish in one call. An app with a slider
needs something different: remove ONE seam, show it, and be able to put seams
back when the user drags the slider the other way. CarveSession does that.

Key idea: instead of carving the image itself, carve an *index map*.

    index_map[row, col] = which pixel of the ORIGINAL image is currently
                          sitting at (row, col), as a flat index row*W + col

It starts as 0, 1, 2, ... and loses one entry per row for every vertical seam.
From it we get, with no extra bookkeeping:

    current image       = original pixels looked up through index_map
    all removed seams   = every original pixel that is no longer in index_map
    undo                = put the removed indices back where the seam was

The seams themselves come from the same DP code as carve() (backward.py /
forward.py), so stepping a session to WxH gives exactly carve(img, W, H) when
the width is reduced before the height.
"""
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .backward import find_vertical_seam
from .carve import remove_vertical_seam
from .energy import compute_energy
from .forward import find_vertical_seam_forward

VERTICAL = "vertical"       # seam runs top -> bottom, removing it makes the image narrower
HORIZONTAL = "horizontal"   # seam runs left -> right, removing it makes the image shorter

# values stored in CarveSession.removed_by
KEPT, REMOVED_BY_VERTICAL, REMOVED_BY_HORIZONTAL = 0, 1, 2


class Step(NamedTuple):
    """What one call to step_towards() did -- everything the app needs to draw it."""
    action: str         # "remove" or "restore"
    kind: str           # VERTICAL or HORIZONTAL
    seam: np.ndarray    # vertical: seam[row] = column;  horizontal: seam[col] = row
    image: np.ndarray   # the image that contains the seam
                        # (remove: just before removal, restore: just after putting it back)


def insert_vertical_seam(img: np.ndarray, seam: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Exact opposite of remove_vertical_seam: put values[i] back at column seam[i]
    # of row i -> array one column wider. (Only used on the 2-D index map.)
    height, width = img.shape
    is_old_pixel = np.ones((height, width + 1), dtype=bool)
    is_old_pixel[np.arange(height), seam] = False            # the gap the seam left behind
    wider = np.empty((height, width + 1), dtype=img.dtype)
    wider[is_old_pixel] = img.reshape(-1)                    # existing pixels, row by row
    wider[~is_old_pixel] = values                            # one restored pixel per row
    return wider


class CarveSession:
    def __init__(self, original: np.ndarray, forward: bool = False):
        self.original = original
        self.forward = forward      # True = forward-energy DP, False = backward-energy DP
        self.reset()

    def reset(self) -> None:
        # Back to the untouched original.
        height, width = self.original.shape[:2]
        self.index_map = np.arange(height * width).reshape(height, width)
        self.removed_by = np.zeros((height, width), dtype=np.uint8)   # per ORIGINAL pixel: KEPT / REMOVED_BY_...
        # one entry per removed seam, newest last: (kind, seam, original indices of its pixels)
        self.history: List[Tuple[str, np.ndarray, np.ndarray]] = []

    @property
    def image(self) -> np.ndarray:
        # The current (carved) image: look every surviving pixel up in the original.
        flat_original = self.original.reshape((-1,) + self.original.shape[2:])
        return flat_original[self.index_map]

    @property
    def size(self) -> Tuple[int, int]:
        height, width = self.index_map.shape
        return width, height

    def seams_removed(self, kind: str) -> int:
        return sum(1 for seam_kind, _, _ in self.history if seam_kind == kind)

    def step_towards(self, target_width: int, target_height: int) -> Optional[Step]:
        """Do ONE step towards the target size; return None when already there.

        Too small in either direction -> undo the most recent seam (history is a
        stack, so seams come back in the reverse order they were removed).
        Otherwise remove a vertical seam if still too wide, else a horizontal one
        if still too tall -- the same "width first, then height" order as carve().
        """
        width, height = self.size
        if width < target_width or height < target_height:
            return self.restore_last_seam()
        if width > target_width:
            return self.remove_next_seam(VERTICAL)
        if height > target_height:
            return self.remove_next_seam(HORIZONTAL)
        return None

    def remove_next_seam(self, kind: str) -> Step:
        image = self.image
        vertical = kind == VERTICAL
        # A horizontal seam is a vertical seam of the transposed image (same trick as carve()).
        working_image = image if vertical else np.swapaxes(image, 0, 1)
        working_map = self.index_map if vertical else self.index_map.T

        if self.forward:
            seam = find_vertical_seam_forward(working_image)
        else:
            seam = find_vertical_seam(compute_energy(working_image))

        removed_indices = working_map[np.arange(len(seam)), seam]   # original pixels on the seam
        narrower_map = remove_vertical_seam(working_map, seam)
        self.index_map = narrower_map if vertical else np.ascontiguousarray(narrower_map.T)
        self.removed_by.flat[removed_indices] = REMOVED_BY_VERTICAL if vertical else REMOVED_BY_HORIZONTAL
        self.history.append((kind, seam, removed_indices))
        return Step("remove", kind, seam, image)

    def restore_last_seam(self) -> Step:
        kind, seam, removed_indices = self.history.pop()
        vertical = kind == VERTICAL
        working_map = self.index_map if vertical else self.index_map.T
        wider_map = insert_vertical_seam(working_map, seam, removed_indices)
        self.index_map = wider_map if vertical else np.ascontiguousarray(wider_map.T)
        self.removed_by.flat[removed_indices] = KEPT
        return Step("restore", kind, seam, self.image)