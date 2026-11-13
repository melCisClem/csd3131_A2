"""Step (c): remove the seam, recompute energy, repeat.

`seam_finder` is any function  energy (H, W) -> vertical seam (int array, len H).
Greedy and graph-cut implementations just need that signature to be plugged in:

    carve(img, 300, 200, seam_finder=greedy.find_vertical_seam)

Horizontal seams are handled by transposing the image, so a seam finder only
ever has to know about vertical seams.
"""
from typing import Callable, Optional

import numpy as np

from .backward import find_vertical_seam as backward_find_vertical_seam
from .energy import compute_energy
from .forward import find_vertical_seam_forward

SeamFinder = Callable[[np.ndarray], np.ndarray]


def remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    # Delete seam[i] from each row i -> image one pixel narrower.
    height, width = img.shape[:2]
    keep_pixel = np.ones((height, width), dtype=bool)         # True = keep this pixel
    keep_pixel[np.arange(height), seam] = False               # in row i, mark column seam[i] for deletion
    # img[keep_pixel] gives the kept pixels row by row; each row now has width-1 of them
    return img[keep_pixel].reshape((height, width - 1) + img.shape[2:])


def remove_horizontal_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    # Delete seam[j] from each column j -> image one pixel shorter.
    return np.swapaxes(remove_vertical_seam(np.swapaxes(img, 0, 1), seam), 0, 1)


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    # Total energy along a vertical seam (what DP minimises).
    return float(energy[np.arange(energy.shape[0]), seam].sum())


def is_valid_seam(seam: np.ndarray, height: int, width: int) -> bool:
    # One pixel per row, in range, successive pixels in same/adjacent column.
    seam = np.asarray(seam)
    return (
        seam.shape == (height,)
        and seam.min() >= 0
        and seam.max() < width
        and (height < 2 or np.abs(np.diff(seam)).max() <= 1)
    )


def carve(
    img: np.ndarray,
    new_width: int,
    new_height: int,
    seam_finder: SeamFinder = backward_find_vertical_seam,
    on_seam: Optional[Callable[[str, np.ndarray, np.ndarray], None]] = None,
    forward: bool = False,
) -> np.ndarray:
    """Shrink img to (new_height, new_width) by removing vertical then horizontal seams.

    on_seam(kind, seam, img_before) is called before every removal, with
    kind in {"vertical", "horizontal"} and the seam in the coordinates of
    img_before -- handy for animations / step-by-step visualisation.

    forward=True uses forward-energy DP (forward.py) instead of seam_finder;
    it works on the image itself rather than a precomputed energy map.
    """
    original_height, original_width = img.shape[:2]
    if not (1 <= new_width <= original_width and 1 <= new_height <= original_height):
        raise ValueError(
            f"target {new_width}x{new_height} must be between 1x1 and "
            f"{original_width}x{original_height} (this implementation only shrinks)"
        )

    def find_vertical_seam_of(image: np.ndarray) -> np.ndarray:
        # one seam for the given image, using whichever method was chosen
        if forward:
            return find_vertical_seam_forward(image)
        return seam_finder(compute_energy(image))

    number_of_vertical_seams = original_width - new_width
    number_of_horizontal_seams = original_height - new_height

    for _ in range(number_of_vertical_seams):
        seam = find_vertical_seam_of(img)
        if on_seam:
            on_seam("vertical", seam, img)
        img = remove_vertical_seam(img, seam)

    for _ in range(number_of_horizontal_seams):
        transposed_img = np.swapaxes(img, 0, 1)   # rows <-> columns: horizontal seam becomes vertical
        seam = find_vertical_seam_of(transposed_img)
        if on_seam:
            on_seam("horizontal", seam, img)
        img = remove_horizontal_seam(img, seam)

    return img
