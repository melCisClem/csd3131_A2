"""Tests for the forward-energy DP seam (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import carve, find_vertical_seam_forward, forward_seam_cost, is_valid_seam
from seam_oracle import all_vertical_seams


def random_gray(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)


def reference_forward_cost(gray, seam):
    # The forward-energy formula written out pixel by pixel, straight from the
    # README (neighbours outside the image are clamped to the edge pixel)
    gray = np.asarray(gray, dtype=float)
    width = gray.shape[1]
    total = 0.0
    for row, col in enumerate(seam):
        left = gray[row, max(col - 1, 0)]
        right = gray[row, min(col + 1, width - 1)]
        total += abs(right - left)                     # new left/right neighbours
        if row > 0:
            above = gray[row - 1, col]
            if col == seam[row - 1] + 1:               # seam came from up-left
                total += abs(above - left)
            elif col == seam[row - 1] - 1:             # seam came from up-right
                total += abs(above - right)
    return total


def test_worked_example():
    # Removing the 5 and the pixel below-left of it leaves [0 0] over [0 9]:
    # no new difference is created next to a removed pixel, so the cost is 0
    gray = np.array([
        [0, 5, 0],
        [0, 0, 9],
    ], dtype=np.uint8)
    seam = find_vertical_seam_forward(gray)
    assert seam.tolist() == [1, 0]
    assert forward_seam_cost(gray, seam) == 0


@pytest.mark.parametrize("seed", range(5))
def test_cost_function_matches_the_formula(seed):
    gray = random_gray(4, 4, seed=seed)
    for seam in all_vertical_seams(4, 4):
        assert forward_seam_cost(gray, np.array(seam)) == pytest.approx(reference_forward_cost(gray, seam))


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (6, 1), (2, 2), (4, 3), (3, 7), (6, 5)])
@pytest.mark.parametrize("seed", range(5))
def test_seam_is_optimal(shape, seed):
    # Cheapest of ALL seams under the hand-written formula = what the DP found
    gray = random_gray(*shape, seed=seed)
    seam = find_vertical_seam_forward(gray)
    assert is_valid_seam(seam, *shape)
    cheapest = min(reference_forward_cost(gray, s) for s in all_vertical_seams(*shape))
    assert reference_forward_cost(gray, seam) == pytest.approx(cheapest)


def test_flat_image_costs_nothing():
    gray = np.full((5, 6), 7, dtype=np.uint8)
    seam = find_vertical_seam_forward(gray)
    assert is_valid_seam(seam, 5, 6)
    assert forward_seam_cost(gray, seam) == 0


def test_colour_and_alpha_use_the_channel_average():
    bgra = np.random.default_rng(1).integers(0, 256, (8, 9, 4), dtype=np.uint8)
    bgr = bgra[:, :, :3]
    seam = find_vertical_seam_forward(bgr)
    assert np.array_equal(seam, find_vertical_seam_forward(bgra))
    assert np.array_equal(seam, find_vertical_seam_forward(bgr.astype(np.float64).mean(axis=2)))


def test_carve_forward_does_not_use_the_seam_finder():
    def must_not_be_called(energy):
        raise AssertionError("forward=True should not use seam_finder")

    img = np.random.default_rng(2).integers(0, 256, (12, 15, 3), dtype=np.uint8)
    assert carve(img, 11, 9, seam_finder=must_not_be_called, forward=True).shape == (9, 11, 3)
