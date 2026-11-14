"""Tests for the greedy seam finder (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import carve, is_valid_seam, seam_cost
from seamcarving.backward import find_vertical_seam as dp_seam
from seamcarving.greedy import find_horizontal_seam, find_vertical_seam


def random_energy(height=24, width=31, seed=0):
    return np.random.default_rng(seed).random((height, width)) * 100


@pytest.mark.parametrize("seed", range(20))
def test_seam_is_valid_and_never_beats_dp(seed):
    energy = random_energy(seed=seed)
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, *energy.shape)
    assert seam_cost(energy, seam) >= seam_cost(energy, dp_seam(energy)) - 1e-9


def test_follows_local_minimum_row_by_row():
    energy = np.array([
        [5, 1, 5, 5],
        [9, 2, 0, 9],   # From column 1: neighbours 9, 2, 0 -> column 2
        [9, 9, 9, 3],   # From column 2: neighbours 9, 9, 3 -> column 3
    ], dtype=float)
    
    assert find_vertical_seam(energy).tolist() == [1, 2, 3]


def test_greedy_trap_where_dp_wins():
    # The cheap start in column 0 leads into a wall of 100s
    # DP pays a bit more up top and goes down column 3 instead
    energy = np.array([
        [0, 9, 9, 1],
        [1, 9, 9, 1],
        [100, 100, 9, 1],
        [100, 100, 9, 1],
    ], dtype=float)

    assert find_vertical_seam(energy).tolist()[0] == 0
    assert seam_cost(energy, find_vertical_seam(energy)) > seam_cost(energy, dp_seam(energy)) == 4


def test_edges_and_single_column():
    assert find_vertical_seam(np.array([[3.0], [1.0], [2.0]])).tolist() == [0, 0, 0]
    energy = np.array([[9, 0], [9, 0], [0, 9]], dtype=float)   # Hugs the right edge, then steps left
    assert find_vertical_seam(energy).tolist() == [1, 1, 0]


def test_horizontal_seam_is_vertical_seam_of_transpose():
    energy = random_energy(17, 23, seed=3)
    seam = find_horizontal_seam(energy)
    assert seam.shape == (23,)
    assert np.array_equal(seam, find_vertical_seam(energy.T))


def test_carve_with_greedy():
    img = np.random.default_rng(1).integers(0, 256, (24, 31, 3), dtype=np.uint8)
    assert carve(img, 20, 15, seam_finder=find_vertical_seam).shape == (15, 20, 3)
