"""Tests for the backward-energy DP seam, step (b) (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import (
    cumulative_energy,
    find_horizontal_seam,
    find_vertical_seam,
    is_valid_seam,
    seam_cost,
)
from seam_oracle import all_vertical_seams, brute_force_min_cost, count_seams

SHAPES = [(1, 1), (1, 6), (6, 1), (2, 2), (4, 3), (3, 7), (6, 5)]


def random_energy(height, width, seed=0):
    return np.random.default_rng(seed).random((height, width)) * 100


def test_worked_example():
    # Small enough to fill in the table M by hand
    energy = np.array([
        [3, 1, 4],
        [1, 5, 9],
        [2, 6, 5],
    ], dtype=float)
    cost_table, came_from_column = cumulative_energy(energy)
    assert np.array_equal(cost_table, [
        [3, 1, 4],
        [2, 6, 10],    # 1+1, 5+1, 9+1
        [4, 8, 11],    # 2+2, 6+2, 5+6
    ])
    assert np.array_equal(came_from_column[1:], [[1, 1, 1], [0, 0, 1]])
    seam = find_vertical_seam(energy)
    assert seam.tolist() == [1, 0, 0]
    assert seam_cost(energy, seam) == 4


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(5))
def test_seam_is_optimal(shape, seed):
    # The DP seam costs exactly as much as the cheapest of ALL possible seams
    energy = random_energy(*shape, seed=seed)
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, *shape)
    assert np.issubdtype(seam.dtype, np.integer)
    assert seam_cost(energy, seam) == pytest.approx(brute_force_min_cost(energy))


@pytest.mark.parametrize("seed", range(10))
def test_seam_is_optimal_with_many_ties(seed):
    # Whole-number energies 0..3: lots of equal-cost seams and zero-cost pixels
    energy = np.random.default_rng(seed).integers(0, 4, (6, 6))
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, 6, 6)
    assert seam_cost(energy, seam) == brute_force_min_cost(energy)


@pytest.mark.parametrize("seed", range(5))
def test_cost_table_follows_the_recurrence(seed):
    energy = random_energy(7, 9, seed=seed)
    cost_table, came_from_column = cumulative_energy(energy)

    # M[0][j] = e[0][j];  M[i][j] = e[i][j] + min of the (up to) three cells above
    expected = energy.copy()
    for row in range(1, 7):
        for col in range(9):
            expected[row, col] += expected[row - 1, max(col - 1, 0):col + 2].min()
    np.testing.assert_allclose(cost_table, expected)

    # Every backpointer names an adjacent column, and it is the cheapest one
    for row in range(1, 7):
        for col in range(9):
            parent = came_from_column[row, col]
            assert 0 <= parent < 9 and abs(parent - col) <= 1
            assert cost_table[row - 1, parent] == cost_table[row - 1, max(col - 1, 0):col + 2].min()

    # The answer read off the last row is the cost of the seam that is returned
    assert cost_table[-1].min() == pytest.approx(seam_cost(energy, find_vertical_seam(energy)))


def test_ties_go_to_the_leftmost_seam():
    # Every seam costs the same: the documented tie-break gives the left edge
    assert find_vertical_seam(np.ones((4, 5))).tolist() == [0, 0, 0, 0]


def test_input_energy_is_not_modified():
    energy = random_energy(6, 8)
    before = energy.copy()
    cumulative_energy(energy)
    find_vertical_seam(energy)
    find_horizontal_seam(energy)
    assert np.array_equal(energy, before)


def test_horizontal_worked_example():
    # seam[col] = row removed in that column. Cheapest path left to right:
    # 9 (row 0) -> 1 (row 0) -> 2 (row 1) -> 3 (row 1) = 15
    energy = np.array([
        [9, 1, 9, 9],
        [10, 9, 2, 3],
    ], dtype=float)
    seam = find_horizontal_seam(energy)
    assert seam.tolist() == [0, 0, 1, 1]
    assert seam_cost(energy.T, seam) == 15
    # The vertical seam of the same energy is a different thing: 1 then 2
    assert find_vertical_seam(energy).tolist() == [1, 2]


@pytest.mark.parametrize("shape", SHAPES)
def test_horizontal_seam_is_optimal(shape):
    height, width = shape
    energy = random_energy(height, width, seed=height + width)
    seam = find_horizontal_seam(energy)
    assert is_valid_seam(seam, width, height)          # one row index per column
    assert seam_cost(energy.T, seam) == pytest.approx(brute_force_min_cost(energy.T))


@pytest.mark.parametrize("height", range(1, 8))
@pytest.mark.parametrize("width", [2, 3, 4])
def test_number_of_seams_grows_exponentially(height, width):
    # Report question 2(a)(i): with more than one column every row offers at
    # least two choices, so there are at least 2^(m-1) seams. Listing them all
    # is hopeless on a real image; the DP gets the best one in O(m*n).
    # (This also checks the brute-force helper really lists every seam.)
    listed = sum(1 for _ in all_vertical_seams(height, width))
    assert listed == count_seams(height, width)
    assert listed >= 2 ** (height - 1)
