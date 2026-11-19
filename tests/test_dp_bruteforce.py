"""Tests for the DP seam finder in backward.py, used in report section 2(a)(ii) (run with: pytest)."""
import itertools

import numpy as np
import pytest

from seamcarving import is_valid_seam, seam_cost
from seamcarving.backward import cumulative_energy, find_vertical_seam


def brute_force_min_cost(energy):
    # Try every column sequence and keep the cheapest valid seam (tiny images only).
    height, width = energy.shape
    return min(seam_cost(energy, np.array(cols))
               for cols in itertools.product(range(width), repeat=height)
               if is_valid_seam(np.array(cols), height, width))


def test_worked_example_from_report():
    energy = np.array([[1, 4, 3], [5, 2, 6], [3, 7, 1]], dtype=float)
    cost_table, came_from_column = cumulative_energy(energy)
    assert cost_table.tolist() == [[1, 4, 3], [6, 3, 9], [6, 10, 4]]
    assert came_from_column[1:].tolist() == [[0, 0, 2], [1, 1, 1]]
    assert find_vertical_seam(energy).tolist() == [0, 1, 2]   # columns 1, 2, 3 in the report


@pytest.mark.parametrize("seed", range(500))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    height, width = rng.integers(1, 7), rng.integers(1, 6)
    energy = rng.integers(0, 10, size=(height, width)).astype(float)
    seam = find_vertical_seam(energy)
    cost_table, _ = cumulative_energy(energy)
    best = brute_force_min_cost(energy)
    assert is_valid_seam(seam, height, width)
    assert seam_cost(energy, seam) == pytest.approx(best)       # the returned seam is optimal
    assert cost_table[-1].min() == pytest.approx(best)          # and the table reports that cost


def test_ties_choose_leftmost_parent():
    energy = np.zeros((3, 4))
    assert find_vertical_seam(energy).tolist() == [0, 0, 0]
