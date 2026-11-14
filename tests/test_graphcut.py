"""Tests for the graph-cut seam finder (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import carve, is_valid_seam, seam_cost
from seamcarving.backward import find_vertical_seam as dp_seam
from seamcarving.graphcut import (
    find_horizontal_seam,
    find_vertical_seam,
    find_vertical_seam_with_stats,
)


def random_energy(height=12, width=15, seed=0):
    return np.random.default_rng(seed).random((height, width)) * 100


@pytest.mark.parametrize("seed", range(30))
def test_same_cost_as_dp(seed):
    # Min cut = min seam, so the cost must equal DP's (the seam itself may differ on ties)
    energy = random_energy(seed=seed)
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, *energy.shape)
    assert seam_cost(energy, seam) == pytest.approx(seam_cost(energy, dp_seam(energy)))


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (2, 2), (9, 3), (3, 9)])
def test_odd_shapes(shape):
    energy = random_energy(*shape, seed=sum(shape))
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, *shape)
    assert seam_cost(energy, seam) == pytest.approx(seam_cost(energy, dp_seam(energy)))


def test_max_flow_equals_seam_cost():
    # Max-flow min-cut theorem: the flow value is the capacity of the cut found
    energy = random_energy(10, 13, seed=7)
    seam, flow, _, _ = find_vertical_seam_with_stats(energy)
    assert flow == pytest.approx(seam_cost(energy, seam))


def test_last_column_can_be_removed():
    # The cheap pixels are all in the last column, which DP can remove too
    energy = np.full((4, 5), 50.0)
    energy[:, -1] = 1
    assert find_vertical_seam(energy).tolist() == [4, 4, 4, 4]


def test_diagonal_edges_keep_seam_connected():
    # Cheapest pixel per row jumps from column 0 to column 3 (cost 0); a seam may
    # not, so it has to pay 9 in the middle row
    energy = np.array([
        [0, 9, 9, 9],
        [9, 9, 9, 0],
        [0, 9, 9, 9],
    ], dtype=float)
    seam = find_vertical_seam(energy)
    assert is_valid_seam(seam, *energy.shape)
    assert seam_cost(energy, seam) == seam_cost(energy, dp_seam(energy)) == 9


def test_horizontal_seam_is_vertical_seam_of_transpose():
    energy = random_energy(9, 13, seed=3)
    seam = find_horizontal_seam(energy)
    assert seam.shape == (13,)
    assert np.array_equal(seam, find_vertical_seam(energy.T))


def test_carve_with_graphcut():
    img = np.random.default_rng(1).integers(0, 256, (14, 18, 3), dtype=np.uint8)
    out = carve(img, 15, 11, seam_finder=find_vertical_seam)
    assert out.shape == (11, 15, 3)
