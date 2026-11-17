"""Tests that put DP, greedy and graph cut side by side (run with: pytest).

What the report claims, checked on the same energy map every time:
    graph cut cost == DP cost == cheapest of all seams   (both are optimal)
    greedy cost    >= DP cost                            (greedy is not)
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from seamcarving import carve, compute_energy, is_valid_seam, seam_cost
from seamcarving import backward, graphcut, greedy
from seam_oracle import brute_force_min_cost

ROOT = Path(__file__).resolve().parent.parent

FINDERS = {
    "dp": backward.find_vertical_seam,
    "greedy": greedy.find_vertical_seam,
    "graphcut": graphcut.find_vertical_seam,
}
# keyword arguments that make carve() use each method
CARVE_OPTIONS = {
    "dp": {},
    "forward": {"forward": True},
    "greedy": {"seam_finder": greedy.find_vertical_seam},
    "graphcut": {"seam_finder": graphcut.find_vertical_seam},
}


def random_energy(height, width, seed=0):
    return np.random.default_rng(seed).random((height, width)) * 100


@pytest.fixture(scope="module")
def castle_energy():
    # The real test image, shrunk to 34x23 so that graph cut takes a moment, not minutes
    img = cv2.imread(str(ROOT / "castle.png"))
    assert img is not None, "castle.png should be in the repository root"
    return compute_energy(cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA))


@pytest.mark.parametrize("method", FINDERS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (2, 2), (5, 9), (9, 5)])
def test_every_method_returns_a_valid_seam(method, shape):
    seam = FINDERS[method](random_energy(*shape, seed=sum(shape)))
    assert is_valid_seam(seam, *shape)


@pytest.mark.parametrize("seed", range(10))
def test_cost_ordering_on_random_energy(seed):
    energy = random_energy(6, 6, seed=seed)
    cost = {name: seam_cost(energy, finder(energy)) for name, finder in FINDERS.items()}
    assert cost["dp"] == pytest.approx(brute_force_min_cost(energy))
    assert cost["graphcut"] == pytest.approx(cost["dp"])
    assert cost["greedy"] >= cost["dp"] - 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_cost_ordering_with_many_ties(seed):
    # Whole numbers 0..3: many equal-cost seams and zero-capacity edges in the
    # flow network. The seams may differ, the costs may not.
    energy = np.random.default_rng(seed).integers(0, 4, (6, 6)).astype(float)
    cost = {name: seam_cost(energy, finder(energy)) for name, finder in FINDERS.items()}
    assert cost["dp"] == cost["graphcut"] == brute_force_min_cost(energy)
    assert cost["greedy"] >= cost["dp"]


@pytest.mark.parametrize("method", FINDERS)
def test_all_zero_energy(method):
    energy = np.zeros((5, 7))
    seam = FINDERS[method](energy)
    assert is_valid_seam(seam, 5, 7)
    assert seam_cost(energy, seam) == 0


@pytest.mark.parametrize("method", FINDERS)
def test_all_methods_find_an_obvious_seam(method):
    # One cheap column in an expensive image: nobody should miss it
    energy = np.full((6, 7), 50.0)
    energy[:, 2] = 1
    assert FINDERS[method](energy).tolist() == [2] * 6


def test_methods_on_the_castle_image(castle_energy):
    dp_cost = seam_cost(castle_energy, backward.find_vertical_seam(castle_energy))
    seam, flow, _, _ = graphcut.find_vertical_seam_with_stats(castle_energy)
    assert is_valid_seam(seam, *castle_energy.shape)
    assert seam_cost(castle_energy, seam) == pytest.approx(dp_cost)     # same cost as DP
    assert flow == pytest.approx(dp_cost)                               # max flow = min cut
    greedy_seam = greedy.find_vertical_seam(castle_energy)
    assert seam_cost(castle_energy, greedy_seam) >= dp_cost - 1e-9


def test_horizontal_seams_on_the_castle_image(castle_energy):
    height, width = castle_energy.shape
    sideways = castle_energy.T                 # a horizontal seam is a vertical seam of this
    dp_seam = backward.find_horizontal_seam(castle_energy)
    cut_seam = graphcut.find_horizontal_seam(castle_energy)
    greedy_seam = greedy.find_horizontal_seam(castle_energy)
    for seam in (dp_seam, cut_seam, greedy_seam):
        assert is_valid_seam(seam, width, height)
    assert seam_cost(sideways, cut_seam) == pytest.approx(seam_cost(sideways, dp_seam))
    assert seam_cost(sideways, greedy_seam) >= seam_cost(sideways, dp_seam) - 1e-9


def test_graph_cut_stays_optimal_through_a_whole_carve():
    # At every step of a real carve (energy recomputed each time), the seam graph
    # cut removes costs the same as the DP seam on that same image
    img = np.random.default_rng(5).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    checked = []

    def on_seam(kind, seam, before):
        oriented = before if kind == "vertical" else np.swapaxes(before, 0, 1)
        energy = compute_energy(oriented)
        optimal = seam_cost(energy, backward.find_vertical_seam(energy))
        assert seam_cost(energy, seam) == pytest.approx(optimal)
        checked.append(kind)

    carve(img, 8, 8, seam_finder=graphcut.find_vertical_seam, on_seam=on_seam)
    assert checked == ["vertical"] * 4 + ["horizontal"] * 2


@pytest.mark.parametrize("method", CARVE_OPTIONS)
def test_every_method_shrinks_one_pixel_per_seam(method):
    # 12x10 -> 9x8: three vertical seams (width 12, 11, 10), then two horizontal
    # seams (height 10, 9), each a valid seam of the image it is removed from
    img = np.random.default_rng(6).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    seen = []

    def on_seam(kind, seam, before):
        height, width = before.shape[:2]
        seen.append((kind, height, width))
        if kind == "vertical":
            assert is_valid_seam(seam, height, width)
        else:
            assert is_valid_seam(seam, width, height)

    out = carve(img, 9, 8, on_seam=on_seam, **CARVE_OPTIONS[method])
    assert out.shape == (8, 9, 3)
    assert seen == [
        ("vertical", 10, 12), ("vertical", 10, 11), ("vertical", 10, 10),
        ("horizontal", 10, 9), ("horizontal", 9, 9),
    ]
