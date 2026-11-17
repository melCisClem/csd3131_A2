"""Tests for the energy map, step (a) (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import compute_energy


def random_image(*shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 3), (5, 7, 4), (1, 1), (1, 6), (6, 1)])
def test_one_non_negative_float_per_pixel(shape):
    # Grayscale, colour and colour+alpha all give one energy value per pixel
    energy = compute_energy(random_image(*shape))
    assert energy.shape == shape[:2]
    assert energy.dtype == np.float64
    assert np.isfinite(energy).all() and (energy >= 0).all()


def test_flat_image_has_zero_energy():
    # Every pixel equals its neighbours, so nothing is "important"
    assert not compute_energy(np.full((6, 8, 3), 123, dtype=np.uint8)).any()


def test_vertical_edge_known_values():
    # Left half black, right half white. Sobel x = (1 + 2 + 1) * (right - left),
    # so the two columns touching the edge get 4 * 255 = 1020 and the rest 0.
    # 1020 > 255 also shows the uint8 input does not overflow.
    img = np.zeros((5, 6), dtype=np.uint8)
    img[:, 3:] = 255
    expected = np.tile([0, 0, 1020, 1020, 0, 0], (5, 1))
    assert np.array_equal(compute_energy(img), expected)


def test_horizontal_edge_known_values():
    # The same picture turned on its side: now Sobel y does the work
    img = np.zeros((6, 5), dtype=np.uint8)
    img[3:, :] = 255
    expected = np.tile([[0], [0], [1020], [1020], [0], [0]], (1, 5))
    assert np.array_equal(compute_energy(img), expected)


def test_matches_sobel_written_by_hand():
    # Recompute |Gx| + |Gy| with plain NumPy slices (edge pixels repeated at the
    # border) and compare with the OpenCV-based implementation
    gray = random_image(9, 11, seed=1)
    p = np.pad(gray.astype(np.float64), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    np.testing.assert_allclose(compute_energy(gray), np.abs(gx) + np.abs(gy))


def test_colour_is_averaged_to_grayscale():
    # A colour image has the energy of its channel average ...
    img = random_image(7, 9, 3, seed=2)
    np.testing.assert_allclose(compute_energy(img), compute_energy(img.astype(np.float64).mean(axis=2)))
    # ... so the channel order (BGR or RGB) makes no difference
    np.testing.assert_allclose(compute_energy(img), compute_energy(img[:, :, ::-1]))


def test_alpha_channel_is_ignored():
    bgra = random_image(7, 9, 4, seed=3)
    assert np.array_equal(compute_energy(bgra), compute_energy(bgra[:, :, :3]))


def test_energy_of_transposed_image_is_transposed_energy():
    # This is what lets horizontal seams reuse the vertical-seam code
    gray = random_image(8, 11, seed=4)
    np.testing.assert_allclose(compute_energy(np.ascontiguousarray(gray.T)), compute_energy(gray).T)
