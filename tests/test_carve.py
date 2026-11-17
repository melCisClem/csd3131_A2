"""Tests for seam removal and the resize loop, step (c) (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import (
    carve,
    is_valid_seam,
    remove_horizontal_seam,
    remove_vertical_seam,
    seam_cost,
)
from seamcarving import find_vertical_seam_graphcut


def random_image(height=24, width=31, channels=3, seed=0):
    shape = (height, width) if channels is None else (height, width, channels)
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def random_seam(height, width, seed=0):
    # A random valid seam: random start, then a step of -1, 0 or +1 per row
    rng = np.random.default_rng(seed)
    seam = [int(rng.integers(width))]
    for _ in range(height - 1):
        seam.append(int(np.clip(seam[-1] + rng.integers(-1, 2), 0, width - 1)))
    return np.array(seam)


# ---- removing one seam ----

def test_remove_vertical_seam_worked_example():
    img = np.arange(12).reshape(3, 4)        # 0..3 / 4..7 / 8..11
    out = remove_vertical_seam(img, np.array([0, 1, 2]))
    assert np.array_equal(out, [
        [1, 2, 3],       # 0 removed
        [4, 6, 7],       # 5 removed
        [8, 9, 11],      # 10 removed
    ])


def test_remove_horizontal_seam_worked_example():
    img = np.arange(12).reshape(3, 4)
    out = remove_horizontal_seam(img, np.array([0, 1, 1, 2]))   # row removed in each column
    assert np.array_equal(out, [
        [4, 1, 2, 3],    # column 0 lost 0, column 3 lost 11
        [8, 9, 10, 7],   # columns 1 and 2 lost 5 and 6
    ])


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_image_gets_exactly_one_pixel_narrower(channels):
    img = random_image(9, 12, channels)
    before = img.copy()
    out = remove_vertical_seam(img, random_seam(9, 12))
    assert out.shape == (9, 11) + img.shape[2:]
    assert out.dtype == img.dtype
    assert np.array_equal(img, before)       # the input image is left alone


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_image_gets_exactly_one_pixel_shorter(channels):
    img = random_image(9, 12, channels)
    before = img.copy()
    out = remove_horizontal_seam(img, random_seam(12, 9))
    assert out.shape == (8, 12) + img.shape[2:]
    assert out.dtype == img.dtype
    assert np.array_equal(img, before)


@pytest.mark.parametrize("seed", range(10))
def test_only_the_seam_pixels_are_removed(seed):
    # Row by row: the result is the old row with just the seam pixel deleted
    img = random_image(9, 12, seed=seed)
    seam = random_seam(9, 12, seed=seed)
    out = remove_vertical_seam(img, seam)
    for row in range(9):
        assert np.array_equal(out[row], np.delete(img[row], seam[row], axis=0))

    seam = random_seam(12, 9, seed=seed)
    out = remove_horizontal_seam(img, seam)
    for col in range(12):
        assert np.array_equal(out[:, col], np.delete(img[:, col], seam[col], axis=0))


# ---- seam helpers ----

def test_seam_cost_adds_the_energy_on_the_seam():
    energy = np.arange(12, dtype=float).reshape(3, 4)
    assert seam_cost(energy, np.array([0, 1, 2])) == 0 + 5 + 10


def test_is_valid_seam():
    assert is_valid_seam([1, 1, 2], 3, 4)
    assert is_valid_seam([3, 2, 3], 3, 4)
    assert is_valid_seam([2], 1, 4)
    assert not is_valid_seam([0, 2, 2], 3, 4)      # jumps two columns
    assert not is_valid_seam([0, 1], 3, 4)         # one pixel per row: needs 3
    assert not is_valid_seam([-1, 0, 0], 3, 4)     # off the left edge
    assert not is_valid_seam([3, 4, 4], 3, 4)      # off the right edge


# ---- the whole resize loop ----

@pytest.mark.parametrize("size", [(31, 24), (30, 24), (31, 23), (20, 15), (1, 24), (31, 1), (1, 1)])
def test_carve_reaches_the_requested_size(size):
    width, height = size
    img = random_image()
    before = img.copy()
    out = carve(img, width, height)
    assert out.shape == (height, width, 3)
    assert out.dtype == img.dtype
    assert np.array_equal(img, before)


def test_carve_to_the_same_size_changes_nothing():
    img = random_image()
    assert np.array_equal(carve(img, 31, 24), img)


def test_carve_works_on_grayscale():
    assert carve(random_image(12, 15, None), 10, 9).shape == (9, 10)


@pytest.mark.parametrize("size", [(32, 24), (31, 25), (0, 10), (10, 0), (-1, 5)])
def test_carve_rejects_sizes_it_cannot_reach(size):
    # It only shrinks, and at least one pixel has to remain
    with pytest.raises(ValueError):
        carve(random_image(), *size)


def test_carve_removes_what_the_seam_finder_says():
    # A fake finder that always picks the first column / row
    def first_column(energy):
        return np.zeros(energy.shape[0], dtype=np.int64)

    img = random_image()
    assert np.array_equal(carve(img, 28, 24, seam_finder=first_column), img[:, 3:])
    assert np.array_equal(carve(img, 31, 22, seam_finder=first_column), img[2:, :])


def test_on_seam_reports_every_step():
    # Replaying the reported seams by hand must rebuild every in-between image
    # and the final result: vertical seams first, then horizontal
    img = random_image(12, 15)
    steps = []
    out = carve(img, 12, 10, on_seam=lambda kind, seam, before: steps.append((kind, seam, before)))

    assert [kind for kind, _, _ in steps] == ["vertical"] * 3 + ["horizontal"] * 2
    assert np.array_equal(steps[0][2], img)
    for k, (kind, seam, before) in enumerate(steps):
        remove = remove_vertical_seam if kind == "vertical" else remove_horizontal_seam
        after = steps[k + 1][2] if k + 1 < len(steps) else out
        assert np.array_equal(remove(before, seam), after)


@pytest.mark.parametrize("seed", range(5))
def test_carve_only_deletes_pixels(seed):
    # Narrowing never changes or reorders pixels: every output row is the
    # original row with some pixels taken out
    img = random_image(10, 14, seed=seed)
    out = carve(img, 9, 10)
    for row in range(10):
        remaining = iter(map(tuple, img[row].tolist()))
        assert all(pixel in remaining for pixel in map(tuple, out[row].tolist()))


@pytest.mark.parametrize("method", ["dp", "forward", "graphcut"])
def test_flat_background_is_removed_before_the_object(method):
    # A 6-pixel-wide striped object on a flat background, 9 flat columns on each
    # side. Flat columns cost 0 and every object pixel costs more than 0, so an
    # optimal method has to take all 10 seams from the background.
    stripes = [150, 160, 170, 180, 190, 200]
    img = np.full((12, 24, 3), 100, dtype=np.uint8)
    img[3:9, 9:15] = np.array(stripes, dtype=np.uint8)[None, :, None]

    options = {
        "dp": {},
        "forward": {"forward": True},
        "graphcut": {"seam_finder": find_vertical_seam_graphcut},
    }[method]
    out = carve(img, 14, 12, **options)

    assert out.shape == (12, 14, 3)
    for row in range(3, 9):
        object_pixels = out[row, :, 0][out[row, :, 0] > 100]
        assert object_pixels.tolist() == stripes
