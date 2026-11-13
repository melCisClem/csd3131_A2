"""Tests for the step-by-step session behind the app (run with: pytest)."""
import numpy as np
import pytest

from seamcarving import carve
from seamcarving.interactive import HORIZONTAL, VERTICAL, CarveSession


def random_image(height=24, width=31, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def run_to(session, width, height):
    steps = []
    while (step := session.step_towards(width, height)) is not None:
        steps.append(step)
    return steps


@pytest.mark.parametrize("forward", [False, True])
def test_stepping_matches_carve(forward):
    img = random_image()
    session = CarveSession(img, forward=forward)
    run_to(session, 20, 15)
    assert np.array_equal(session.image, carve(img, 20, 15, forward=forward))
    assert session.seams_removed(VERTICAL) == 11
    assert session.seams_removed(HORIZONTAL) == 9


def test_removed_pixels_are_tracked_on_the_original():
    img = random_image()
    session = CarveSession(img)
    run_to(session, 20, 15)
    assert (session.removed_by != 0).sum() == img.shape[0] * img.shape[1] - 20 * 15


def test_growing_again_restores_seams_in_reverse():
    img = random_image()
    session = CarveSession(img)
    run_to(session, 20, 15)
    steps = run_to(session, 25, 15)   # wider again: must undo, then re-carve the height
    assert steps[0].action == "restore"
    assert np.array_equal(session.image, carve(img, 25, 15))
    run_to(session, 31, 24)           # all the way back
    assert np.array_equal(session.image, img)
    assert not session.history and not session.removed_by.any()


def test_down_to_one_pixel():
    img = random_image(6, 7)
    session = CarveSession(img)
    run_to(session, 1, 1)
    assert session.image.shape == (1, 1, 3)