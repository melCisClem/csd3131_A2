"""Tests for the command line, python -m seamcarving (run with: pytest)."""
import sys

import cv2
import numpy as np
import pytest

from seamcarving import carve
from seamcarving.__main__ import main


@pytest.fixture
def image_file(tmp_path):
    # A small 16x12 PNG on disk (PNG is lossless, so pixels survive the round trip)
    img = np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    path = tmp_path / "in.png"
    assert cv2.imwrite(str(path), img)
    return path, img


def run(monkeypatch, *arguments):
    # Call main() as if the arguments had been typed on the command line
    monkeypatch.setattr(sys, "argv", ["seamcarving", *map(str, arguments)])
    return main()


@pytest.mark.parametrize("flag", [None, "--forward", "--greedy", "--graphcut"])
def test_writes_an_image_of_the_requested_size(monkeypatch, capsys, tmp_path, image_file, flag):
    path, _ = image_file
    out_path = tmp_path / "out.png"
    extra = [flag] if flag else []
    assert run(monkeypatch, path, out_path, "--width", 13, "--height", 10, *extra) == 0
    assert cv2.imread(str(out_path)).shape == (10, 13, 3)
    assert "16x12 -> 13x10" in capsys.readouterr().out


def test_output_is_the_same_as_calling_carve(monkeypatch, tmp_path, image_file):
    path, img = image_file
    out_path = tmp_path / "out.png"
    assert run(monkeypatch, path, out_path, "--width", 13, "--height", 10) == 0
    assert np.array_equal(cv2.imread(str(out_path)), carve(img, 13, 10))


def test_a_size_that_is_left_out_stays_unchanged(monkeypatch, tmp_path, image_file):
    path, _ = image_file
    out_path = tmp_path / "out.png"
    assert run(monkeypatch, path, out_path, "--width", 14) == 0
    assert cv2.imread(str(out_path)).shape == (12, 14, 3)


@pytest.mark.parametrize("arguments", [
    ["--width", 17],                             # wider than the image: it only shrinks
    ["--height", 13],
    ["--width", 0],                              # nothing left
    ["--width", 13, "--forward", "--greedy"],    # two methods at once
    ["--width", 13, "--greedy", "--graphcut"],
])
def test_bad_arguments_fail_without_writing(monkeypatch, capsys, tmp_path, image_file, arguments):
    path, _ = image_file
    out_path = tmp_path / "out.png"
    assert run(monkeypatch, path, out_path, *arguments) == 1
    assert "error" in capsys.readouterr().err
    assert not out_path.exists()


def test_missing_input_file_fails(monkeypatch, tmp_path):
    out_path = tmp_path / "out.png"
    assert run(monkeypatch, tmp_path / "no_such_file.png", out_path, "--width", 5) == 1
    assert not out_path.exists()
