"""Tests for the timing script (run with: pytest).

Timings differ on every machine, so nothing here checks how fast a method is.
The tests only check that the script measures the right things and still runs.
"""
import sys

import cv2
import numpy as np
import pytest

from seamcarving import benchmark


@pytest.fixture
def small_image():
    return np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)


def test_one_seam_rows_time_every_method_on_the_same_image(small_image):
    rows = benchmark.one_seam_rows(small_image, [0.5, 1.0], graphcut_max_pixels=200)
    assert [(r["width"], r["height"], r["pixels"]) for r in rows] == [(15, 10, 150), (30, 20, 600)]
    for row in rows:
        assert row["dp_seconds"] > 0 and row["greedy_seconds"] > 0
        assert row["greedy_cost"] >= row["dp_cost"] - 1e-9
    # graph cut ran on the 150-pixel image (and found the DP cost), not on the 600-pixel one
    assert rows[0]["graphcut_seconds"] > 0
    assert rows[0]["graphcut_cost"] == pytest.approx(rows[0]["dp_cost"])
    assert rows[1]["graphcut_seconds"] is None and rows[1]["graphcut_cost"] is None


def test_growth_rate_is_the_slope_on_log_axes():
    # Made-up timings that grow exactly like pixels^2 and pixels^0.5
    rows = [{"pixels": p, "dp_seconds": p ** 2.0, "greedy_seconds": p ** 0.5, "graphcut_seconds": None}
            for p in (100, 1000, 10000)]
    assert benchmark.growth_rate(rows, "dp") == pytest.approx(2.0)
    assert benchmark.growth_rate(rows, "greedy") == pytest.approx(0.5)
    assert np.isnan(benchmark.growth_rate(rows, "graphcut"))      # nothing was measured


def test_carve_breakdown_times_the_three_steps(small_image):
    rows = benchmark.carve_breakdown(small_image, seam_count=3)
    assert [r["method"] for r in rows] == ["dp", "greedy"]
    for row in rows:
        assert all(row[step] > 0 for step in ("energy", "search", "removal"))


def test_script_writes_charts_and_tables(monkeypatch, tmp_path, small_image):
    pytest.importorskip("matplotlib")
    image_path = tmp_path / "in.png"
    assert cv2.imwrite(str(image_path), small_image)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "benchmark", str(image_path), "--scales", "0.5,1.0", "--graphcut-max-pixels", "200",
        "--seams", "2", "--out", str(out_dir),
    ])
    assert benchmark.main() == 0
    for name in ("timing_one_seam.png", "timing_one_seam.csv",
                 "carve_breakdown.png", "carve_breakdown.csv", "summary.md"):
        assert (out_dir / name).stat().st_size > 0
