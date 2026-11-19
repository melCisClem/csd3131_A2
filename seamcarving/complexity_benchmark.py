"""Item 5(b): measure the time AND peak memory of one seam search for each method,
and the time of a full carve as the number of removed seams k grows.

    python -m seamcarving.complexity_benchmark castle.png

Writes to results/complexity/:
    one_seam.csv   width, height, pixels, method, seconds, peak_bytes (+ graph size for graph cut)
    k_seams.csv    k, seconds for carve() with DP and with greedy on the same image

Peak memory is measured with tracemalloc, which also tracks NumPy arrays. It is
the extra memory the seam finder allocates on top of the energy map it is given,
so it is exactly the "space" in the complexity analysis.
"""
import argparse
import csv
import os
import time
import tracemalloc

import cv2

from . import backward, graphcut, greedy
from .carve import carve
from .energy import compute_energy

DP_SCALES = (0.25, 0.5, 1.0, 2.0, 3.0)                 # castle 341x229 -> up to 1023x687
GRAPHCUT_SCALES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)     # graph cut is too slow for bigger images
K_VALUES = (10, 20, 40, 80)


def resized(img, scale):
    if scale == 1:
        return img
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)


def best_time(function, argument, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        function(argument)
        best = min(best, time.perf_counter() - t0)
    return best


def peak_memory(function, argument):
    # Bytes allocated at the busiest moment of one call (the input is not counted)
    tracemalloc.start()
    tracemalloc.reset_peak()
    function(argument)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def one_seam_rows(img):
    rows = []
    methods = [("dp", backward.find_vertical_seam, DP_SCALES, 5),
               ("greedy", greedy.find_vertical_seam, DP_SCALES, 5),
               ("dp", backward.find_vertical_seam, GRAPHCUT_SCALES, 5),
               ("graphcut", graphcut.find_vertical_seam, GRAPHCUT_SCALES, 1)]
    for name, finder, scales, repeats in methods:
        for scale in scales:
            energy = compute_energy(resized(img, scale))
            height, width = energy.shape
            print(f"{name:8s} {width}x{height} ...", flush=True)
            row = {"width": width, "height": height, "pixels": width * height, "method": name,
                   "seconds": best_time(finder, energy, repeats),
                   "peak_bytes": peak_memory(finder, energy)}
            if name == "graphcut":
                _, _, paths, edges = graphcut.find_vertical_seam_with_stats(energy)
                row.update({"nodes": width * height + 2, "edges": edges, "augmenting_paths": paths})
            rows.append(row)
    return rows


def k_seam_rows(img):
    rows = []
    height, width = img.shape[:2]
    for k in K_VALUES:
        for name, finder in (("dp", backward.find_vertical_seam), ("greedy", greedy.find_vertical_seam)):
            t0 = time.perf_counter()
            carve(img, width - k, height, seam_finder=finder)
            rows.append({"k": k, "method": name, "width": width, "height": height,
                         "seconds": time.perf_counter() - t0})
            print(f"carve k={k} {name}: {rows[-1]['seconds']:.3f} s", flush=True)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", default="castle.png")
    parser.add_argument("--out", default=os.path.join("results", "complexity"))
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "one_seam.csv"), one_seam_rows(img))
    write_csv(os.path.join(args.out, "k_seams.csv"), k_seam_rows(img))
    print(f"wrote {args.out}/one_seam.csv and k_seams.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
