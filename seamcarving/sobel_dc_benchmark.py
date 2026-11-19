"""Q2b: does divide and conquer speed up the Sobel energy map?

Times four ways of computing the same energy map as compute_energy (energy.py):
  1. direct      - cv2.Sobel on the whole image (what energy.py does)
  2. dc_serial   - recursive divide and conquer: split the larger side in half until
                   a tile has at most 64*64 pixels, filter each tile with a 1-pixel halo
  3. dc_parallel - split into one horizontal strip per thread (with halos), run in parallel
  4. fft         - convolution through the FFT (scipy.signal.fftconvolve)
Every method is checked against compute_energy before it is timed.

    python -m seamcarving.sobel_dc_benchmark castle.png surfer.png
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.signal import fftconvolve

from .backward import find_vertical_seam
from .energy import compute_energy

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def sobel_tile(padded: np.ndarray) -> np.ndarray:
    # Energy of a tile that already carries a 1-pixel halo; the halo is cut off again.
    gradient_x = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)
    return (np.abs(gradient_x) + np.abs(gradient_y))[1:-1, 1:-1]


def direct(gray):
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    return sobel_tile(padded)


def dc_serial(gray, leaf_pixels=64 * 64):
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    out = np.empty_like(gray)

    def solve(r0, r1, c0, c1):
        if (r1 - r0) * (c1 - c0) <= leaf_pixels:              # base case: filter the tile
            out[r0:r1, c0:c1] = sobel_tile(padded[r0:r1 + 2, c0:c1 + 2])
        elif r1 - r0 >= c1 - c0:                              # divide the taller side
            mid = (r0 + r1) // 2
            solve(r0, mid, c0, c1)
            solve(mid, r1, c0, c1)
        else:                                                 # divide the wider side
            mid = (c0 + c1) // 2
            solve(r0, r1, c0, mid)
            solve(r0, r1, mid, c1)

    solve(0, gray.shape[0], 0, gray.shape[1])
    return out


def dc_parallel(gray, threads=4):
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    out = np.empty_like(gray)
    bounds = np.linspace(0, gray.shape[0], threads + 1).astype(int)

    def strip(k):
        r0, r1 = bounds[k], bounds[k + 1]
        out[r0:r1] = sobel_tile(padded[r0:r1 + 2])

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(strip, range(threads)))
    return out


def fft(gray):
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # fftconvolve flips the kernel, so flip it first to get correlation like cv2.Sobel
    gradient_x = fftconvolve(padded, SOBEL_X[::-1, ::-1], mode="valid")
    gradient_y = fftconvolve(padded, SOBEL_X.T[::-1, ::-1], mode="valid")
    return np.abs(gradient_x) + np.abs(gradient_y)


def median_ms(fn, arg, repeats=15):
    fn(arg)                                                   # warm-up run
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


if __name__ == "__main__":
    cv2.setNumThreads(1)                                      # fair comparison: OpenCV uses one core
    methods = [direct, dc_serial, dc_parallel, fft]
    print("image, size, " + ", ".join(m.__name__ for m in methods) + ", dp_seam (ms)")
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        gray = img.astype(np.float64).mean(axis=2)            # same grayscale as energy.py
        reference = compute_energy(img)
        for method in methods:
            assert np.allclose(method(gray), reference), method.__name__
        row = [median_ms(m, gray) for m in methods] + [median_ms(find_vertical_seam, reference, 5)]
        print(f"{path}, {gray.shape[0]}x{gray.shape[1]}, " + ", ".join(f"{t:.2f}" for t in row))
