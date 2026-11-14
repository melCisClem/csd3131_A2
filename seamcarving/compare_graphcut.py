"""Graph cut vs DP comparison (assignment item 4): seams, visual quality, running time.

Usage: python -m seamcarving.compare_graphcut castle.png --scale 0.2 --width 52 --height 40
           [--sizes 0.1,0.15,0.2,0.25,0.3,0.4] [--out results/castle_graphcut]

The pure-Python max-flow takes minutes per seam on a full-size image, so the image
is first shrunk by --scale; --width/--height are targets for the shrunk image.
The pictures (not result_*.png) are blown back up with nearest-neighbour so each
pixel, and so each seam, stays visible.

Writes to the output folder:
    first_seam.png       first vertical seam of each method on the image and on its energy map
                         (red = DP, cyan = graph cut, white = both)
    all_seams_dp.png     shrunk image with every seam DP removed (red = vertical, blue = horizontal)
    all_seams_graphcut.png  the same for graph cut
    side_by_side.png     original | DP | graph cut | plain scaling
    result_dp.png, result_graphcut.png   the carved images
    seam_costs.csv       cost of every removed seam for both methods
    timing.csv, timing.png   time to find ONE seam at each image size in --sizes
    summary.md           all the numbers (also printed)
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from . import backward, graphcut
from .carve import seam_cost
from .compare import (
    RED, CYAN, draw_first_seams, draw_removed_seams, energy_as_image,
    hstack_padded, label, timed_carve, traced_carve,
)
from .energy import compute_energy

METHODS = {"dp": backward.find_vertical_seam, "graphcut": graphcut.find_vertical_seam}


def enlarge(img, factor):
    # Nearest-neighbour blow-up: every pixel becomes a factor x factor block
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def best_time(function, argument, repeats):
    # Fastest of `repeats` runs (DP is so fast that a single run is mostly noise)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        function(argument)
        best = min(best, time.perf_counter() - t0)
    return best


def timing_rows(img, scales):
    # Time to find one vertical seam at each size, plus the graph-cut statistics
    rows = []
    for scale in scales:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        energy = compute_energy(small)
        height, width = energy.shape
        print(f"timing one seam at {width}x{height} ...", flush=True)
        dp_time = best_time(backward.find_vertical_seam, energy, 5)
        t0 = time.perf_counter()
        seam, flow, paths, edges = graphcut.find_vertical_seam_with_stats(energy)
        graphcut_time = time.perf_counter() - t0
        dp_cost = seam_cost(energy, backward.find_vertical_seam(energy))
        rows.append({
            "width": width, "height": height, "pixels": width * height,
            "nodes": width * height + 2, "edges": edges, "augmenting_paths": paths,
            "dp_seconds": dp_time, "graphcut_seconds": graphcut_time,
            "dp_cost": dp_cost, "graphcut_cost": seam_cost(energy, seam), "max_flow": flow,
        })
    return rows


def timing_chart(rows, width=800, height=380, margin=60):
    # Pixels (x) vs seconds for one seam (y, log scale): DP red, graph cut cyan
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    times = [r[k] for r in rows for k in ("dp_seconds", "graphcut_seconds")]
    low, high = np.log10(max(min(times), 1e-6)) - 0.3, np.log10(max(times)) + 0.3
    most_pixels = max(r["pixels"] for r in rows) * 1.05

    def point(pixels, seconds):
        x = margin + pixels / most_pixels * (width - 2 * margin)
        y = height - margin - (np.log10(max(seconds, 1e-6)) - low) / (high - low) * (height - 2 * margin)
        return int(x), int(y)

    text = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (0, 0, 0), 1)
    for power in range(int(np.ceil(low)), int(np.floor(high)) + 1):   # one grid line per power of 10
        _, y = point(0, 10.0 ** power)
        cv2.line(canvas, (margin, y), (width - margin, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"1e{power}s", (5, y + 4), text, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for key, colour in (("dp_seconds", RED), ("graphcut_seconds", (200, 160, 0))):
        points = np.array([point(r["pixels"], r[key]) for r in rows], dtype=np.int32)
        cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, tuple(int(v) for v in p), 3, colour, -1, cv2.LINE_AA)

    cv2.putText(canvas, "time to find one seam (log scale)", (margin, 30), text, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DP", (width - margin - 130, 30), text, 0.6, RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, "graph cut", (width - margin - 90, 30), text, 0.6, (200, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"pixels (0 .. {max(r['pixels'] for r in rows)})", (width // 2 - 70, height - 15),
                text, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare graph-cut and DP seam carving")
    ap.add_argument("input")
    ap.add_argument("--scale", type=float, default=0.2, help="shrink the image first (default 0.2)")
    ap.add_argument("--width", type=int, help="target width of the shrunk image (default: unchanged)")
    ap.add_argument("--height", type=int, help="target height of the shrunk image (default: unchanged)")
    ap.add_argument("--sizes", default="0.1,0.15,0.2,0.25,0.3,0.4",
                    help="comma-separated scales for the one-seam timing table")
    ap.add_argument("--out", help="output folder (default: results/<image name>_graphcut)")
    args = ap.parse_args()

    original = cv2.imread(args.input)
    if original is None:
        print(f"could not read {args.input}", file=sys.stderr)
        return 1
    img = cv2.resize(original, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    w = W if args.width is None else args.width
    h = H if args.height is None else args.height
    if not (1 <= w <= W and 1 <= h <= H):
        print(f"error: target {w}x{h} must be between 1x1 and {W}x{H} (the shrunk image)", file=sys.stderr)
        return 1
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join("results", f"{name}_graphcut")
    os.makedirs(out_dir, exist_ok=True)
    zoom = max(1, round(1 / args.scale))

    # First seam of each method on the shrunk image
    energy = compute_energy(img)
    first = {method: finder(energy) for method, finder in METHODS.items()}
    shared_rows = int((first["dp"] == first["graphcut"]).sum())
    cv2.imwrite(os.path.join(out_dir, "first_seam.png"), hstack_padded([
        enlarge(draw_first_seams(img, first["dp"], first["graphcut"]), zoom),
        enlarge(draw_first_seams(energy_as_image(energy), first["dp"], first["graphcut"]), zoom),
    ]))

    # Running time of the whole carve (separate, untraced runs)
    results, timings = {}, {}
    for method, finder in METHODS.items():
        print(f"carving with {method} ...", flush=True)
        results[method], total, search = timed_carve(img, w, h, finder)
        timings[method] = (total, search)
        cv2.imwrite(os.path.join(out_dir, f"result_{method}.png"), results[method])

    # Seam costs and all removed seams
    traces = {}
    for method, finder in METHODS.items():
        print(f"tracing {method} seams ...", flush=True)
        records, removed_by = traced_carve(img, w, h, finder)
        traces[method] = records
        cv2.imwrite(os.path.join(out_dir, f"all_seams_{method}.png"), enlarge(draw_removed_seams(img, removed_by), zoom))

    scaled = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "side_by_side.png"), hstack_padded([
        label(enlarge(img, zoom), "original"), label(enlarge(results["dp"], zoom), "DP"),
        label(enlarge(results["graphcut"], zoom), "graph cut"), label(enlarge(scaled, zoom), "scaling"),
    ]))

    dp_costs = [cost for _, cost, _ in traces["dp"]]
    graphcut_costs = [cost for _, cost, _ in traces["graphcut"]]
    # graph-cut seam vs the DP seam on the SAME image: should always be exactly 1
    graphcut_gaps = [cost / optimal if optimal > 0 else 1.0 for _, cost, optimal in traces["graphcut"]]
    with open(os.path.join(out_dir, "seam_costs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seam", "kind", "dp_cost", "graphcut_cost", "dp_optimal_on_graphcut_image"])
        for k, ((kind, dp_cost, _), (_, gc_cost, optimal)) in enumerate(zip(traces["dp"], traces["graphcut"]), 1):
            writer.writerow([k, kind, f"{dp_cost:.2f}", f"{gc_cost:.2f}", f"{optimal:.2f}"])

    # One-seam timing at several sizes
    rows = timing_rows(original, [float(s) for s in args.sizes.split(",")])
    with open(os.path.join(out_dir, "timing.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(os.path.join(out_dir, "timing.png"), timing_chart(rows))

    # Summary
    identical = results["dp"].shape == results["graphcut"].shape and np.array_equal(results["dp"], results["graphcut"])
    seams = len(dp_costs)
    lines = [
        f"# Graph cut vs DP: {os.path.basename(args.input)} shrunk to {W}x{H} -> {w}x{h} ({seams} seams)",
        "",
        "| | DP | Graph cut |",
        "|---|---|---|",
        f"| Total carve time (s) | {timings['dp'][0]:.2f} | {timings['graphcut'][0]:.2f} |",
        f"| Seam search only (s) | {timings['dp'][1]:.3f} | {timings['graphcut'][1]:.2f} "
        f"({timings['graphcut'][1] / max(timings['dp'][1], 1e-9):.0f}x slower) |",
        f"| First seam energy | {seam_cost(energy, first['dp']):.1f} | {seam_cost(energy, first['graphcut']):.1f} |",
        f"| Total energy removed | {sum(dp_costs):.1f} | {sum(graphcut_costs):.1f} |",
        "",
        f"- First seams share {shared_rows} of {H} rows.",
        f"- Graph-cut seam / optimal DP seam on the same image: worst {max(graphcut_gaps, default=1):.6f}x "
        f"(optimal for {sum(abs(g - 1) < 1e-9 for g in graphcut_gaps)} of {seams} seams).",
        f"- Carved images are {'pixel-for-pixel identical' if identical else 'not identical (equal-cost seams broke ties differently)'}.",
        "",
        "## Time to find one seam",
        "",
        "| Image | Nodes | Edges | Augmenting paths (BFS) | DP (s) | Graph cut (s) | Slower by | Same cost | Max flow = seam cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['width']}x{r['height']} | {r['nodes']} | {r['edges']} | {r['augmenting_paths']} "
        f"| {r['dp_seconds']:.5f} | {r['graphcut_seconds']:.2f} "
        f"| {r['graphcut_seconds'] / max(r['dp_seconds'], 1e-9):,.0f}x "
        f"| {'yes' if abs(r['dp_cost'] - r['graphcut_cost']) < 1e-6 else 'NO'} "
        f"| {'yes' if abs(r['max_flow'] - r['graphcut_cost']) < 1e-6 else 'NO'} |"
        for r in rows
    ]

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)

    print(summary)
    print(f"images written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
