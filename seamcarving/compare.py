"""Greedy vs DP comparison (assignment item 3): seams, visual quality, running time.

Usage: python -m seamcarving.compare castle.png --width 250 --height 180 [--out results/castle_greedy]
   or: python -m seamcarving castle.png out.png --width 250 --height 180 --greedy
       (carves, saves out.png and writes this whole report to the same folder)

Writes to the output folder (default results/<image name>_greedy):
    first_seam.png       first vertical seam of each method on the image and on its energy map
                         (red = DP, cyan = greedy, white = both)
    all_seams_dp.png     original image with every seam DP removed (red = vertical, blue = horizontal)
    all_seams_greedy.png the same for greedy
    side_by_side.png     original | DP | greedy | plain scaling, all at the same height
    result_dp.png, result_greedy.png   the carved images
    seam_costs.png       energy of every removed seam, DP (red) vs greedy (cyan)
    seam_costs.csv       the same numbers, one row per seam
    timing.csv, timing.png   time to find ONE seam at several image sizes (see TIMING_SCALES)
    summary.md           timing and seam-cost table (also printed)

Seam cost = sum of the (backward) energy along the seam, i.e. what the DP
minimises. "Optimality gap" compares each greedy seam with the DP seam on the
SAME image, so it measures only the greedy decision, not the drift between runs.
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from . import backward, greedy
from .carve import carve, seam_cost
from .energy import compute_energy

METHODS = {"dp": backward.find_vertical_seam, "greedy": greedy.find_vertical_seam} 
RED, CYAN, BLUE, WHITE = (0, 0, 255), (255, 255, 0), (255, 0, 0), (255, 255, 255)   # BGR


# Resize the image and returns 2 timings - 1 for how long the resizing took and 
# 1 for how long it took to find the seams 
def timed_carve(img, width, height, seam_finder):
    search_seconds = 0.0 # Running total 

    # How much time it took to find 1 seam and add to the total times 
    def timed_finder(energy):
        nonlocal search_seconds #UnboundLocalError if without nonlocal 
        t0 = time.perf_counter()
        seam = seam_finder(energy)
        search_seconds += time.perf_counter() - t0 
        return seam

    t0 = time.perf_counter()
    out = carve(img, width, height, seam_finder=timed_finder)
    return out, time.perf_counter() - t0, search_seconds

# Indicates how good each seam was and where the seam was in the original image 
def traced_carve(img, width, height, seam_finder):

    original_height, original_width = img.shape[:2]
    index_map = np.arange(original_height * original_width).reshape(original_height, original_width)
    removed_by = np.zeros(original_height * original_width, dtype=np.uint8)   # 0 kept, 1 vertical, 2 horizontal
    records = []   # (kind, cost, optimal_cost)

    def on_seam(kind, seam, img_before):
        # For scoring of the seam 
        nonlocal index_map
        oriented = img_before if kind == "vertical" else np.swapaxes(img_before, 0, 1) # Transpose the image if it is horizontal seam as the seam code only works for vertical 
        energy = compute_energy(oriented)
        optimal = seam_cost(energy, backward.find_vertical_seam(energy)) # Calculates the cost of the seam 
        records.append((kind, seam_cost(energy, seam), optimal))

        # Track where the seam was in the original image 
        index_view = index_map if kind == "vertical" else index_map.T
        rows = np.arange(len(seam))
        removed_by[index_view[rows, seam]] = 1 if kind == "vertical" else 2 # Original position of the seam about to be removed and indicate whether it is removed by horizontal or vertical
        keep = np.ones(index_view.shape, dtype=bool)
        keep[rows, seam] = False
        index_view = index_view[keep].reshape(index_view.shape[0], index_view.shape[1] - 1)
        index_map = index_view if kind == "vertical" else index_view.T

    carve(img, width, height, seam_finder=seam_finder, on_seam=on_seam)
    return records, removed_by.reshape(original_height, original_width)


def draw_removed_seams(img, removed_by):
    out = img[:, :, :3].copy()
    out[removed_by == 1] = RED
    out[removed_by == 2] = BLUE
    return out


def draw_first_seams(img, dp_seam, greedy_seam):
    out = img[:, :, :3].copy()
    rows = np.arange(len(dp_seam))
    out[rows, greedy_seam] = CYAN
    out[rows, dp_seam] = RED
    both = dp_seam == greedy_seam
    out[rows[both], dp_seam[both]] = WHITE
    return out


def energy_as_image(energy):
    scaled = np.clip(energy / np.percentile(energy, 99) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return out


def hstack_padded(images, gap=8):
    # Place images side by side, top-aligned, on a white background
    height = max(i.shape[0] for i in images)
    width = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for i in images:
        canvas[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return canvas


def cost_chart(dp_costs, greedy_costs, width=800, height=360, margin=50):
    # Minimal line chart: seam number vs seam energy
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    top = max(max(dp_costs), max(greedy_costs)) * 1.05
    count = len(dp_costs)

    def point(k, cost):
        x = margin + k * (width - 2 * margin) / max(count - 1, 1)
        y = height - margin - cost / top * (height - 2 * margin)
        return int(x), int(y)

    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (0, 0, 0), 1)

    for costs, colour in ((greedy_costs, (200, 160, 0)), (dp_costs, RED)):
        points = np.array([point(k, c) for k, c in enumerate(costs)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, colour, 1, cv2.LINE_AA)

    text = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, "energy of each removed seam", (margin, 30), text, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DP", (width - margin - 90, 30), text, 0.6, RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, "greedy", (width - margin - 55, 30), text, 0.6, (200, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"seam 1 .. {count}", (width // 2 - 40, height - 15), text, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{top:.0f}", (5, margin + 5), text, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (30, height - margin + 5), text, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


# Image sizes for the one-seam timing table, as multiples of the input size.
# Above 1 the image is enlarged so the growth of the DP time is easy to see.
TIMING_SCALES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def best_time(function, argument, repeats=5):
    # Fastest of `repeats` runs (a single run of something this fast is mostly noise)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        function(argument)
        best = min(best, time.perf_counter() - t0)
    return best


def timing_rows(img, scales=TIMING_SCALES):
    # Time to find ONE vertical seam with each method at each size
    rows = []
    for scale in scales:
        if scale == 1:
            sized = img
        else:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            sized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
        energy = compute_energy(sized)
        height, width = energy.shape
        print(f"timing one seam at {width}x{height} ...", flush=True)
        rows.append({
            "width": width, "height": height, "pixels": width * height,
            "dp_seconds": best_time(backward.find_vertical_seam, energy),
            "greedy_seconds": best_time(greedy.find_vertical_seam, energy),
            "dp_cost": seam_cost(energy, backward.find_vertical_seam(energy)),
            "greedy_cost": seam_cost(energy, greedy.find_vertical_seam(energy)),
        })
    return rows


def timing_chart(rows, width=800, height=380, margin=60):
    # Pixels (x) vs seconds for one seam (y, log scale): DP red, greedy cyan
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    times = [r[k] for r in rows for k in ("dp_seconds", "greedy_seconds")]
    low, high = np.log10(max(min(times), 1e-7)) - 0.3, np.log10(max(times)) + 0.3
    most_pixels = max(r["pixels"] for r in rows) * 1.05

    def point(pixels, seconds):
        x = margin + pixels / most_pixels * (width - 2 * margin)
        y = height - margin - (np.log10(max(seconds, 1e-7)) - low) / (high - low) * (height - 2 * margin)
        return int(x), int(y)

    text = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (0, 0, 0), 1)
    for power in range(int(np.ceil(low)), int(np.floor(high)) + 1):   # one grid line per power of 10
        _, y = point(0, 10.0 ** power)
        cv2.line(canvas, (margin, y), (width - margin, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"1e{power}s", (5, y + 4), text, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for key, colour in (("dp_seconds", RED), ("greedy_seconds", (200, 160, 0))):
        points = np.array([point(r["pixels"], r[key]) for r in rows], dtype=np.int32)
        cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, tuple(int(v) for v in p), 3, colour, -1, cv2.LINE_AA)

    cv2.putText(canvas, "time to find one seam (log scale)", (margin, 30), text, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DP", (width - margin - 110, 30), text, 0.6, RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, "greedy", (width - margin - 70, 30), text, 0.6, (200, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"pixels (0 .. {max(r['pixels'] for r in rows)})", (width // 2 - 70, height - 15),
                text, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def default_out_dir(input_path):
    # results/<image name>_greedy, e.g. results/castle_greedy
    return os.path.join("results", os.path.splitext(os.path.basename(input_path))[0] + "_greedy")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare greedy and DP seam carving")
    ap.add_argument("input")
    ap.add_argument("--width", type=int, help="target width (default: unchanged)")
    ap.add_argument("--height", type=int, help="target height (default: unchanged)")
    ap.add_argument("--out", help="output folder (default: results/<image name>_greedy)")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"could not read {args.input}", file=sys.stderr)
        return 1
    H, W = img.shape[:2]
    w = W if args.width is None else args.width
    h = H if args.height is None else args.height
    if not (1 <= w <= W and 1 <= h <= H):
        print(f"error: target {w}x{h} must be between 1x1 and {W}x{H}", file=sys.stderr)
        return 1
    write_report(args.input, img, w, h, args.out or default_out_dir(args.input))
    return 0


def write_report(input_path, img, w, h, out_dir):
    # Everything listed at the top of this file, written to out_dir
    H, W = img.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    # First seam of each method on the original image
    energy = compute_energy(img)
    first = {name: finder(energy) for name, finder in METHODS.items()}
    shared_rows = int((first["dp"] == first["greedy"]).sum())
    cv2.imwrite(os.path.join(out_dir, "first_seam.png"), hstack_padded([
        draw_first_seams(img, first["dp"], first["greedy"]),
        draw_first_seams(energy_as_image(energy), first["dp"], first["greedy"]),
    ]))

    # Running time (separate, untraced runs so the bookkeeping is not timed)
    results, timings = {}, {}
    for name, finder in METHODS.items():
        print(f"carving with {name} ...", flush=True)
        results[name], total, search = timed_carve(img, w, h, finder)
        timings[name] = (total, search)
        cv2.imwrite(os.path.join(out_dir, f"result_{name}.png"), results[name])

    # Seam costs and all removed seams
    traces = {}
    for name, finder in METHODS.items():
        print(f"tracing {name} seams ...", flush=True)
        records, removed_by = traced_carve(img, w, h, finder)
        traces[name] = records
        cv2.imwrite(os.path.join(out_dir, f"all_seams_{name}.png"), draw_removed_seams(img, removed_by))

    scaled = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "side_by_side.png"), hstack_padded([
        label(img, "original"), label(results["dp"], "DP"),
        label(results["greedy"], "greedy"), label(scaled, "scaling"),
    ]))

    dp_costs = [cost for _, cost, _ in traces["dp"]]
    greedy_costs = [cost for _, cost, _ in traces["greedy"]]
    greedy_gaps = [cost / optimal if optimal > 0 else 1.0 for _, cost, optimal in traces["greedy"]]
    with open(os.path.join(out_dir, "seam_costs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seam", "kind", "dp_cost", "greedy_cost", "dp_optimal_on_greedy_image"])
        for k, ((kind, dp_cost, _), (_, greedy_cost, optimal)) in enumerate(zip(traces["dp"], traces["greedy"]), 1):
            writer.writerow([k, kind, f"{dp_cost:.1f}", f"{greedy_cost:.1f}", f"{optimal:.1f}"])
    if dp_costs:
        cv2.imwrite(os.path.join(out_dir, "seam_costs.png"), cost_chart(dp_costs, greedy_costs))

    # One-seam timing at several sizes
    rows = timing_rows(img)
    with open(os.path.join(out_dir, "timing.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(os.path.join(out_dir, "timing.png"), timing_chart(rows))

    # Summary
    dp_first, greedy_first = (seam_cost(energy, first[n]) for n in ("dp", "greedy"))
    lines = [
        f"# Greedy vs DP: {os.path.basename(input_path)} {W}x{H} -> {w}x{h}",
        "",
        "| | DP | Greedy |",
        "|---|---|---|",
        f"| Total carve time (s) | {timings['dp'][0]:.2f} | {timings['greedy'][0]:.2f} |",
        f"| Seam search only (s) | {timings['dp'][1]:.2f} | {timings['greedy'][1]:.2f} |",
        f"| First seam energy | {dp_first:.0f} | {greedy_first:.0f} ({greedy_first / dp_first:.2f}x) |",
        f"| Total energy removed | {sum(dp_costs):.0f} | {sum(greedy_costs):.0f} "
        f"({sum(greedy_costs) / max(sum(dp_costs), 1e-9):.2f}x) |",
        f"| Mean energy per seam | {np.mean(dp_costs):.0f} | {np.mean(greedy_costs):.0f} |",
        "",
        f"- First seams share {shared_rows} of {H} rows.",
        f"- Greedy seam / optimal DP seam on the same image: median {np.median(greedy_gaps):.2f}x, "
        f"worst {max(greedy_gaps):.2f}x; greedy was optimal for "
        f"{sum(g <= 1 + 1e-9 for g in greedy_gaps)} of {len(greedy_gaps)} seams.",
        f"- Seam search speed-up: {timings['dp'][1] / max(timings['greedy'][1], 1e-9):.1f}x; "
        f"whole carve: {timings['dp'][0] / max(timings['greedy'][0], 1e-9):.1f}x "
        "(energy recomputation and seam removal are the same for both).",
    ] if dp_costs else ["no seams to remove"]

    lines += [
        "",
        "## Time to find one seam",
        "",
        "| Image | Pixels | DP (s) | Greedy (s) | DP / greedy | DP seam cost | Greedy seam cost |",
        "|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['width']}x{r['height']} | {r['pixels']} | {r['dp_seconds']:.5f} | {r['greedy_seconds']:.5f} "
        f"| {r['dp_seconds'] / max(r['greedy_seconds'], 1e-9):.1f}x "
        f"| {r['dp_cost']:.0f} | {r['greedy_cost']:.0f} |"
        for r in rows
    ]

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)

    print(summary)
    print(f"images written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
