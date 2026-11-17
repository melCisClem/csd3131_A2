"""Running time of all three seam finders: DP, greedy and graph cut.

Usage: python -m seamcarving.benchmark castle.png [--out results/benchmark]
           [--scales 0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0] [--graphcut-max-pixels 13000]
           [--seams 20]

Two measurements, both on the same image:

1. One seam at several image sizes. The image is shrunk to each scale and every
   method finds one vertical seam on the same energy map. Graph cut is skipped
   above --graphcut-max-pixels (pure-Python max-flow needs minutes per seam at
   full size). The slope of each line on the log-log chart is the measured
   growth rate: time ~ pixels^slope.
2. Where the time goes in a whole carve. --seams vertical seams are removed from
   the full-size image with DP and with greedy, timing the three steps of the
   loop separately: energy map, seam search, seam removal.

Writes to the output folder:
    timing_one_seam.png     chart for measurement 1 (log-log)
    timing_one_seam.csv     the same numbers
    carve_breakdown.png     chart for measurement 2
    carve_breakdown.csv     the same numbers
    summary.md              both tables and the growth rates (also printed)
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from . import backward, graphcut, greedy
from .carve import remove_vertical_seam, seam_cost
from .energy import compute_energy

# name -> (seam finder, label, colour, timing repeats). The fastest of the
# repeats is kept: DP and greedy are so fast that one run is mostly noise.
METHODS = {
    "dp": (backward.find_vertical_seam, "DP", "#2a78d6", 7),
    "greedy": (greedy.find_vertical_seam, "Greedy", "#eb6834", 25),
    "graphcut": (graphcut.find_vertical_seam, "Graph cut", "#1baf7a", 1),
}
INK, MUTED, GRID, SHARED_DARK, SHARED_LIGHT = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984", "#c9c8c2"


def best_time(function, argument, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        function(argument)
        best = min(best, time.perf_counter() - t0)
    return best


def one_seam_rows(img, scales, graphcut_max_pixels):
    # Time to find ONE vertical seam with each method at each image size.
    # A method that was skipped gets None.
    rows = []
    for scale in scales:
        small = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        energy = compute_energy(small)
        height, width = energy.shape
        print(f"timing one seam at {width}x{height} ...", flush=True)
        row = {"width": width, "height": height, "pixels": width * height}
        for name, (finder, _, _, repeats) in METHODS.items():
            skipped = name == "graphcut" and width * height > graphcut_max_pixels
            row[f"{name}_seconds"] = None if skipped else best_time(finder, energy, repeats)
            row[f"{name}_cost"] = None if skipped else seam_cost(energy, finder(energy)) if name != "graphcut" else None
        if row["graphcut_seconds"] is not None:
            row["graphcut_cost"] = seam_cost(energy, graphcut.find_vertical_seam(energy))
        rows.append(row)
    return rows


def growth_rate(rows, name):
    # Slope of log(time) against log(pixels): time ~ pixels^slope
    points = [(r["pixels"], r[f"{name}_seconds"]) for r in rows if r[f"{name}_seconds"]]
    if len(points) < 2:
        return float("nan")
    pixels, seconds = zip(*points)
    return float(np.polyfit(np.log(pixels), np.log(seconds), 1)[0])


def carve_breakdown(img, seam_count, names=("dp", "greedy")):
    # Remove seam_count vertical seams; return seconds PER SEAM spent in each step
    rows = []
    for name in names:
        finder = METHODS[name][0]
        current = img
        spent = {"energy": 0.0, "search": 0.0, "removal": 0.0}
        print(f"carving {seam_count} seams with {name} ...", flush=True)
        for _ in range(seam_count):
            t0 = time.perf_counter()
            energy = compute_energy(current)
            t1 = time.perf_counter()
            seam = finder(energy)
            t2 = time.perf_counter()
            current = remove_vertical_seam(current, seam)
            t3 = time.perf_counter()
            spent["energy"] += t1 - t0
            spent["search"] += t2 - t1
            spent["removal"] += t3 - t2
        rows.append({"method": name, **{step: seconds / seam_count for step, seconds in spent.items()}})
    return rows


def human_seconds(seconds, _position=None):
    if seconds >= 1:
        return f"{seconds:g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:g} ms"
    return f"{seconds * 1e6:g} µs"


def new_axes(plt, title, subtitle, size):
    figure, axes = plt.subplots(figsize=size, dpi=150)
    figure.suptitle(title, x=0.02, y=0.97, ha="left", fontsize=13, fontweight="bold", color=INK)
    axes.set_title(subtitle, loc="left", fontsize=9.5, color=MUTED, pad=10)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9, length=0)
    return figure, axes


def one_seam_chart(rows, path, image_name):
    import matplotlib
    matplotlib.use("Agg")                 # draw straight to a file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    figure, axes = new_axes(
        plt, "Time to find one seam",
        f"{image_name} at increasing sizes. Both axes are logarithmic; fastest of several runs.", (8, 4.8))
    axes.set_xscale("log")
    axes.set_yscale("log")
    for name, (_, label, colour, _) in METHODS.items():
        points = [(r["pixels"], r[f"{name}_seconds"]) for r in rows if r[f"{name}_seconds"]]
        if not points:
            continue
        pixels, seconds = zip(*points)
        axes.plot(pixels, seconds, color=colour, linewidth=2, marker="o", markersize=6,
                  markeredgecolor="white", markeredgewidth=1.2, label=label)
        slope = growth_rate(rows, name)
        axes.annotate(f"{label}\ntime ~ pixels^{slope:.2f}", (pixels[-1], seconds[-1]),
                      xytext=(9, 0), textcoords="offset points", va="center", fontsize=9, color=INK)

    axes.set_xticks([r["pixels"] for r in rows])
    axes.set_xticklabels([f"{r['width']}x{r['height']}" for r in rows], rotation=35, ha="right")
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.tick_params(axis="x", which="minor", bottom=False)
    axes.yaxis.set_major_formatter(FuncFormatter(human_seconds))
    axes.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_xlabel("image size (width x height)", fontsize=9.5, color=MUTED)
    axes.margins(x=0.05)
    axes.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    figure.subplots_adjust(left=0.1, right=0.8, top=0.84, bottom=0.2)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def breakdown_chart(rows, path, image_name, size_text, seam_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = new_axes(
        plt, "Where the time goes when removing one seam",
        f"{image_name} {size_text}, average over {seam_count} seams. Grey steps are the same for every method.",
        (8, 3.2))
    labels = [METHODS[r["method"]][1] for r in rows]
    for index, row in enumerate(rows):
        colour = METHODS[row["method"]][2]
        left = 0.0
        for step, fill, name in (("energy", SHARED_DARK, "Energy map (Sobel)"),
                                 ("removal", SHARED_LIGHT, "Seam removal"),
                                 ("search", colour, "Seam search")):
            milliseconds = row[step] * 1e3
            axes.barh(index, milliseconds, left=left, height=0.5, color=fill, edgecolor="white", linewidth=2,
                      label=name if index == 0 and step != "search" else None)
            left += milliseconds
        search_ms, total_ms = row["search"] * 1e3, left
        axes.text(total_ms, index, f"  {total_ms:.2f} ms total, search {search_ms:.3g} ms "
                  f"({search_ms / total_ms:.0%})", va="center", fontsize=9, color=INK)
    axes.barh([-9], [0], color="#ffffff", edgecolor=MUTED, linewidth=0.8, label="Seam search (method colour)")
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels(labels, fontsize=10, color=INK)
    axes.set_ylim(len(rows) - 0.5, -0.5)
    axes.set_xlim(0, max(sum(r[s] for s in ("energy", "search", "removal")) for r in rows) * 1e3 * 1.75)
    axes.set_xlabel("milliseconds per seam", fontsize=9.5, color=MUTED)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    figure.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.2)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def summary_lines(image_name, rows, breakdown, size_text, seam_count):
    def cell(seconds):
        return "not run" if seconds is None else human_seconds(float(f"{seconds:.3g}"))

    lines = [
        f"# Running time of DP, greedy and graph cut: {image_name}",
        "",
        "## Time to find one seam",
        "",
        "| Image | Pixels | DP | Greedy | Graph cut | Graph cut / DP | DP / Greedy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dp, gr, gc = r["dp_seconds"], r["greedy_seconds"], r["graphcut_seconds"]
        lines.append(
            f"| {r['width']}x{r['height']} | {r['pixels']:,} | {cell(dp)} | {cell(gr)} | {cell(gc)} "
            f"| {'' if gc is None else f'{gc / dp:,.0f}x'} | {dp / gr:,.0f}x |")
    lines += [
        "",
        "Measured growth (slope of the log-log line, time ~ pixels^slope):",
        "",
    ] + [
        f"- {label}: {growth_rate(rows, name):.2f}" for name, (_, label, _, _) in METHODS.items()
        if not np.isnan(growth_rate(rows, name))
    ] + [
        "",
        f"## One seam of a whole carve ({size_text}, average of {seam_count} seams)",
        "",
        "| Method | Energy map | Seam search | Seam removal | Total | Search share |",
        "|---|---|---|---|---|---|",
    ]
    for r in breakdown:
        total = r["energy"] + r["search"] + r["removal"]
        lines.append(
            f"| {METHODS[r['method']][1]} | {cell(r['energy'])} | {cell(r['search'])} | {cell(r['removal'])} "
            f"| {cell(total)} | {r['search'] / total:.0%} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Time DP, greedy and graph-cut seam finding")
    ap.add_argument("input")
    ap.add_argument("--scales", default="0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0",
                    help="comma-separated scales of the input image to time one seam at")
    ap.add_argument("--graphcut-max-pixels", type=int, default=13000,
                    help="skip graph cut on images with more pixels than this (default 13000)")
    ap.add_argument("--seams", type=int, default=20, help="seams removed for the breakdown (default 20)")
    ap.add_argument("--out", default=os.path.join("results", "benchmark"), help="output folder")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"could not read {args.input}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    image_name = os.path.basename(args.input)
    size_text = f"{img.shape[1]}x{img.shape[0]}"

    rows = one_seam_rows(img, [float(s) for s in args.scales.split(",")], args.graphcut_max_pixels)
    seam_count = max(1, min(args.seams, img.shape[1] - 1))
    breakdown = carve_breakdown(img, seam_count)

    with open(os.path.join(args.out, "timing_one_seam.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out, "carve_breakdown.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(breakdown[0]))
        writer.writeheader()
        writer.writerows(breakdown)

    one_seam_chart(rows, os.path.join(args.out, "timing_one_seam.png"), image_name)
    breakdown_chart(breakdown, os.path.join(args.out, "carve_breakdown.png"), image_name, size_text, seam_count)

    summary = "\n".join(summary_lines(image_name, rows, breakdown, size_text, seam_count)) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    print(f"charts written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
