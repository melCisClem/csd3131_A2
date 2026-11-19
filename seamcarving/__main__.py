"""Usage: python -m seamcarving in.png out.png --width desired_width --height desired_height [--forward | --greedy | --graphcut]

A bare output file name is saved in results/<input name>_<method>/, e.g.
    python -m seamcarving castle.png out.png --width 250 --height 180 --greedy
writes results/castle_greedy/out.png. With --greedy the full greedy vs DP report
from compare.py (seam_costs.csv, timing.csv, summary.md, ...) goes in the same
folder. An output that already names a folder
(e.g. some/folder/out.png) is saved exactly there.
"""
import argparse
import os
import sys
import time

import cv2

from . import graphcut, greedy
from .carve import carve


def main() -> int:
    ap = argparse.ArgumentParser(description="Seam carving (DP)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--width", type=int, help="target width (default: unchanged)")
    ap.add_argument("--height", type=int, help="target height (default: unchanged)")
    ap.add_argument("--forward", action="store_true",
                    help="use forward-energy DP (fewer artifacts)")
    ap.add_argument("--greedy", action="store_true",
                    help="use the greedy row-by-row seam instead of DP (faster, worse seams)")
    ap.add_argument("--graphcut", action="store_true",
                    help="use min-cut/max-flow instead of DP (same seams, much slower: small images only)")
    args = ap.parse_args()
    if args.forward + args.greedy + args.graphcut > 1:
        print("error: choose only one of --forward, --greedy and --graphcut", file=sys.stderr)
        return 1

    img = cv2.imread(args.input)
    if img is None:
        print(f"could not read {args.input}", file=sys.stderr)
        return 1
    H, W = img.shape[:2]
    # `is None` (not `or`) so that an explicit 0 is rejected instead of meaning "unchanged"
    w = W if args.width is None else args.width
    h = H if args.height is None else args.height

    t0 = time.perf_counter()
    try:
        if args.greedy:
            out = carve(img, w, h, seam_finder=greedy.find_vertical_seam)
        elif args.graphcut:
            out = carve(img, w, h, seam_finder=graphcut.find_vertical_seam)
        else:
            out = carve(img, w, h, forward=args.forward)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0

    # Bare file name -> results/<input name>_<method>/<file name>
    output_path = args.output
    if not os.path.dirname(output_path):
        method = "greedy" if args.greedy else "graphcut" if args.graphcut else "forward" if args.forward else "dp"
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join("results", f"{input_name}_{method}", output_path)
    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    try:
        saved = cv2.imwrite(output_path, out)   # returns False (e.g. missing folder) or raises (unknown extension)
    except cv2.error:
        saved = False
    if not saved:
        print(f"error: could not write {output_path}", file=sys.stderr)
        return 1
    print(f"{W}x{H} -> {out.shape[1]}x{out.shape[0]} in {dt:.2f}s, saved to {output_path}")

    if args.greedy:
        # Full greedy vs DP report (CSVs, charts, seams, summary) next to the image
        from .compare import write_report
        write_report(args.input, img, w, h, output_folder or ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
