"""Usage: python -m seamcarving in.png out.png --width desired_width --height desired_height [--forward | --greedy]"""
import argparse
import sys
import time

import cv2

from . import greedy
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
    args = ap.parse_args()
    if args.forward and args.greedy:
        print("error: choose either --forward or --greedy, not both", file=sys.stderr)
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
        else:
            out = carve(img, w, h, forward=args.forward)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0

    try:
        saved = cv2.imwrite(args.output, out)   # returns False (e.g. missing folder) or raises (unknown extension)
    except cv2.error:
        saved = False
    if not saved:
        print(f"error: could not write {args.output}", file=sys.stderr)
        return 1
    print(f"{W}x{H} -> {out.shape[1]}x{out.shape[0]} in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
