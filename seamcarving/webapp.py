"""Seam-carving web app: a tiny local server + one web page (webapp.html).

    python -m seamcarving.webapp [photo.jpg] [--forward] [--max-side 500] [--port 8000]

then use the page that opens in the browser (http://localhost:8000). Needs only
Python, OpenCV and NumPy -- the server is Python's built-in http.server.

Who does what:
    browser (webapp.html)   sliders and buttons; about 30 times a second it asks
                            the server "take me N seams closer to this size"
    server (this file)      owns ONE CarveSession (interactive.py), removes or
                            restores the seams, and answers with four pictures

So all the carving is still done by the Python DP code; the page only displays.

Requests:
    GET  /                                  the page
    GET  /step?width=W&height=H&count=N     up to N steps towards WxH -> state + pictures (JSON)
    POST /reset?forward=0|1                 back to the original, choose the DP method
    POST /upload   (body = an image file)   start again with a new image
"""
import argparse
import base64
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .energy import compute_energy
from .interactive import (
    HORIZONTAL,
    REMOVED_BY_HORIZONTAL,
    REMOVED_BY_VERTICAL,
    VERTICAL,
    CarveSession,
    Step,
)

# colours are BGR because OpenCV images are BGR
RED = (0, 0, 255)        # seam about to be removed / removed vertical seams
GREEN = (0, 220, 0)      # seam that was just put back
BLUE = (255, 110, 0)     # removed horizontal seams

PAGE_FILE = Path(__file__).with_name("webapp.html")

# ---- the app's state: one image, one session, shared by every request ----
session: Optional[CarveSession] = None     # None until an image is given
max_side = 500
session_lock = threading.Lock()            # one request at a time may touch the session


def shrink_to_max_side(image: np.ndarray) -> np.ndarray:
    # Shrink big images so the animation stays smooth.
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def draw_seam(image: np.ndarray, kind: str, seam: np.ndarray, colour) -> None:
    # Paint the seam's pixels (in place).
    if kind == VERTICAL:
        image[np.arange(len(seam)), seam] = colour     # seam[row] = column
    else:
        image[seam, np.arange(len(seam))] = colour     # seam[col] = row


def energy_picture(energy: np.ndarray) -> np.ndarray:
    # Grey picture of the energy map: brighter = more important.
    # The square root lifts the dark values so faint edges are visible too.
    brightness = np.sqrt(energy / max(energy.max(), 1e-9))
    grey = (brightness * 255).astype(np.uint8)
    return np.dstack([grey, grey, grey])


def as_data_url(image: np.ndarray) -> str:
    # PNG bytes written as text, which the page can put straight into <img src=...>.
    _, png = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def take_steps(target_width: int, target_height: int, count: int) -> dict:
    """Move up to `count` seams towards the target size and describe the result."""
    original_height, original_width = session.original.shape[:2]
    target_width = min(max(target_width, 1), original_width)       # keep the request in range
    target_height = min(max(target_height, 1), original_height)

    last_step: Optional[Step] = None
    for _ in range(count):
        step = session.step_towards(target_width, target_height)
        if step is None:        # already at the requested size
            break
        last_step = step

    # Top pictures: the image that contains the seam, with the seam coloured in
    # (or just the current image when nothing happened).
    if last_step is None:
        carved = session.image
        energy = energy_picture(compute_energy(carved))
    else:
        colour = RED if last_step.action == "remove" else GREEN
        carved = last_step.image.copy()
        energy = energy_picture(compute_energy(last_step.image))
        draw_seam(carved, last_step.kind, last_step.seam, colour)
        draw_seam(energy, last_step.kind, last_step.seam, colour)

    # Bottom-left: every removed pixel painted onto the original.
    all_seams = session.original.copy()
    all_seams[session.removed_by == REMOVED_BY_VERTICAL] = RED
    all_seams[session.removed_by == REMOVED_BY_HORIZONTAL] = BLUE

    # Bottom-right: ordinary scaling of the original to the same size.
    width, height = session.size
    scaled = cv2.resize(session.original, (width, height), interpolation=cv2.INTER_AREA)

    return {
        "original": [original_width, original_height],
        "size": [width, height],
        "vertical": session.seams_removed(VERTICAL),
        "horizontal": session.seams_removed(HORIZONTAL),
        "forward": session.forward,
        # done = nothing left to animate: the page can stop asking until a slider moves
        "done": last_step is None and (width, height) == (target_width, target_height),
        "panels": {
            "carved": as_data_url(carved),
            "energy": as_data_url(energy),
            "seams": as_data_url(all_seams),
            "scaled": as_data_url(scaled),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE_FILE.read_bytes())
        elif url.path == "/step":
            query = parse_qs(url.query)
            with session_lock:
                if session is None:
                    self.send_json(200, {"empty": True})     # no image yet: the page asks for an upload
                    return
                current_width, current_height = session.size
                try:
                    # a missing number means "stay as you are"
                    width = int(query.get("width", [current_width])[0])
                    height = int(query.get("height", [current_height])[0])
                    count = min(int(query.get("count", [0])[0]), 100)
                except ValueError:
                    self.send_json(400, {"error": "width, height and count must be whole numbers"})
                    return
                self.send_json(200, take_steps(width, height, count))
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        global session
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if url.path == "/reset":
            forward = parse_qs(url.query).get("forward", ["0"])[0] == "1"
            with session_lock:
                if session is not None:
                    session.forward = forward
                    session.reset()
            self.send_json(200, {"ok": True})
        elif url.path == "/upload":
            image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)   # None if not an image
            if image is None:
                self.send_json(400, {"error": "that file is not an image OpenCV can read"})
                return
            with session_lock:
                forward = session.forward if session is not None else False
                session = CarveSession(shrink_to_max_side(image), forward=forward)
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "not found"})

    def send_json(self, status: int, data: dict) -> None:
        self.send_bytes(status, "application/json", json.dumps(data).encode("utf-8"))

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass    # stay quiet: otherwise every animation frame prints a line


def main() -> int:
    global session, max_side
    ap = argparse.ArgumentParser(description="Seam carving web app (sliders + seam animation)")
    ap.add_argument("input", nargs="?", help="image to start with (optional: you can upload one in the page)")
    ap.add_argument("--forward", action="store_true", help="start with forward-energy DP")
    ap.add_argument("--max-side", type=int, default=500,
                    help="shrink images first so the longer side is at most this (default: 500)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    max_side = args.max_side
    if args.input is not None:
        image = cv2.imread(args.input)    # always 3-channel BGR
        if image is None:
            print(f"error: could not read {args.input}", file=sys.stderr)
            return 1
        session = CarveSession(shrink_to_max_side(image), forward=args.forward)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)   # 127.0.0.1 = this computer only
    except OSError as error:
        print(f"error: could not start on port {args.port} ({error}); try --port 8001", file=sys.stderr)
        return 1
    address = f"http://localhost:{args.port}"
    print(f"Seam carving app running at {address}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())