# Seam Carving - CSD3131 Assignment 2

Content-aware image resizing using seam carving (Avidan & Shamir, 2007). The project implements three seam-finding strategies behind a common interface and compares them on seam quality, visual quality and running time:

| Method | Approach | Status |
|---|---|---|
| Dynamic programming | Exact minimum-cost seam in O(H*W) | Implemented |
| Greedy | Row-by-row local minimum in O(H) | Implemented |
| Graph cut | Min s t cut via max-flow | In progress |

## Requirements

- Python 3.10 or newer
- Dependencies listed in [`requirements.txt`](requirements.txt) (NumPy, OpenCV, pytest)

## Installation

1. Clone the repository and create a virtual environment:

   ```bash
   git clone <repository-url>
   cd csd3131_A2
   python -m venv .venv
   ```

2. Activate it:

   | Shell | Command |
   |---|---|
   | Linux / WSL | `source .venv/bin/activate` |
   | Windows PowerShell | `.venv\Scripts\Activate.ps1` |
   | Windows Command Prompt | `.venv\Scripts\activate.bat` |

   If PowerShell reports that running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python -m seamcarving <input> <output> [--width W] [--height H] [--forward | --greedy]
```

`--forward` selects forward-energy DP (see [Forward energy](#forward-energy-improvement)), which reduces visual artifacts in textured regions but takes slightly longer. `--greedy` selects the greedy seam (see [Greedy vs DP](#greedy-algorithm-vs-dynamic-programming)), which is faster but finds much worse seams.

Test examples: 
```bash
python -m seamcarving castle.png out.png --width 250 --height 180
```
341x229 -> 250x180 in 0.96s
```bash
python -m seamcarving surfer.png out2.png --width 920 --height 880
```
1920x1079 -> 920x880 in 145.69s

```bash
python -m seamcarving castle.png out.png --width 250 --height 180 --forward
```
341x229 -> 250x180 in 1.04s
```bash
python -m seamcarving surfer.png out2.png --width 920 --height 880 --forward
```
1920x1079 -> 920x880 in 161.60s

```bash
python -m seamcarving castle.png out.png --width 250 --height 180 --greedy
```
341x229 -> 250x180 in 0.31s

Omitted dimensions are left unchanged. The current implementation only reduces image size.

## Interactive Web App

Resize an image with sliders and watch every seam being removed, one at a time. It needs no extra dependencies: the server is Python's built-in `http.server`, and all carving is done by the same DP code as the command line.

```bash
python -m seamcarving.webapp [image] [--forward] [--max-side 500] [--port 8000] [--no-browser]
```

The browser opens at http://localhost:8000. The image is optional; one can also be uploaded on the page. Images are first shrunk so the longer side is at most `--max-side` pixels, to keep the animation smooth. Press `Ctrl+C` in the terminal to stop.

| Control | What it does |
|---|---|
| Width / Height | Target size. The image follows one seam at a time, width first and then height. Dragging back up restores seams in reverse order. |
| Speed | Seams per animation frame (1 to 20). |
| Pause / Play, Step | Stop or continue the animation; Step removes or restores exactly one seam. |
| Reset | Back to the original image and full-size sliders. |
| Upload image | Start again with another image. |
| Save result | Download the carved image as `carved.png` (wait for the animation to finish, or the current seam is included). |
| Forward energy | Switch DP method and replay the same resize from the original. |

| Panel | Shows |
|---|---|
| Seam carving | The carved image. Red is the seam about to be removed, green a seam just put back. |
| Energy map | Energy of the carved image (bright = important) with the same seam. |
| Original with every removed seam | All seams removed so far: red = vertical, blue = horizontal. |
| Plain scaling | The original scaled to the same size, for comparison. |

How it works: the page (`webapp.html`) only has the controls and displays pictures. About 30 times a second it asks the server (`webapp.py`) to move a few seams closer to the slider size. The server owns one `CarveSession` (`interactive.py`), which carves an *index map* (which original pixel sits at each position) instead of the image itself. The current image, the set of all removed seams and undo all follow from that map.

## Project Structure

```
seamcarving/
- energy.py     Energy map (Sobel gradient magnitude)
- backward.py   Standard (backward-energy) DP: cumulative-cost table and minimum seam
- forward.py    Forward-energy DP variant (optional improvement)
- greedy.py     Greedy row-by-row seam (alternative to DP)
- carve.py      Seam removal and the iterative resize loop
- __main__.py   Command-line interface
- compare.py    Greedy vs DP comparison: seams, visual quality, running time
- interactive.py  Step-by-step carving session with undo (used by the web app)
- webapp.py     Local web server for the interactive app
- webapp.html   The web page: sliders, buttons and the four panels
```

## How It Works

1. **Energy map.** Each pixel is scored by its Sobel gradient magnitude, `|dI/dx| + |dI/dy|`. Low energy means the pixel resembles its neighbours and is cheap to remove.
2. **Minimum seam.** A vertical seam is one pixel per row, with consecutive pixels in the same or adjacent columns. The cheapest seam is found with the recurrence

   ```
   M[0][j] = e[0][j]
   M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])
   ```

   The seam ends at `argmin M[H-1]` and is recovered by following backpointers upward.
3. **Remove and repeat.** The seam is deleted and the energy map is recomputed, until the target size is reached. Horizontal seams are found by transposing the image.

### Forward energy (improvement)

The standard method above uses *backward* energy: it scores each pixel by how different it is from its neighbours before removal. It ignores the new adjacencies a removal creates, so seams can cut through textured regions and edges and join pixels that never touched. *Forward energy* (Rubinstein et al., 2008) instead charges each seam step for the intensity differences it creates:

```
cU = |I[i,j+1] - I[i,j-1]|                  seam moves straight down
cL = cU + |I[i-1,j] - I[i,j-1]|             seam came from up-left
cR = cU + |I[i-1,j] - I[i,j+1]|             seam came from up-right

M[0][j] = cU(0, j)
M[i][j] = min(M[i-1][j-1] + cL, M[i-1][j] + cU, M[i-1][j+1] + cR)
```

The DP structure and complexity are unchanged (O(H*W)). It needs the image rather than an energy map, so it is selected with `carve(..., forward=True)` or exposed directly as `find_vertical_seam_forward(img)`.

## Greedy Algorithm vs Dynamic Programming

The greedy seam (`greedy.py`) makes one local decision per row and never looks back:

```
seam[0] = argmin_j e[0][j]
seam[i] = argmin over j in {seam[i-1]-1, seam[i-1], seam[i-1]+1} of e[i][j]
```

It starts at the cheapest pixel of the top row and steps down to the cheapest of the (up to) three pixels below. The search is O(W + H) time and O(H) space per seam, against O(H*W) time and space for the DP table. It is not optimal: a cheap pixel now can lead the seam into a wall of high energy later, and since the decision is never revisited the seam has to cut through it. The DP avoids this because `M[i][j]` already includes the best way to reach every pixel.

Run the comparison with

```bash
python -m seamcarving.compare castle.png --width 250 --height 180
```

It writes images, a per-seam CSV and `summary.md` to `results/<image>/`. "Optimality gap" compares each greedy seam with the DP seam on the same image, so it isolates the greedy decision itself.

Results (backward energy, same machine):

| | castle 341x229 -> 250x180 (DP / Greedy) | surfer 1920x1079 -> 920x880 (DP / Greedy) |
|---|---|---|
| Whole carve | 0.64 s / 0.31 s (2.0x faster) | 134.1 s / 77.4 s (1.7x faster) |
| Seam search only | 0.34 s / 0.02 s (19x faster) | 57.6 s / 1.7 s (33x faster) |
| Total energy removed | 837,827 / 1,515,325 (1.81x) | 39.6M / 51.3M (1.30x) |
| Greedy seam vs optimal on the same image | median 3.21x, worst 12.61x | median 3.06x, worst 7.59x |
| Seams where greedy was optimal | 0 of 140 | 0 of 1199 |

- **Seams.** The first greedy and DP seams share no rows on either image. DP seams bend around objects to stay in low-energy sky or water. Greedy seams follow whatever is cheapest right below them, so they drift straight down and cut through the castle towers and across the surfers.
- **Visual quality.** Greedy is not comparable to DP. On castle the middle tower is visibly squeezed and the towers lose their shape; on surfer the left surfer is squashed and the board is distorted, while DP keeps both surfers intact.
- **Running time.** The greedy search itself is 19-33x faster (O(H) instead of O(H*W) per seam), but the whole carve is only about 2x faster, because both methods spend the same time recomputing the Sobel energy map and removing the seam after every step. Those steps, not the seam search, are what dominate the greedy run.

## API Reference

All functions are importable from the `seamcarving` package. A *vertical seam* is an integer array of length `H`, where `seam[i]` is the column removed in row `i`.

| Function | Description |
|---|---|
| `compute_energy(img)` | Returns an `(H, W)` float64 energy map. |
| `find_vertical_seam(energy)` | Returns the minimum-cost vertical seam (DP). |
| `find_horizontal_seam(energy)` | Returns the minimum-cost horizontal seam; `seam[j]` is the row removed in column `j`. |
| `remove_vertical_seam(img, seam)` | Returns the image one pixel narrower. |
| `remove_horizontal_seam(img, seam)` | Returns the image one pixel shorter. |
| `seam_cost(energy, seam)` | Sum of energy along a seam. |
| `is_valid_seam(seam, H, W)` | Checks one pixel per row, in range, and connectivity. |
| `carve(img, new_w, new_h, seam_finder, on_seam, forward)` | Runs the full resize pipeline. `forward=True` uses forward-energy DP. |
| `find_vertical_seam_forward(img)` | Minimum forward-energy vertical seam; takes the image, not an energy map. |
| `forward_seam_cost(img, seam)` | Forward-energy cost of a seam (for comparisons and tests). |
| `find_vertical_seam_greedy(energy)` | Greedy row-by-row vertical seam (from `greedy.py`). |
| `find_horizontal_seam_greedy(energy)` | Greedy horizontal seam. |

### Adding a seam-finding method

A seam finder is any function with the signature `energy (H, W) -> seam (int array, length H)`. Pass it to `carve`; horizontal seams are handled automatically by transposition.

```python
from seamcarving import carve

result = carve(img, 300, 200, seam_finder=my_find_vertical_seam)
```

### Step-by-step visualisation

`carve` accepts an optional `on_seam(kind, seam, img_before)` callback, invoked before each removal. `kind` is `"vertical"` or `"horizontal"`, and `seam` is expressed in the coordinates of `img_before`.

### Comparing methods

Different methods may return different seams of equal cost, since ties are broken differently (DP prefers the leftmost). When testing equivalence, compare `seam_cost` values rather than seam arrays.

## Testing

```bash
pytest
```

## References

- Avidan & Shamir, [*Seam Carving for Content-Aware Image Resizing*](https://dl.acm.org/doi/10.1145/1275808.1276390), SIGGRAPH 2007
- Rubinstein, Shamir & Avidan, [*Improved Seam Carving for Video Retargeting*](https://dl.acm.org/doi/pdf/10.1145/1360612.1360615), SIGGRAPH 2008
- [Seam carving - Wikipedia](https://en.wikipedia.org/wiki/Seam_carving)
- [*Seam Carving | Week 2, lecture 7 | 18.S191 MIT Fall 2020*](https://www.youtube.com/watch?v=rpB6zQNsbQU)

## AI Usage Declaration

Model used: Claude Opus 5.5 and Claude Sonnet 5.5 both on High effort

usage: 
- Helped to plot the graph for comparison between DP and Greedy 
- Checking through of code for bugs 

## Team

Group A2 - CSD3131 Assignment 2. Team members and SIT IDs to be added.