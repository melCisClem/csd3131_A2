# Seam Carving - CSD3131 Assignment 2

Content-aware image resizing using seam carving (Avidan & Shamir, 2007). The project implements three seam-finding strategies behind a common interface and compares them on seam quality, visual quality and running time:

| Method | Approach | Status |
|---|---|---|
| Dynamic programming | Exact minimum-cost seam in O(H*W) | Implemented |
| Greedy | Row-by-row local minimum | In progress |
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
python -m seamcarving <input> <output> [--width W] [--height H] [--forward]
```

`--forward` selects forward-energy DP (see [Forward energy](#forward-energy-improvement)), which reduces visual artifacts in textured regions but takes slightly longer.

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
- carve.py      Seam removal and the iterative resize loop
- __main__.py   Command-line interface
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
- 
- Checking through of code for bugs 

## Team

Group A2 - CSD3131 Assignment 2. Team members and SIT IDs to be added.