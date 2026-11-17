# Testing and Verification

## 1 Approach

The three seam finders share one interface (`energy -> seam`), so they can all be checked the same way. We test in three layers:

1. **Worked examples.** Tiny inputs where the answer is computed by hand (a 3x3 DP table, a black/white edge for Sobel), so a reader can verify the expected values without running anything.
2. **Brute-force oracle.** `tests/seam_oracle.py` lists *every* valid seam of a small image and takes the cheapest. It is exponential, but obviously correct. DP, forward-energy DP and graph cut must all match it. This is what proves "optimal", rather than only "DP and graph cut agree with each other".
3. **Properties of the whole pipeline.** Things that must hold for any image: the image gets exactly one pixel narrower per seam, only seam pixels disappear, the input is not modified, the requested size is reached.

Random inputs use fixed seeds, so every run is identical. Seams are compared by **cost**, not by position, because two methods can pick different seams of equal cost.

## 2 What each test file proves

| File | Tests | What it proves |
|---|---|---|
| `test_energy.py` | 13 | Energy map is one non-negative float per pixel; flat image = 0; a black/white edge gives exactly 4 x 255 = 1020 on the two columns beside it; matches a Sobel written by hand in NumPy; colour is averaged, alpha ignored; energy of the transposed image is the transposed energy. |
| `test_dp.py` | 82 | DP table and backpointers match a hand-filled example; the table follows the recurrence cell by cell; the DP seam costs the same as the cheapest of all seams (7 shapes x 5 seeds, plus 10 tie-heavy cases); ties go left; horizontal seams are optimal too; the number of seams is at least 2^(m-1) (question 2a-i). |
| `test_forward.py` | 44 | Forward-energy cost matches the formula written out pixel by pixel; the forward DP seam is the cheapest of all seams under that formula; colour/alpha handled; `carve(forward=True)` bypasses the seam finder. |
| `test_carve.py` | 44 | Removing a seam makes the image exactly one pixel narrower (or shorter) and deletes only the seam pixels; `carve` reaches the requested size, rejects enlarging and zero sizes, reports every step through `on_seam`; a striped object on a flat background survives 10 seams untouched with DP, forward DP and graph cut. |
| `test_methods_agree.py` | 51 | On the same energy map: graph cut cost = DP cost = brute-force minimum, and greedy cost >= DP cost; holds with ties and all-zero energy, on the real castle image (vertical and horizontal), and at every step of a full graph-cut carve; max-flow value = seam cost; all four methods shrink the image one pixel per seam. |
| `test_cli.py` | 12 | `python -m seamcarving` writes an image of the requested size with every method flag, gives the same pixels as calling `carve`, and exits with an error (writing nothing) for bad sizes, two method flags or a missing file. |
| `test_benchmark.py` | 4 | The timing script times every method on the same image, skips graph cut above the size limit, computes the growth rate correctly and writes its charts. No test asserts a speed. |
| existing: `test_greedy.py`, `test_graphcut.py`, `test_interactive.py` | 71 | Written by the method owners: greedy follows the local minimum and falls into a designed trap; graph cut matches DP on 30 random maps and can use the last column; the interactive session matches `carve` and undo restores the original. |

Total: 321 tests (250 new, 71 existing).

## 3 How the tests map to the report's claims

| Claim in the report | Test that backs it |
|---|---|
| DP finds the minimum-energy seam | `test_dp.py::test_seam_is_optimal` (against all seams) |
| Graph cut finds the same cost as DP | `test_methods_agree.py::test_cost_ordering_*`, `test_methods_on_the_castle_image` |
| Max-flow value equals the seam cost | `test_methods_on_the_castle_image`, `test_graphcut.py::test_max_flow_equals_seam_cost` |
| Greedy is never better than DP, and can be worse | `test_cost_ordering_*`, `test_greedy.py::test_greedy_trap_where_dp_wins` |
| Each seam removes exactly one pixel per row | `test_carve.py::test_image_gets_exactly_one_pixel_narrower`, `test_methods_agree.py::test_every_method_shrinks_one_pixel_per_seam` |
| Seam carving protects content | `test_carve.py::test_flat_background_is_removed_before_the_object` |
| Number of seams is exponential in m | `test_dp.py::test_number_of_seams_grows_exponentially` |

## 4 Results

`python -m pytest` on Python 3.13.17, Windows 10: **321 passed, 0 failed, in about 2 seconds**. The slowest single test is graph cut on the 34x23 castle (0.07 s). No bugs were found in the seam-finding code.

A passing suite only means something if it would fail on wrong code. To check that, we broke a copy of the code in six ways, one at a time, and re-ran the 317 tests that check the algorithms:

| Deliberate bug | Tests that fail |
|---|---|
| DP never looks at the up-right neighbour | 90 |
| DP always ends the seam in column 0 | 102 |
| Energy uses Sobel x only | 3 |
| Seam removal uses the seam upside down | 14 |
| Forward DP charges the wrong diagonal cost | 4 |
| Graph cut built without the diagonal edges | 56 |

Every bug was caught. The graph-cut case shows the diagonal infinite edges are what keep the seam connected: without them the "seam" jumps between columns.

One setup problem was found: without `pytest.ini`, the plain `pytest` command from the README fails at import (`seamcarving` is not on the path); only `python -m pytest` worked. `pytest.ini` fixes this.

## 5 Running time

`python -m seamcarving.benchmark castle.png` times all three methods on the same energy maps. *Figures: `results/benchmark/timing_one_seam.png`, `carve_breakdown.png`.*

Time to find one seam (fastest of several runs):

| Image | Pixels | DP | Greedy | Graph cut | Graph cut / DP | DP / Greedy |
|---|---|---|---|---|---|---|
| 34x23 | 782 | 286 us | 19.3 us | 63.4 ms | 221x | 15x |
| 68x46 | 3,128 | 530 us | 39.9 us | 593 ms | 1,120x | 13x |
| 136x92 | 12,512 | 1.18 ms | 79.8 us | 8.51 s | 7,190x | 15x |
| 341x229 (castle) | 78,089 | 4.3 ms | 203 us | not run | | 21x |
| 1920x1079 (surfer) | 2,071,680 | 62.3 ms | 1.02 ms | not run | | 61x |

Fitting a straight line on log-log axes gives the measured growth, time ~ pixels^k:

| Method | Theory (one seam) | Measured k |
|---|---|---|
| Greedy | O(W + H), about pixels^0.5 | 0.51 (castle), 0.52 (surfer) |
| DP | O(H*W) = pixels^1 | 0.60 (castle), 0.81 (surfer) |
| Graph cut | O(V*E^2) worst case | 1.76 |

- **Greedy matches theory.** It walks down H rows, and H grows like the square root of the pixel count.
- **DP measures below its O(H*W).** Each row is one vectorised NumPy step, so the time is H x (fixed cost per row + cost per pixel x W). On small images the fixed cost dominates and DP looks like O(H). The exponent rises towards 1 as the image grows (0.60 on castle, 0.81 on surfer), and the DP/greedy gap widens with it, from 15x to 61x.
- **Graph cut grows faster than linear**, so the gap to DP widens with every size: 221x at 34x23, 7,190x at 136x92. It is far below the O(V*E^2) worst case, but each augmenting path is a full BFS and the number of paths also grows with the image.

One seam inside the carve loop (energy map, search, removal), average per seam:

| Image | Method | Energy map | Seam search | Seam removal | Total | Search share |
|---|---|---|---|---|---|---|
| castle 341x229 | DP | 2.61 ms | 4.62 ms | 2.37 ms | 9.6 ms | 48% |
| castle 341x229 | Greedy | 2.66 ms | 0.235 ms | 2.23 ms | 5.13 ms | 5% |
| surfer 1920x1079 | DP | 63.2 ms | 67.9 ms | 55.2 ms | 186 ms | 36% |
| surfer 1920x1079 | Greedy | 62.9 ms | 1.17 ms | 56.9 ms | 121 ms | 1% |

The greedy search is 20 to 60 times faster than DP, but the whole step is only 1.5 to 1.9 times faster, because recomputing the energy map and removing the seam cost the same for every method. For greedy, those two steps are 95% or more of the time. Making DP's search faster has the same ceiling: the bigger win is to avoid recomputing the whole energy map after each seam, since only the pixels next to the seam change.

Timings are from one Windows 10 laptop and will differ elsewhere; the ratios and growth rates are the result.

## 6 Continuous integration (bonus)

`.github/workflows/tests.yml` runs on every push and pull request:

1. **test**: installs `requirements.txt` and runs all 321 tests on four combinations, Linux and Windows with Python 3.10 and 3.13 (the oldest version the README supports and the newest).
2. **benchmark**: only if the tests pass, runs the timing script on castle.png, shows the tables on the run page and attaches the charts and CSV files as a downloadable artifact.

A newer push to the same branch cancels the older run. *Status: the workflow file is written and parses as valid YAML, but it has not run on GitHub yet, because nothing has been pushed.*

## 7 Limits

- The oracle only scales to about 6x6, so optimality on large images is shown indirectly: graph cut and DP are independent algorithms and agree on the 34x23 castle.
- Graph cut is only tested on small images because one full-size seam takes minutes.
- Visual quality is not tested automatically, apart from the synthetic object-on-background case. It is judged from the comparison images in `results/`.
- The web page itself (`webapp.py`, `webapp.html`) is not tested; the session logic behind it is.
- Running time is measured by the benchmark and comparison scripts, not asserted in tests, since timings vary between machines.
