# Greedy vs DP: surfer.png 1920x1079 -> 250x180

| | DP | Greedy |
|---|---|---|
| Total carve time (s) | 261.06 | 71.49 |
| Seam search only (s) | 120.86 | 1.24 |
| First seam energy | 11252 | 35111 (3.12x) |
| Total energy removed | 95865494 | 98016657 (1.02x) |
| Mean energy per seam | 37316 | 38154 |

- First seams share 0 of 1079 rows.
- Greedy seam / optimal DP seam on the same image: median 2.86x, worst 7.59x; greedy was optimal for 0 of 2569 seams.
- Seam search speed-up: 97.6x; whole carve: 3.7x (energy recomputation and seam removal are the same for both).

## Time to find one seam

| Image | Pixels | DP (s) | Greedy (s) | DP / greedy | DP seam cost | Greedy seam cost |
|---|---|---|---|---|---|---|
| 480x270 | 129600 | 0.00833 | 0.00055 | 15.1x | 2217 | 10277 |
| 960x540 | 518400 | 0.02892 | 0.00089 | 32.4x | 4171 | 15982 |
| 1440x809 | 1164960 | 0.05363 | 0.00135 | 39.7x | 6435 | 18149 |
| 1920x1079 | 2071680 | 0.08181 | 0.00181 | 45.2x | 11252 | 35111 |
| 2880x1618 | 4659840 | 0.16660 | 0.00401 | 41.5x | 12523 | 33751 |
| 3840x2158 | 8286720 | 0.33738 | 0.00363 | 93.0x | 13763 | 47589 |
| 5760x3237 | 18645120 | 0.67350 | 0.00569 | 118.3x | 16169 | 53279 |
