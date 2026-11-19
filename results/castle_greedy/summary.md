# Greedy vs DP: castle.png 341x229 -> 250x180

| | DP | Greedy |
|---|---|---|
| Total carve time (s) | 0.60 | 0.29 |
| Seam search only (s) | 0.33 | 0.02 |
| First seam energy | 2689 | 7109 (2.64x) |
| Total energy removed | 837827 | 1515325 (1.81x) |
| Mean energy per seam | 5984 | 10824 |

- First seams share 0 of 229 rows.
- Greedy seam / optimal DP seam on the same image: median 3.21x, worst 12.61x; greedy was optimal for 0 of 140 seams.
- Seam search speed-up: 18.8x; whole carve: 2.1x (energy recomputation and seam removal are the same for both).

## Time to find one seam

| Image | Pixels | DP (s) | Greedy (s) | DP / greedy | DP seam cost | Greedy seam cost |
|---|---|---|---|---|---|---|
| 85x57 | 4845 | 0.00034 | 0.00003 | 12.2x | 1823 | 2933 |
| 170x114 | 19380 | 0.00088 | 0.00006 | 15.9x | 2032 | 11419 |
| 256x172 | 44032 | 0.00148 | 0.00009 | 17.3x | 2293 | 5739 |
| 341x229 | 78089 | 0.00226 | 0.00012 | 19.6x | 2689 | 7109 |
| 512x344 | 176128 | 0.00474 | 0.00017 | 27.6x | 2618 | 8206 |
| 682x458 | 312356 | 0.00743 | 0.00023 | 32.7x | 2705 | 8294 |
| 1023x687 | 702801 | 0.01505 | 0.00034 | 44.6x | 2805 | 15620 |
