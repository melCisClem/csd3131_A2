# Graph cut vs DP: castle.png shrunk to 68x46 -> 52x40 (22 seams)

| | DP | Graph cut |
|---|---|---|
| Total carve time (s) | 0.03 | 20.40 |
| Seam search only (s) | 0.023 | 20.39 (896x slower) |
| First seam energy | 1774.7 | 1774.7 |
| Total energy removed | 47310.0 | 47318.7 |

- First seams share 44 of 46 rows.
- Graph-cut seam / optimal DP seam on the same image: worst 1.000000x (optimal for 22 of 22 seams).
- Carved images are not identical (equal-cost seams broke ties differently).

## Time to find one seam

| Image | Nodes | Edges | Augmenting paths (BFS) | DP (s) | Graph cut (s) | Slower by | Same cost | Max flow = seam cost |
|---|---|---|---|---|---|---|---|---|
| 34x23 | 784 | 3016 | 213 | 0.00036 | 0.11 | 303x | yes | yes |
| 51x34 | 1736 | 6768 | 297 | 0.00052 | 0.32 | 619x | yes | yes |
| 68x46 | 3130 | 12286 | 470 | 0.00072 | 0.94 | 1,302x | yes | yes |
| 85x57 | 4847 | 19098 | 610 | 0.00095 | 1.92 | 2,027x | yes | yes |
| 102x69 | 7040 | 27812 | 770 | 0.00116 | 3.68 | 3,162x | yes | yes |
| 136x92 | 12514 | 49594 | 1067 | 0.00180 | 10.21 | 5,670x | yes | yes |
