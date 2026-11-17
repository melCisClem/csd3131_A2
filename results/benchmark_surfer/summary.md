# Running time of DP, greedy and graph cut: surfer.png

## Time to find one seam

| Image | Pixels | DP | Greedy | Graph cut | Graph cut / DP | DP / Greedy |
|---|---|---|---|---|---|---|
| 240x135 | 32,400 | 2.07 ms | 118 µs | not run |  | 18x |
| 480x270 | 129,600 | 6.2 ms | 240 µs | not run |  | 26x |
| 960x540 | 518,400 | 17.9 ms | 483 µs | not run |  | 37x |
| 1440x809 | 1,164,960 | 36.7 ms | 764 µs | not run |  | 48x |
| 1920x1079 | 2,071,680 | 62.3 ms | 1.02 ms | not run |  | 61x |

Measured growth (slope of the log-log line, time ~ pixels^slope):

- DP: 0.81
- Greedy: 0.52

## One seam of a whole carve (1920x1079, average of 10 seams)

| Method | Energy map | Seam search | Seam removal | Total | Search share |
|---|---|---|---|---|---|
| DP | 63.2 ms | 67.9 ms | 55.2 ms | 186 ms | 36% |
| Greedy | 62.9 ms | 1.17 ms | 56.9 ms | 121 ms | 1% |
