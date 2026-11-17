# Running time of DP, greedy and graph cut: castle.png

## Time to find one seam

| Image | Pixels | DP | Greedy | Graph cut | Graph cut / DP | DP / Greedy |
|---|---|---|---|---|---|---|
| 34x23 | 782 | 286 µs | 19.3 µs | 63.4 ms | 221x | 15x |
| 51x34 | 1,734 | 375 µs | 29.4 µs | 219 ms | 584x | 13x |
| 68x46 | 3,128 | 530 µs | 39.9 µs | 593 ms | 1,120x | 13x |
| 102x69 | 7,038 | 829 µs | 60.3 µs | 2.55 s | 3,080x | 14x |
| 136x92 | 12,512 | 1.18 ms | 79.8 µs | 8.51 s | 7,190x | 15x |
| 205x137 | 28,085 | 2 ms | 121 µs | not run |  | 17x |
| 273x183 | 49,959 | 3.12 ms | 162 µs | not run |  | 19x |
| 341x229 | 78,089 | 4.3 ms | 203 µs | not run |  | 21x |

Measured growth (slope of the log-log line, time ~ pixels^slope):

- DP: 0.60
- Greedy: 0.51
- Graph cut: 1.76

## One seam of a whole carve (341x229, average of 20 seams)

| Method | Energy map | Seam search | Seam removal | Total | Search share |
|---|---|---|---|---|---|
| DP | 2.61 ms | 4.62 ms | 2.37 ms | 9.6 ms | 48% |
| Greedy | 2.66 ms | 235 µs | 2.23 ms | 5.13 ms | 5% |
