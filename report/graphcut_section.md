# 4. Seam Carving with Graph Cut (Min-Cut / Max-Flow)

## 4.1 Idea

Instead of building the seam row by row as DP does, we turn the image into a flow network: think of it as a network of water pipes. Water enters at a **source** on the left edge and leaves at a **sink** on the right edge. To stop the flow, we must cut pipes, and each pipe costs its capacity to cut. We arrange the pipes so that **every cheapest way of cutting off the water is exactly a vertical seam**, and the cost of the cut is the total energy of that seam. Finding the minimum seam then becomes finding the **minimum s-t cut**, which the max-flow min-cut theorem lets us solve with a max-flow algorithm.

## 4.2 Graph construction

For an H x W energy map `e` (the same Sobel energy used by DP and greedy), we create one node per pixel plus a source `s` and a sink `t`:

| Edge | Capacity | Purpose |
|---|---|---|
| `s -> (i, 0)` | inf | The left column is always on the source side |
| `(i, j) -> (i, j+1)` | `e[i, j]` | Cutting this edge = removing pixel `(i, j)` |
| `(i, W-1) -> t` | `e[i, W-1]` | Lets the seam use the last column too |
| `(i, j+1) -> (i, j)` (backward) | inf | Each row is cut exactly once |
| `(i, j) -> (i-1, j-1)`, `(i, j) -> (i+1, j-1)` (diagonal) | inf | The seam is connected |

A cut (S, T) has finite capacity only if no infinite edge goes from S to T. Let `s_i` be the rightmost pixel of row `i` that lies in S.

- **Backward edges.** If `(i, j+1)` is in S, then `(i, j)` must be in S too. So each row of S is a solid block starting from the left, and the row is crossed by exactly one finite edge: `(i, s_i) -> (i, s_i + 1)`, which costs `e[i, s_i]`.
- **Diagonal edges.** If `(i, j)` is in S, then `(i-1, j-1)` and `(i+1, j-1)` are too. So `s_{i-1} >= s_i - 1` and `s_{i+1} >= s_i - 1`, which together give `|s_i - s_{i+1}| <= 1`. This is exactly the seam connectivity rule.

So the finite cuts are exactly the vertical seams, and cut capacity = sum of `e[i, s_i]` = seam cost. The minimum cut is the minimum-energy seam.

**Design decision.** The assignment brief connects the last column to the sink with infinite capacity. That would put the last column permanently on the sink side, so a seam could never remove a last-column pixel, while DP can. We give those edges capacity `e[i, W-1]` instead, and the graph cut then always finds the same minimum cost as DP. This matters in practice: on the castle image the very first minimum seam runs down the right edge of the image.

## 4.3 Algorithm

1. **Build the residual graph.** Every edge is stored with a paired reverse edge that starts at capacity 0 (edge `k` and edge `k XOR 1`).
2. **Max-flow (Edmonds-Karp).** Repeatedly BFS from `s` for the shortest path with spare capacity. Push the path's bottleneck capacity along it: forward residuals go down by that amount and reverse residuals go up. Stop when no path reaches `t`.
   - *Small speed-up:* each row on its own (`s -> (i,0) -> ... -> (i,W-1) -> t`) is a path of length W+1, which is the shortest possible, so Edmonds-Karp would pick these first anyway. We push them directly and skip H BFS passes.
3. **Find the cut.** The nodes reached by the final, failed BFS form the source side S.
4. **Read the seam.** In each row, `seam[i]` = rightmost pixel in S = (number of row-`i` pixels in S) - 1.

By the max-flow min-cut theorem, the final flow value equals the capacity of this cut, i.e. the seam cost. Our tests check this. Horizontal seams reuse the same code on the transposed energy map, exactly like DP and greedy. The function has the same signature as the DP seam finder, so it plugs straight into the shared `carve()` loop.

## 4.4 Results

Castle image shrunk to 68x46 and carved to 52x40 (22 seams). Commands: `python -m seamcarving.compare_graphcut castle.png --scale 0.2 --width 52 --height 40`. Pure-Python max-flow needs minutes per seam at full size, so the full carve uses a shrunk image.

*Figures: `results/castle_graphcut/first_seam.png`, `side_by_side.png`, `timing.png`.*

| | DP | Graph cut |
|---|---|---|
| Seam search time, whole carve | 0.02 s | 20.4 s (~900x slower) |
| First seam energy | 1774.7 | 1774.7 |
| Graph-cut seam / optimal DP seam on the same image | | 1.000x on all 22 seams |

Time to find **one** seam at different sizes:

| Image | Nodes | Edges | Augmenting paths | DP | Graph cut | Slower by |
|---|---|---|---|---|---|---|
| 34x23 | 784 | 3,016 | 213 | 0.36 ms | 0.11 s | 300x |
| 51x34 | 1,736 | 6,768 | 297 | 0.52 ms | 0.32 s | 620x |
| 68x46 | 3,130 | 12,286 | 470 | 0.72 ms | 0.94 s | 1,300x |
| 85x57 | 4,847 | 19,098 | 610 | 0.95 ms | 1.92 s | 2,000x |
| 102x69 | 7,040 | 27,812 | 770 | 1.2 ms | 3.7 s | 3,200x |
| 136x92 | 12,514 | 49,594 | 1,067 | 1.8 ms | 10.2 s | 5,700x |
| 341x229 (full size) | 78,091 | 311,218 | not measured | 10 ms | ~400 s (one run, busy machine) | ~40,000x |

- **Seams.** Every graph-cut seam has exactly the minimum cost, and the max-flow value always equals the seam cost. The seams themselves can differ from DP's where two seams tie: DP prefers the leftmost, while graph cut prefers the smallest source side. The first seams share 44 of 46 rows, with the same cost.
- **Visual quality.** It is the same as DP, because both remove optimal seams. The carved images differ only by these ties and cannot be told apart.
- **Running time.** Graph cut is 300x to 5,700x slower, and the gap grows with image size.

## 4.5 Complexity and discussion

| | DP | Graph cut (Edmonds-Karp) |
|---|---|---|
| Time per seam | O(HW) | O(V E^2) worst case, V = HW + 2, E ≈ 4HW |
| In practice | one pass over the image | one O(V + E) BFS **per augmenting path**, and the number of paths grows with the image (213 -> 1,067 above) |
| Space | O(HW) table | O(V + E): about 4HW edges, each stored with its reverse edge |

Exact edge count: `2HW + 2(W-1)(H-1)` ≈ 4HW.

**Why it is so much slower.** Both methods find the optimal seam, but DP uses the structure of the problem: a seam goes down one row at a time, so each pixel only needs its three parents. Max-flow is a general algorithm. It does not know about rows, so it keeps searching the whole graph for augmenting paths, and much of that flow is later cancelled through the reverse edges. A faster max-flow algorithm (Dinic, push-relabel, or Boykov-Kolmogorov as used in image segmentation) or a compiled language would shrink the constant, but not the gap to O(HW) DP.

**Why graph cut is still worth knowing.** The graph formulation is more flexible than DP. Rubinstein et al. (2008) use it for **video**, where a seam becomes a connected surface through a 3-D (x, y, time) volume. DP cannot do that, because a surface has no single "row by row" order to follow. The same graph also takes forward energy directly, by using `|I[i, j+1] - I[i, j-1]|` as the edge capacity.

**Conclusion.** For a single image, graph cut gives exactly the same seams and visual quality as DP, but is thousands of times slower. DP is the right choice here. Graph cut earns its cost only on problems DP cannot handle, such as video.
