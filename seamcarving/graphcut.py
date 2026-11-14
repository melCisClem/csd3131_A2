"""Graph-cut seam (assignment item 4): seam finding as a minimum s-t cut.

Build a flow network with one node per pixel, plus a source s and a sink t:

    s -> (i, 0)                 capacity inf     left column is always on the source side
    (i, j) -> (i, j+1)          capacity e[i, j] cutting this edge = removing pixel (i, j)
    (i, W-1) -> t               capacity e[i, W-1]  (so the last column can be removed too)
    (i, j+1) -> (i, j)          capacity inf     backward edge
    (i, j) -> (i-1, j-1)        capacity inf     diagonal edges
    (i, j) -> (i+1, j-1)        capacity inf

A cut (S, T) only has finite capacity if no inf edge goes from S to T:
  - backward edges: if (i, j+1) is in S then so is (i, j), so S is a prefix of every
    row, i.e. each row is cut exactly once, at s_i = rightmost column of row i in S;
  - diagonal edges: if (i, j) is in S then so are (i-1, j-1) and (i+1, j-1), so
    s_{i-1} >= s_i - 1 and s_{i+1} >= s_i - 1, i.e. |s_i - s_{i+1}| <= 1.
So finite cuts are exactly the vertical seams, and the capacity of a cut is
sum_i e[i, s_i] = the seam cost. The minimum cut is therefore the minimum seam.

The PDF connects the last column to t with infinite capacity; that would stop
a seam from ever using the last column, which DP can. Giving that edge e[i, W-1]
instead makes the minimum cut cost exactly equal to the DP minimum.

Max-flow is Edmonds-Karp (Ford-Fulkerson with BFS shortest augmenting paths).
By the max-flow min-cut theorem the final flow value equals the minimum cut.
The source side S is the set of nodes still reachable from s in the residual
graph after the last (failed) BFS; in each row, the rightmost pixel in S is the
seam pixel.

The first H augmenting paths are found without BFS: every row on its own is a
path s -> (i, 0) -> ... -> (i, W-1) -> t of length W+1, which is the shortest
possible, so pushing min_j e[i, j] along each row is exactly what Edmonds-Karp
would do first. It just saves H BFS passes.

V = H*W + 2 nodes, E = 2HW + 2(W-1)(H-1) ~ 4*H*W edges. Edmonds-Karp is O(V * E^2) in the worst
case; in practice each augmenting path costs one O(V + E) BFS, so it is far
slower than the O(H*W) DP, which uses the row-by-row structure the general
max-flow algorithm does not know about. Space O(V + E) for the graph.

Same signature and seam format as backward.find_vertical_seam, so it plugs
straight into carve(img, w, h, seam_finder=graphcut.find_vertical_seam).
"""
from collections import deque

import numpy as np

INF = float("inf")
EPS = 1e-9   # residual capacity below this counts as zero (float rounding)


class FlowNetwork:
    # Residual graph stored as edge lists. Edge k and edge k ^ 1 are a pair:
    # an edge and its reverse (which starts with capacity 0). Pushing f units
    # along k lowers capacity[k] by f and raises capacity[k ^ 1] by f.

    def __init__(self, node_count: int):
        self.edges_from = [[] for _ in range(node_count)]   # node -> list of edge ids
        self.edge_to = []                                   # edge id -> node it points to
        self.capacity = []                                  # edge id -> residual capacity
        self.augmenting_paths = 0                           # paths pushed so far (for the report)

    def add_edge(self, u: int, v: int, capacity: float) -> int:
        edge = len(self.edge_to)
        self.edges_from[u].append(edge)
        self.edge_to.append(v)
        self.capacity.append(capacity)
        self.edges_from[v].append(edge + 1)                 # reverse edge, empty for now
        self.edge_to.append(u)
        self.capacity.append(0.0)
        return edge

    def push(self, edge: int, amount: float) -> None:
        self.capacity[edge] -= amount
        self.capacity[edge ^ 1] += amount

    def bfs(self, source: int, sink: int):
        # Shortest path in the residual graph. Returns (came_by_edge, reached_sink):
        # came_by_edge[v] = edge used to reach v, -1 if v was not reached
        # (the source is marked -2 so it counts as reached).
        came_by_edge = [-1] * len(self.edges_from)
        came_by_edge[source] = -2
        queue = deque([source])
        edges_from, edge_to, capacity = self.edges_from, self.edge_to, self.capacity
        while queue:
            u = queue.popleft()
            for edge in edges_from[u]:
                v = edge_to[edge]
                if came_by_edge[v] == -1 and capacity[edge] > EPS:
                    came_by_edge[v] = edge
                    if v == sink:
                        return came_by_edge, True
                    queue.append(v)
        return came_by_edge, False

    def max_flow(self, source: int, sink: int, initial_flow: float = 0.0):
        # Edmonds-Karp. Returns (flow value, came_by_edge of the final BFS);
        # nodes with came_by_edge != -1 are the source side S of the min cut.
        flow = initial_flow
        while True:
            came_by_edge, reached_sink = self.bfs(source, sink)
            if not reached_sink:
                return flow, came_by_edge

            # Walk back from the sink: bottleneck = smallest residual capacity on the path
            path = []
            v = sink
            while v != source:
                edge = came_by_edge[v]
                path.append(edge)
                v = self.edge_to[edge ^ 1]                  # the reverse edge points back to the tail
            bottleneck = min(self.capacity[edge] for edge in path)
            for edge in path:
                self.push(edge, bottleneck)
            flow += bottleneck
            self.augmenting_paths += 1


def build_seam_network(energy: np.ndarray):
    # The network described at the top of the file. Returns (network, source,
    # sink, row_edges) where row_edges[i] lists the finite edges of row i from
    # left to right (the W edges a straight row path goes through).
    height, width = energy.shape
    source, sink = height * width, height * width + 1
    network = FlowNetwork(height * width + 2)
    energy = energy.tolist()                                # plain floats are faster to read

    def node(row, col):
        return row * width + col

    row_edges = []
    for row in range(height):
        network.add_edge(source, node(row, 0), INF)
        edges = []
        for col in range(width):
            next_node = node(row, col + 1) if col + 1 < width else sink
            edges.append(network.add_edge(node(row, col), next_node, energy[row][col]))
            if col + 1 < width:
                network.add_edge(node(row, col + 1), node(row, col), INF)      # backward
            if col >= 1:
                if row >= 1:
                    network.add_edge(node(row, col), node(row - 1, col - 1), INF)   # up-left
                if row + 1 < height:
                    network.add_edge(node(row, col), node(row + 1, col - 1), INF)   # down-left
        row_edges.append(edges)
    return network, source, sink, row_edges


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    # Minimum-energy vertical seam via min-cut / max-flow. Returns int array of length height.
    return find_vertical_seam_with_stats(energy)[0]


def find_vertical_seam_with_stats(energy: np.ndarray):
    # Same seam, plus (max-flow value, augmenting paths found by BFS, edge count)
    # so the comparison can show that flow value == seam cost and why it is slow.
    height, width = energy.shape
    network, source, sink, row_edges = build_seam_network(energy)

    # Shortest augmenting paths first: each row straight across (see top of file)
    initial_flow = 0.0
    for edges in row_edges:
        bottleneck = min(network.capacity[edge] for edge in edges)
        for edge in edges:
            network.push(edge, bottleneck)
        initial_flow += bottleneck

    flow, came_by_edge = network.max_flow(source, sink, initial_flow)

    # Source side S = nodes reached by the last BFS. Each row of S is a prefix
    # (backward inf edges), so its rightmost pixel is (pixels in S) - 1.
    in_source_side = np.array(came_by_edge[:height * width]) != -1
    seam = in_source_side.reshape(height, width).sum(axis=1).astype(np.int64) - 1
    return seam, flow, network.augmenting_paths, len(network.edge_to) // 2


def find_horizontal_seam(energy: np.ndarray) -> np.ndarray:
    # Minimum-energy horizontal seam: seam[col] = row removed in column col
    return find_vertical_seam(energy.T)  # Transpose the energy so the same code can be reused
