"""Exact NetworkX independent-set strategy with cached induced degrees.

Preserves the actual Python set iteration order used to resolve degree ties.
It does not substitute lowest-node-ID ties or another coloring strategy.
The only change is caching degree counts and doing the same argmin in NumPy.
"""
import time
import numpy as np
import networkx as nx

def maximal_independent_set_cached(G):
    result = set()
    remaining = set(G)
    # Keep the same views and set operations as the reference implementation.
    base = G
    degrees = np.zeros(max(G, default=0) + 1, dtype=np.int64)
    for node, degree in G.degree:
        degrees[node] = degree
    while remaining:
        G = G.subgraph(remaining)
        order = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
        v = int(order[np.argmin(degrees[order])])
        result.add(v)
        removed = set(G[v]) | {v}
        remaining -= removed
        for node in removed:
            for neighbor in base[node]:
                if neighbor in remaining:
                    degrees[neighbor] -= 1
    return result

def cached_independent_strategy(G, colors):
    remaining_nodes = set(G)
    index = 0
    start = time.time()
    while remaining_nodes:
        nodes = maximal_independent_set_cached(G.subgraph(remaining_nodes))
        remaining_nodes -= nodes
        print(f'Color class {index}: {len(nodes)} nodes; remaining={len(remaining_nodes)}; '
              f'seconds={time.time()-start:.1f}', flush=True)
        index += 1
        yield from nodes

def verify_equivalence():
    import contextlib
    import io
    from networkx.algorithms.coloring.greedy_coloring import _maximal_independent_set
    graphs = [nx.empty_graph(150), nx.path_graph(200), nx.complete_graph(45),
              nx.star_graph(200), nx.complete_bipartite_graph(35,80), nx.cycle_graph(150)]
    for seed in range(20):
        graphs.append(nx.gnp_random_graph(80 + seed*3, .015 + .01*(seed%7), seed=seed))
        # Sparse, non-consecutive integers exercise set collision and tie order.
        original=nx.gnp_random_graph(100,.04,seed=seed)
        mapping={i: (i*257+seed)%100003 for i in original}
        graphs.append(nx.relabel_nodes(original,mapping))
    for graph in graphs:
        assert _maximal_independent_set(graph) == maximal_independent_set_cached(graph)
        reference = nx.coloring.greedy_color(graph, strategy='independent_set')
        with contextlib.redirect_stdout(io.StringIO()):
            actual = nx.coloring.greedy_color(graph, strategy=cached_independent_strategy)
        assert reference == actual, 'Canonical coloring differs; do not run TGRASP'
    return len(graphs)
