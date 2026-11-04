"""Independent-set initialization and deterministic size-limited splitting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import networkx as nx
import torch

from .metrics import assert_proper_coloring, evaluate_partition, remap_groups, simple_undirected_edges

def independent_set_coloring(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Reproduce the project's canonical NetworkX ``independent_set`` coloring.

    Nodes and canonical undirected edges are inserted in sorted order, making
    tie resolution reproducible for a fixed NetworkX version.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    edges = simple_undirected_edges(edge_index, num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges.tolist())
    coloring = nx.coloring.greedy_color(graph, strategy="independent_set")
    group = torch.tensor(
        [coloring.get(node, -1) for node in range(num_nodes)],
        dtype=torch.long,
    )
    if bool((group < 0).any()):
        raise RuntimeError("canonical independent_set did not color every node")
    group = remap_groups(group)
    assert_proper_coloring(edges, num_nodes, group)
    return group


def split_large_groups(
    group: torch.Tensor,
    num_nodes: int,
    *,
    max_group_ratio: float = 0.58,
) -> torch.Tensor:
    """Split oversized independent color classes without changing membership.

    Every subset of an independent set remains independent, so this operation
    preserves proper coloring.  Nodes are sorted to keep the result bitwise
    deterministic.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if not 0.0 < max_group_ratio <= 1.0:
        raise ValueError("max_group_ratio must satisfy 0 < ratio <= 1")
    group = remap_groups(group)
    if len(group) != num_nodes:
        raise ValueError("group length must equal num_nodes")
    if num_nodes == 0:
        return group
    cap = max(1, int(math.floor(max_group_ratio * num_nodes)))
    normalized = torch.empty(num_nodes, dtype=torch.long)
    next_color = 0
    for color in range(int(group.max()) + 1):
        nodes = torch.where(group == color)[0].sort().values.tolist()
        for start in range(0, len(nodes), cap):
            chunk = nodes[start:start + cap]
            normalized[chunk] = next_color
            next_color += 1
    return remap_groups(normalized)


@dataclass
class InitialPartition:
    """A deterministic proper-coloring scaffold and its audit metadata."""

    method_id: str
    group: torch.Tensor
    diagnostics: dict[str, Any]
    implementation: str


def _finalize(
    method_id: str,
    implementation: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    group: torch.Tensor,
    diagnostics: dict[str, Any] | None = None,
) -> InitialPartition:
    group = remap_groups(group)
    assert_proper_coloring(edge_index, num_nodes, group)
    values = dict(diagnostics or {})
    values["hard_metrics"] = evaluate_partition(edge_index, num_nodes, group)
    return InitialPartition(
        method_id=method_id,
        group=group,
        diagnostics=values,
        implementation=implementation,
    )


def build_initial_partition(
    edge_index: torch.Tensor,
    num_nodes: int,
    canonical_independent_group: torch.Tensor,
    *,
    hard_max_group_ratio: float = 0.58,
) -> InitialPartition:
    """Return the cap-normalized canonical independent-set scaffold."""
    group = split_large_groups(
        canonical_independent_group,
        num_nodes,
        max_group_ratio=hard_max_group_ratio,
    )
    return _finalize(
        "chroma",
        "coloring.initialization.build_initial_partition",
        edge_index,
        num_nodes,
        group,
        {
            "hard_max_group_ratio": hard_max_group_ratio,
            "source": "canonical_independent_set_then_deterministic_cap_split",
        },
    )
