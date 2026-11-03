"""Sparse O(N+E) partition metrics; no N-by-N tensors are constructed."""

from __future__ import annotations

import math
from typing import Any

import torch


def remap_groups(group: torch.Tensor) -> torch.Tensor:
    group = group.detach().cpu().long().flatten()
    if group.numel() == 0:
        return group
    _, inverse = torch.unique(group, sorted=True, return_inverse=True)
    return inverse.long()


def simple_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be rank two")
    edges = edge_index.t() if edge_index.shape[0] == 2 else edge_index
    if edges.shape[1] != 2:
        raise ValueError("edge_index must have shape [2,E] or [E,2]")
    if len(edges) == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if int(edges.min()) < 0 or int(edges.max()) >= num_nodes:
        raise ValueError("edge endpoint outside [0,num_nodes)")
    edges = edges[edges[:, 0] != edges[:, 1]]
    if not len(edges):
        return torch.empty((0, 2), dtype=torch.long)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def assert_proper_coloring(edge_index: torch.Tensor, num_nodes: int, group: torch.Tensor) -> None:
    group = remap_groups(group)
    if len(group) != num_nodes:
        raise ValueError("group length must equal num_nodes")
    edges = simple_undirected_edges(edge_index, num_nodes)
    conflicts = group[edges[:, 0]] == group[edges[:, 1]] if len(edges) else torch.zeros(0, dtype=torch.bool)
    if bool(conflicts.any()):
        raise AssertionError(f"Hard coloring has {int(conflicts.sum())} conflicting edges")


def evaluate_partition(
    edge_index: torch.Tensor,
    num_nodes: int,
    group: torch.Tensor,
) -> dict[str, Any]:
    group = remap_groups(group)
    if len(group) != num_nodes:
        raise ValueError("group length must equal num_nodes")
    edges = simple_undirected_edges(edge_index, num_nodes)
    num_colors = int(group.max()) + 1 if num_nodes else 0
    sizes = torch.bincount(group, minlength=num_colors).double()
    same_pairs = int(((sizes * (sizes - 1)) / 2).sum())
    same_edges = (
        group[edges[:, 0]] == group[edges[:, 1]]
        if len(edges) else torch.zeros(0, dtype=torch.bool)
    )
    edge_conflicts = int(same_edges.sum())
    total_pairs = num_nodes * (num_nodes - 1) // 2
    total_nonedges = total_pairs - len(edges)
    same_nonedges = same_pairs - edge_conflicts
    if same_nonedges < 0:
        raise RuntimeError("same-color nonedge count became negative")

    degree = torch.zeros(num_nodes, dtype=torch.float64)
    local_conflicts = torch.zeros(num_nodes, dtype=torch.float64)
    if len(edges):
        ones = torch.ones(len(edges), dtype=torch.float64)
        degree.scatter_add_(0, edges[:, 0], ones)
        degree.scatter_add_(0, edges[:, 1], ones)
        conflict_values = same_edges.double()
        local_conflicts.scatter_add_(0, edges[:, 0], conflict_values)
        local_conflicts.scatter_add_(0, edges[:, 1], conflict_values)
    active = degree > 0
    local = local_conflicts[active] / degree[active] if bool(active.any()) else torch.zeros(0)
    mean_size = float(sizes.mean()) if len(sizes) else 0.0
    std_size = float(sizes.std(unbiased=False)) if len(sizes) else 0.0
    return {
        "num_nodes": num_nodes,
        "num_edges": int(len(edges)),
        "num_colors": num_colors,
        "min_group_size": int(sizes.min()) if len(sizes) else 0,
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "mean_group_size": mean_size,
        "group_size_std": std_size,
        "group_cv": std_size / mean_size if mean_size else 0.0,
        "same_color_pairs": same_pairs,
        "same_pair_ratio": same_pairs / total_pairs if total_pairs else 0.0,
        "same_color_edges": edge_conflicts,
        "edge_conflict_ratio": edge_conflicts / len(edges) if len(edges) else 0.0,
        "same_color_nonedges": same_nonedges,
        "nonedge_share_ratio": same_nonedges / total_nonedges if total_nonedges else 0.0,
        "max_group_ratio": float(sizes.max()) / num_nodes if num_nodes and len(sizes) else 0.0,
        "local_conflict_mean": float(local.mean()) if len(local) else 0.0,
        "local_conflict_p95": float(torch.quantile(local, 0.95)) if len(local) else 0.0,
        "local_conflict_max": float(local.max()) if len(local) else 0.0,
        "proper_coloring": edge_conflicts == 0,
    }
