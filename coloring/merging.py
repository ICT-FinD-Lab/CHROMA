"""Greedy defective-coloring paths with global and local conflict controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .metrics import evaluate_partition, remap_groups, simple_undirected_edges


@dataclass
class PathState:
    step: int
    group: torch.Tensor
    metrics: dict[str, Any]
    merged_colors: tuple[int, int] | None
    merge_score: float | None


def _pair_matrices(
    group: torch.Tensor, edges: torch.Tensor, num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    colors = int(group.max()) + 1
    sizes = torch.bincount(group, minlength=colors).double()
    counts = torch.zeros(colors * colors, dtype=torch.float64)
    local = torch.zeros(colors * colors, dtype=torch.float64)
    if len(edges):
        degree = torch.zeros(num_nodes, dtype=torch.float64)
        ones = torch.ones(len(edges), dtype=torch.float64)
        degree.scatter_add_(0, edges[:, 0], ones)
        degree.scatter_add_(0, edges[:, 1], ones)
        left = group[edges[:, 0]]; right = group[edges[:, 1]]
        lo = torch.minimum(left, right); hi = torch.maximum(left, right)
        flat = lo * colors + hi
        counts.scatter_add_(0, flat, ones)
        weights = 1.0 / degree[edges[:, 0]].clamp_min(1.0) + 1.0 / degree[edges[:, 1]].clamp_min(1.0)
        local.scatter_add_(0, flat, weights)
    return sizes, counts.view(colors, colors), local.view(colors, colors)


def _merged(group: torch.Tensor, left: int, right: int) -> torch.Tensor:
    candidate = group.clone()
    candidate[candidate == right] = left
    return remap_groups(candidate)


def generate_soft_path(
    edge_index: torch.Tensor,
    num_nodes: int,
    initial_group: torch.Tensor,
    *,
    lambda_local: float = 1.0,
    global_conflict_cap: float = 0.05,
    local_conflict_p95_cap: float = 0.25,
    max_group_ratio_cap: float = 0.60,
    minimum_colors: int = 2,
    audit_top_b: int = 64,
) -> list[PathState]:
    if audit_top_b < 1:
        raise ValueError("audit_top_b must be positive")
    edges = simple_undirected_edges(edge_index, num_nodes)
    group = remap_groups(initial_group)
    initial_metrics = evaluate_partition(edges, num_nodes, group)
    adaptive_max_group = min(
        0.95,
        max(max_group_ratio_cap, float(initial_metrics["max_group_ratio"]) + 0.02),
    )
    states = [PathState(0, group.clone(), initial_metrics, None, None)]
    while int(group.max()) + 1 > minimum_colors:
        sizes, inter_edges, local_cost = _pair_matrices(group, edges, num_nodes)
        colors = len(sizes)
        ranked: list[tuple[float, int, int]] = []
        for left in range(colors):
            for right in range(left + 1, colors):
                merged_size = float(sizes[left] + sizes[right])
                if merged_size / num_nodes > adaptive_max_group:
                    continue
                delta_edges = float(inter_edges[left, right])
                predicted = states[-1].metrics["same_color_edges"] + delta_edges
                if len(edges) and predicted / len(edges) > global_conflict_cap:
                    continue
                delta_nonedges = float(sizes[left] * sizes[right]) - delta_edges
                denominator = delta_edges + lambda_local * float(local_cost[left, right]) + 1.0
                ranked.append((delta_nonedges / denominator, left, right))
        if not ranked:
            break
        chosen: tuple[float, int, int, torch.Tensor, dict[str, Any]] | None = None
        for score, left, right in sorted(ranked, reverse=True)[:audit_top_b]:
            candidate = _merged(group, left, right)
            metrics = evaluate_partition(edges, num_nodes, candidate)
            if (
                metrics["edge_conflict_ratio"] <= global_conflict_cap
                and metrics["local_conflict_p95"] <= local_conflict_p95_cap
                and metrics["max_group_ratio"] <= adaptive_max_group
            ):
                chosen = (score, left, right, candidate, metrics)
                break
        if chosen is None:
            break
        score, left, right, group, metrics = chosen
        previous = states[-1].metrics
        if metrics["num_colors"] != previous["num_colors"] - 1:
            raise RuntimeError("soft path must remove exactly one color per step")
        if metrics["nonedge_share_ratio"] + 1e-15 < previous["nonedge_share_ratio"]:
            raise RuntimeError("nonedge sharing must be monotonic")
        if metrics["edge_conflict_ratio"] + 1e-15 < previous["edge_conflict_ratio"]:
            raise RuntimeError("edge conflicts must be monotonic")
        states.append(PathState(len(states), group.clone(), metrics, (left, right), score))
    return states




def structural_knee_metadata(
    states: list[PathState],
    *,
    epsilon: float = 1e-15,
    risk_weight: float = 1.0,
) -> dict[str, Any]:
    """Describe the unique normalized structural knee of a safe merge path.

    The rule deliberately consumes only partition-structure metrics.  Let the
    hard state be state zero and the last feasible merge be state ``T``.  For
    every state ``t`` we compute::

        x_t = R_edge(t) / R_edge(T)
        y_t = (R_nonedge(t) - R_nonedge(0)) /
              (R_nonedge(T) - R_nonedge(0))

    and select the *earliest* state maximizing ``y_t - risk_weight*x_t``.
    ``risk_weight`` is a single global graph-only trade-off coefficient. Using the
    earliest maximizer makes ties deterministic and avoids silently preferring
    a riskier, deeper merge.  If either normalization denominator is zero (or
    the path only contains the hard state), the hard state is the documented
    fallback.

    No AUC, ACC, label, embedding, backbone, sigma, or validation result is
    accepted by this function.
    """
    if not states:
        raise ValueError("states must contain at least the hard state")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if risk_weight <= 0.0:
        raise ValueError("risk_weight must be positive")

    hard = states[0]
    terminal = states[-1]
    terminal_edge = float(terminal.metrics["edge_conflict_ratio"])
    nonedge_gain = (
        float(terminal.metrics["nonedge_share_ratio"])
        - float(hard.metrics["nonedge_share_ratio"])
    )
    if len(states) == 1:
        return {
            "selected_index": 0,
            "selected_step": int(hard.step),
            "reason": "hard_only_path",
            "terminal_edge_denominator": terminal_edge,
            "terminal_nonedge_gain_denominator": nonedge_gain,
            "risk_weight": risk_weight,
            "scores": [0.0],
        }
    if terminal_edge <= epsilon or nonedge_gain <= epsilon:
        return {
            "selected_index": 0,
            "selected_step": int(hard.step),
            "reason": "degenerate_normalization_fallback_hard",
            "terminal_edge_denominator": terminal_edge,
            "terminal_nonedge_gain_denominator": nonedge_gain,
            "risk_weight": risk_weight,
            "scores": [0.0 for _ in states],
        }

    base_nonedge = float(hard.metrics["nonedge_share_ratio"])
    scores: list[float] = []
    best_index = 0
    best_score = float("-inf")
    for index, state in enumerate(states):
        x_value = float(state.metrics["edge_conflict_ratio"]) / terminal_edge
        y_value = (
            float(state.metrics["nonedge_share_ratio"]) - base_nonedge
        ) / nonedge_gain
        score = y_value - risk_weight * x_value
        scores.append(score)
        # Strict improvement intentionally keeps the earliest exact/tolerance
        # tie.  The tolerance also makes tiny platform roundoff irrelevant.
        if score > best_score + epsilon:
            best_score = score
            best_index = index

    return {
        "selected_index": best_index,
        "selected_step": int(states[best_index].step),
        "reason": "earliest_argmax_normalized_y_minus_x",
        "terminal_edge_denominator": terminal_edge,
        "terminal_nonedge_gain_denominator": nonedge_gain,
        "risk_weight": risk_weight,
        "scores": scores,
    }


def select_structural_knee(
    states: list[PathState],
    *,
    epsilon: float = 1e-15,
    risk_weight: float = 1.0,
) -> PathState:
    """Return the deterministic, structure-only knee state of ``states``."""
    metadata = structural_knee_metadata(
        states, epsilon=epsilon, risk_weight=risk_weight,
    )
    return states[int(metadata["selected_index"])]
