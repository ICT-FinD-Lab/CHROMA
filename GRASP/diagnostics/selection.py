"""Evaluate candidate perturbations on validation nodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


SELECTOR_SCHEMA_VERSION = 1
SUPPORTED_COLORING_STRATEGIES = (
    "connected_sequential_bfs",
    "connected_sequential_dfs",
    "independent_set",
    "largest_first",
    "random_sequential",
    "saturation_largest_first",
    "smallest_last",
)


@dataclass(frozen=True)
class PartitionStatistics:
    num_nodes: int
    num_colors: int
    max_group_size: int
    mean_group_size: float
    group_size_cv: float
    group_size_gini: float
    normalized_entropy: float
    same_color_pair_rate: float


@dataclass(frozen=True)
class CandidateEvaluation:
    algorithm: str
    raw_auc_mean: float
    raw_auc_std: float
    effective_auc_mean: float
    privacy_gap_mean: float
    validation_acc_mean: float
    validation_acc_std: float
    validation_acc_drop: float
    feasible: bool
    score: float
    partition: PartitionStatistics | None


def _as_cpu_long_1d(values: torch.Tensor, name: str) -> torch.Tensor:
    values = values.detach().to(device="cpu", dtype=torch.long).view(-1)
    if values.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return values


def validate_group_partition(group: torch.Tensor, num_nodes: int) -> torch.Tensor:
    group = _as_cpu_long_1d(group, "group")
    if group.numel() != num_nodes:
        raise ValueError(
            f"group has {group.numel()} entries, expected {num_nodes}"
        )
    if int(group.min().item()) < 0:
        raise ValueError("group ids must be non-negative")
    unique = torch.unique(group, sorted=True)
    expected = torch.arange(unique.numel(), dtype=torch.long)
    if not torch.equal(unique, expected):
        raise ValueError("group ids must be contiguous and start at zero")
    return group


def partition_statistics(group: torch.Tensor) -> PartitionStatistics:
    group = _as_cpu_long_1d(group, "group")
    if int(group.min().item()) < 0:
        raise ValueError("group ids must be non-negative")

    counts = torch.bincount(group).to(torch.float64)
    if torch.any(counts == 0):
        raise ValueError("group ids must be contiguous and start at zero")

    num_nodes = int(group.numel())
    num_colors = int(counts.numel())
    mean_group_size = float(counts.mean().item())
    group_size_cv = (
        float(counts.std(unbiased=False).item()) / mean_group_size
        if mean_group_size > 0
        else 0.0
    )

    sorted_counts, _ = torch.sort(counts)
    indices = torch.arange(
        1,
        num_colors + 1,
        dtype=torch.float64,
    )
    weighted_sum = float((indices * sorted_counts).sum().item())
    total = float(sorted_counts.sum().item())
    gini = (
        (2.0 * weighted_sum) / (num_colors * total)
        - (num_colors + 1.0) / num_colors
    )

    probabilities = counts / total
    entropy = float(
        (-(probabilities * torch.log(probabilities))).sum().item()
    )
    normalized_entropy = (
        entropy / math.log(num_colors) if num_colors > 1 else 1.0
    )

    if num_nodes > 1:
        ordered_same_color_pairs = float(
            (counts * (counts - 1.0)).sum().item()
        )
        same_color_pair_rate = ordered_same_color_pairs / (
            num_nodes * (num_nodes - 1)
        )
    else:
        same_color_pair_rate = 0.0

    return PartitionStatistics(
        num_nodes=num_nodes,
        num_colors=num_colors,
        max_group_size=int(counts.max().item()),
        mean_group_size=mean_group_size,
        group_size_cv=group_size_cv,
        group_size_gini=gini,
        normalized_entropy=normalized_entropy,
        same_color_pair_rate=same_color_pair_rate,
    )


def binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().to(device="cpu", dtype=torch.float64).view(-1)
    labels = labels.detach().to(device="cpu", dtype=torch.bool).view(-1)
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length")
    if scores.numel() == 0:
        raise ValueError("scores and labels must not be empty")

    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order].to(torch.float64)

    distinct_ends = torch.nonzero(
        sorted_scores[:-1] != sorted_scores[1:],
        as_tuple=False,
    ).view(-1)
    threshold_ends = torch.cat(
        [
            distinct_ends,
            torch.tensor([scores.numel() - 1], dtype=torch.long),
        ]
    )

    true_positives = torch.cumsum(sorted_labels, dim=0)[threshold_ends]
    false_positives = (
        threshold_ends.to(torch.float64) + 1.0 - true_positives
    )
    true_positives = torch.cat(
        [torch.zeros(1, dtype=torch.float64), true_positives]
    )
    false_positives = torch.cat(
        [torch.zeros(1, dtype=torch.float64), false_positives]
    )

    tpr = true_positives / positive_count
    fpr = false_positives / negative_count
    return float(torch.trapz(tpr, fpr).item())


def correlation_pair_scores(
    embeddings: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_nodes, dimension]")
    if pair_index.ndim != 2 or pair_index.shape[0] != 2:
        raise ValueError("pair_index must have shape [2, num_pairs]")

    centered = embeddings - embeddings.mean(dim=1, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=1)
    source, target = pair_index.to(embeddings.device)
    return (normalized[source] * normalized[target]).sum(dim=1)


def _unique_undirected_edge_codes(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    edge_index = edge_index.detach().to(device="cpu", dtype=torch.long)
    source, target = edge_index
    keep = (
        (source != target)
        & node_mask[source]
        & node_mask[target]
    )
    source = source[keep]
    target = target[keep]
    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    return torch.unique(low * num_nodes + high, sorted=True)


def sample_balanced_validation_pairs(
    edge_index: torch.Tensor,
    valid_idx: torch.Tensor,
    num_nodes: int,
    total_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_samples < 2:
        raise ValueError("total_samples must be at least two")

    valid_nodes = torch.unique(
        _as_cpu_long_1d(valid_idx, "valid_idx"),
        sorted=True,
    )
    if valid_nodes.numel() < 3:
        raise ValueError("at least three validation nodes are required")
    if int(valid_nodes.min().item()) < 0 or int(valid_nodes.max().item()) >= num_nodes:
        raise ValueError("valid_idx contains an out-of-range node")

    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[valid_nodes] = True
    positive_codes = _unique_undirected_edge_codes(
        edge_index,
        num_nodes,
        node_mask,
    )
    if positive_codes.numel() == 0:
        raise ValueError("validation nodes contain no positive graph edges")

    target_per_class = min(
        total_samples // 2,
        int(positive_codes.numel()),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positive_perm = torch.randperm(
        positive_codes.numel(),
        generator=generator,
    )[:target_per_class]
    selected_positive_codes = positive_codes[positive_perm]

    positive_code_set = set(int(value) for value in positive_codes.tolist())
    selected_negative_codes: set[int] = set()
    max_attempts = max(target_per_class * 200, 10_000)
    attempts = 0
    while len(selected_negative_codes) < target_per_class and attempts < max_attempts:
        remaining = target_per_class - len(selected_negative_codes)
        batch_size = max(remaining * 4, 256)
        first = valid_nodes[
            torch.randint(
                valid_nodes.numel(),
                (batch_size,),
                generator=generator,
            )
        ]
        second = valid_nodes[
            torch.randint(
                valid_nodes.numel(),
                (batch_size,),
                generator=generator,
            )
        ]
        low = torch.minimum(first, second)
        high = torch.maximum(first, second)
        for source, target in zip(low.tolist(), high.tolist()):
            attempts += 1
            if source == target:
                continue
            code = int(source * num_nodes + target)
            if code in positive_code_set:
                continue
            selected_negative_codes.add(code)
            if len(selected_negative_codes) == target_per_class:
                break

    if len(selected_negative_codes) < target_per_class:
        raise RuntimeError(
            "could not sample enough validation non-edges; "
            "reduce --pair-samples"
        )

    negative_codes = torch.tensor(
        sorted(selected_negative_codes),
        dtype=torch.long,
    )
    all_codes = torch.cat([selected_positive_codes, negative_codes])
    pair_index = torch.stack(
        [all_codes // num_nodes, all_codes % num_nodes],
        dim=0,
    )
    pair_labels = torch.cat(
        [
            torch.ones(target_per_class, dtype=torch.bool),
            torch.zeros(target_per_class, dtype=torch.bool),
        ]
    )
    return pair_index, pair_labels


def _classification_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    valid_idx: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None,
) -> float:
    logits = embeddings @ classifier_weight.t()
    if classifier_bias is not None:
        logits = logits + classifier_bias
    predictions = logits[valid_idx].argmax(dim=-1)
    targets = labels[valid_idx]
    return float((predictions == targets).to(torch.float32).mean().item())


def _evaluate_noise_candidate(
    *,
    algorithm: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    valid_idx: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None,
    pair_index: torch.Tensor,
    pair_labels: torch.Tensor,
    sigma: float,
    monte_carlo_samples: int,
    seed: int,
    baseline_acc: float,
    acc_tolerance: float,
    lambda_acc: float,
    lambda_std: float,
    noise_sampler: Callable[[torch.Generator], torch.Tensor],
    partition: PartitionStatistics | None,
) -> CandidateEvaluation:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")

    device = embeddings.device
    labels = labels.to(device=device, dtype=torch.long).view(-1)
    valid_idx = valid_idx.to(device=device, dtype=torch.long).view(-1)
    classifier_weight = classifier_weight.to(device)
    if classifier_bias is not None:
        classifier_bias = classifier_bias.to(device)
    pair_index = pair_index.to(device)
    pair_labels = pair_labels.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw_aucs = []
    validation_accs = []
    with torch.no_grad():
        for _ in range(monte_carlo_samples):
            noisy_embeddings = embeddings + noise_sampler(generator)
            pair_scores = correlation_pair_scores(
                noisy_embeddings,
                pair_index,
            )
            raw_aucs.append(binary_roc_auc(pair_scores, pair_labels))
            validation_accs.append(
                _classification_accuracy(
                    noisy_embeddings,
                    labels,
                    valid_idx,
                    classifier_weight,
                    classifier_bias,
                )
            )

    raw_auc_tensor = torch.tensor(raw_aucs, dtype=torch.float64)
    acc_tensor = torch.tensor(validation_accs, dtype=torch.float64)
    privacy_gap_tensor = torch.abs(raw_auc_tensor - 0.5)

    raw_auc_mean = float(raw_auc_tensor.mean().item())
    raw_auc_std = float(raw_auc_tensor.std(unbiased=False).item())
    privacy_gap_mean = float(privacy_gap_tensor.mean().item())
    effective_auc_mean = 0.5 + privacy_gap_mean
    validation_acc_mean = float(acc_tensor.mean().item())
    validation_acc_std = float(acc_tensor.std(unbiased=False).item())
    validation_acc_drop = max(0.0, baseline_acc - validation_acc_mean)
    feasible = validation_acc_drop <= acc_tolerance
    score = (
        privacy_gap_mean
        + lambda_acc * validation_acc_drop
        + lambda_std * raw_auc_std
    )

    return CandidateEvaluation(
        algorithm=algorithm,
        raw_auc_mean=raw_auc_mean,
        raw_auc_std=raw_auc_std,
        effective_auc_mean=effective_auc_mean,
        privacy_gap_mean=privacy_gap_mean,
        validation_acc_mean=validation_acc_mean,
        validation_acc_std=validation_acc_std,
        validation_acc_drop=validation_acc_drop,
        feasible=feasible,
        score=score,
        partition=partition,
    )


def evaluate_coloring_candidate(
    *,
    algorithm: str,
    group: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    valid_idx: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None,
    pair_index: torch.Tensor,
    pair_labels: torch.Tensor,
    sigma: float,
    monte_carlo_samples: int,
    seed: int,
    baseline_acc: float,
    acc_tolerance: float,
    lambda_acc: float,
    lambda_std: float,
) -> CandidateEvaluation:
    device = embeddings.device
    num_nodes, embedding_dim = embeddings.shape
    group_cpu = validate_group_partition(group, num_nodes)
    group_device = group_cpu.to(device)
    num_colors = int(group_cpu.max().item()) + 1

    def sample_coloring_noise(generator: torch.Generator) -> torch.Tensor:
        color_noise = torch.randn(
            (num_colors, embedding_dim),
            device=device,
            generator=generator,
            dtype=embeddings.dtype,
        )
        return sigma * color_noise[group_device]

    return _evaluate_noise_candidate(
        algorithm=algorithm,
        embeddings=embeddings,
        labels=labels,
        valid_idx=valid_idx,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        pair_index=pair_index,
        pair_labels=pair_labels,
        sigma=sigma,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
        baseline_acc=baseline_acc,
        acc_tolerance=acc_tolerance,
        lambda_acc=lambda_acc,
        lambda_std=lambda_std,
        noise_sampler=sample_coloring_noise,
        partition=partition_statistics(group_cpu),
    )


def evaluate_grasp_candidate(
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    valid_idx: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None,
    pair_index: torch.Tensor,
    pair_labels: torch.Tensor,
    sigma: float,
    probability: float,
    monte_carlo_samples: int,
    seed: int,
    baseline_acc: float,
    acc_tolerance: float,
    lambda_acc: float,
    lambda_std: float,
) -> CandidateEvaluation:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")

    device = embeddings.device
    num_nodes, embedding_dim = embeddings.shape

    def sample_grasp_noise(generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(
            embeddings.shape,
            device=device,
            generator=generator,
            dtype=embeddings.dtype,
        )
        identical_mask = (
            torch.rand(
                num_nodes,
                device=device,
                generator=generator,
            )
            < probability
        )
        shared_noise = torch.randn(
            (1, embedding_dim),
            device=device,
            generator=generator,
            dtype=embeddings.dtype,
        )
        noise[identical_mask] = shared_noise
        return sigma * noise

    return _evaluate_noise_candidate(
        algorithm="GRASP",
        embeddings=embeddings,
        labels=labels,
        valid_idx=valid_idx,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        pair_index=pair_index,
        pair_labels=pair_labels,
        sigma=sigma,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
        baseline_acc=baseline_acc,
        acc_tolerance=acc_tolerance,
        lambda_acc=lambda_acc,
        lambda_std=lambda_std,
        noise_sampler=sample_grasp_noise,
        partition=None,
    )


def choose_candidate(
    evaluations: Iterable[CandidateEvaluation],
) -> tuple[CandidateEvaluation, bool]:
    evaluations = list(evaluations)
    if not evaluations:
        raise ValueError("at least one candidate evaluation is required")
    feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
    fallback_used = not feasible
    pool = feasible if feasible else evaluations
    selected = min(
        pool,
        key=lambda evaluation: (
            evaluation.score,
            evaluation.effective_auc_mean,
            -evaluation.validation_acc_mean,
            evaluation.algorithm,
        ),
    )
    return selected, fallback_used


def clean_validation_accuracy(bundle: dict[str, Any], device: torch.device) -> float:
    embeddings = bundle["embeddings"].to(device)
    labels = bundle["labels"].to(device=device, dtype=torch.long).view(-1)
    valid_idx = bundle["split_idx"]["valid"].to(
        device=device,
        dtype=torch.long,
    )
    classifier_weight = bundle["classifier_weight"].to(device)
    classifier_bias = bundle.get("classifier_bias")
    if classifier_bias is not None:
        classifier_bias = classifier_bias.to(device)
    with torch.no_grad():
        return _classification_accuracy(
            embeddings,
            labels,
            valid_idx,
            classifier_weight,
            classifier_bias,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_file(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_export_path(
    requested_path: str | Path,
    run: int,
    runs: int,
) -> Path:
    requested_path = Path(requested_path)
    if "{run}" in str(requested_path):
        return Path(str(requested_path).format(run=run))
    if runs == 1:
        return requested_path
    suffix = requested_path.suffix or ".pt"
    return requested_path.with_name(
        f"{requested_path.stem}.run_{run}{suffix}"
    )


@torch.no_grad()
def export_selector_bundle(
    *,
    model: torch.nn.Module,
    dataset: Any,
    split_idx: dict[str, torch.Tensor],
    args: Any,
    run: int,
) -> Path:
    if args.model != "MPNN" or not hasattr(model, "pred_local"):
        raise ValueError("selector export currently supports MPNN models only")
    if float(args.defence_sigma) != 0.0:
        raise ValueError("selector reference export requires defence_sigma=0")
    if args.defense_group_noise or args.static_correlated_noise:
        raise ValueError("selector reference export requires ungrouped clean inference")

    output_path = resolve_export_path(
        args.selector_export_path,
        run,
        args.runs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    embeddings, logits = model(
        dataset.graph["node_feat"],
        dataset.graph["edge_index"],
        get_embedding=True,
    )
    payload = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "embeddings": embeddings.detach().cpu(),
        "clean_logits": logits.detach().cpu(),
        "labels": dataset.label.detach().cpu().view(-1),
        "edge_index": dataset.graph["edge_index"].detach().cpu(),
        "split_idx": {
            key: value.detach().cpu()
            for key, value in split_idx.items()
        },
        "classifier_weight": model.pred_local.weight.detach().cpu(),
        "classifier_bias": (
            None
            if model.pred_local.bias is None
            else model.pred_local.bias.detach().cpu()
        ),
        "metadata": {
            "dataset": args.dataset,
            "gnn": args.gnn,
            "model": args.model,
            "seed": int(args.seed),
            "run": int(run),
            "local_layers": int(args.local_layers),
            "hidden_channels": int(args.hidden_channels),
            "defence_limit": bool(args.defence_limit),
            "selection_scope": "validation_nodes",
        },
    }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    if was_training:
        model.train()
    return output_path


def candidate_to_dict(evaluation: CandidateEvaluation) -> dict[str, Any]:
    payload = asdict(evaluation)
    return payload


def write_selection_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
