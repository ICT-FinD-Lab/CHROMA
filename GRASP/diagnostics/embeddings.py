"""Isolated same-embedding sampled-vs-exact AUC diagnostic export.

This module is only activated by ``GRASP_SAME_EMBEDDING_DIAGNOSTIC_DIR``.
It does not alter training, perturbation, grouping, or model selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .selection import sample_balanced_validation_pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def export_same_embedding_diagnostic(
    *,
    attacker: Any,
    model: torch.nn.Module,
    dataset: Any,
    split_idx: dict[str, torch.Tensor],
    args: Any,
    run: int,
    exact_auc: float,
    validation_acc: float,
    test_acc: float,
) -> tuple[Path, Path]:
    """Save one perturbed embedding and evaluate sampled/exact AUC on it."""

    output_root = Path(os.environ["GRASP_SAME_EMBEDDING_DIAGNOSTIC_DIR"])
    pair_samples = int(
        os.environ.get("GRASP_SAME_EMBEDDING_PAIR_SAMPLES", "200000")
    )
    release_count = int(
        os.environ.get("GRASP_SAME_EMBEDDING_RELEASES", "1")
    )
    if release_count < 1 or release_count > 16:
        raise ValueError("GRASP_SAME_EMBEDDING_RELEASES must be in [1, 16]")
    output_root.mkdir(parents=True, exist_ok=True)

    num_nodes = int(dataset.graph["num_nodes"])
    pair_index, pair_labels = sample_balanced_validation_pairs(
        dataset.graph["edge_index"],
        torch.arange(num_nodes),
        num_nodes,
        pair_samples,
        int(args.seed),
    )

    old_pair_index = attacker.sampled_pair_index
    old_pair_labels = attacker.sampled_pair_labels
    try:
        attacker.sampled_pair_index = pair_index
        attacker.sampled_pair_labels = pair_labels.to(torch.long)
        sampled_scores = attacker._get_sampled_similarity_scores(attacker.H_A)
        sampled_auc = float(
            attacker._get_auc(sampled_scores, attacker.sampled_pair_labels)
        )
    finally:
        attacker.sampled_pair_index = old_pair_index
        attacker.sampled_pair_labels = old_pair_labels

    group = getattr(model, "group", None)
    bundle_path = output_root / f"run_{run}.pt"
    bundle_tmp = bundle_path.with_suffix(".pt.tmp")
    additional_releases: list[torch.Tensor] = []
    if release_count > 1:
        model.eval()
        for _ in range(release_count - 1):
            release, _ = model(attacker.X, attacker.A, get_embedding=True)
            additional_releases.append(release.detach().cpu().to(torch.float16))

    payload = {
        "schema_version": 1,
        "embeddings": attacker.H_A.detach().cpu(),
        "additional_embedding_releases": (
            None
            if not additional_releases
            else torch.stack(additional_releases, dim=0)
        ),
        "edge_index": dataset.graph["edge_index"].detach().cpu(),
        "split_idx": {
            key: value.detach().cpu() for key, value in split_idx.items()
        },
        "sampled_pair_index": pair_index,
        "sampled_pair_labels": pair_labels,
        "group": None if group is None else group.detach().cpu(),
        "metrics": {
            "sampled_auc": sampled_auc,
            "exact_auc": float(exact_auc),
            "validation_acc": float(validation_acc),
            "test_acc": float(test_acc),
        },
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "gnn": args.gnn,
            "seed": int(args.seed),
            "run": int(run),
            "runs": int(args.runs),
            "epochs": int(args.epochs),
            "sigma": float(args.defence_sigma),
            "attack_pair_samples": pair_samples,
            "attack_node_scope": "selector_all",
            "exact_attack_mode": "existing dense all-pairs",
            "same_embedding": True,
            "release_count": release_count,
            "additional_release_dtype": (
                None if not additional_releases else "float16"
            ),
            "test_used_for_selection": False,
            "defense_group_noise": bool(args.defense_group_noise),
            "defense_group_algorithm": (
                args.defense_group_algorithm
                if args.defense_group_noise
                else None
            ),
            "group_bernoulli_noise": bool(args.group_bernoulli_noise),
            "group_shared_probability": float(args.group_shared_probability),
            "static_correlated_noise": bool(args.static_correlated_noise),
            "correlation_rho": float(args.correlation_rho),
            "static_correlation_use_group_file": bool(
                args.static_correlation_use_group_file
            ),
            "correlated_rdp_calibrated": bool(args.correlated_rdp_calibrated),
            "normalization_profile": os.environ.get(
                "GRASP_NORMALIZATION_PROFILE"
            ),
            "defence_limit": bool(args.defence_limit),
        },
    }
    torch.save(payload, bundle_tmp)
    os.replace(bundle_tmp, bundle_path)

    summary_path = output_root / f"run_{run}.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary = {
        "schema_version": 1,
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "sampled_auc": sampled_auc,
        "exact_auc": float(exact_auc),
        "auc_sampled_minus_exact": sampled_auc - float(exact_auc),
        "validation_acc": float(validation_acc),
        "test_acc": float(test_acc),
        "seed": int(args.seed),
        "run": int(run),
        "epochs": int(args.epochs),
        "pair_samples": pair_samples,
        "same_embedding": True,
        "release_count": release_count,
        "test_used_for_selection": False,
    }
    summary_tmp.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_tmp, summary_path)
    return bundle_path, summary_path
