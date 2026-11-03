"""Isolated best-so-far checkpoint capture for trajectory diagnostics.

This module deliberately performs no model forward pass and no metric
evaluation.  It copies the validation-selected checkpoint already maintained
by the training loop at fixed training horizons so that evaluation can happen in a
separate process without perturbing the training RNG stream.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import torch
import numpy as np


DEFAULT_HORIZONS = (500, 1000, 1500, 2500, 3500, 5000)


def _parse_horizons(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_HORIZONS
    horizons = tuple(sorted({int(item.strip()) for item in value.split(',') if item.strip()}))
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError(f'Invalid trajectory horizons: {value!r}')
    return horizons


def trajectory_enabled() -> bool:
    return bool(os.environ.get('GRASP_TRAJECTORY_CHECKPOINT_DIR'))


def trajectory_checkpoint_only() -> bool:
    return os.environ.get('GRASP_TRAJECTORY_CHECKPOINT_ONLY', '') == '1'


def trajectory_horizons() -> tuple[int, ...]:
    return _parse_horizons(os.environ.get('GRASP_TRAJECTORY_EPOCHS'))


def _best_model_path(args, run: int, unique_id: str) -> Path:
    if args.model == 'MPNN':
        name = f'{args.model}_{args.gnn}_{run}_{unique_id}.pt'
    else:
        name = f'{args.model}_{run}_{unique_id}.pt'
    return Path('models') / args.dataset / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture_best_checkpoint(
    *,
    args,
    model,
    optimizer,
    run: int,
    unique_id: str,
    horizon: int,
    best_val: float,
    split_idx: dict[str, torch.Tensor],
    group: torch.Tensor | None,
) -> tuple[Path, Path] | None:
    """Save exact-epoch and validation-best-so-far states at one horizon."""
    if not trajectory_enabled() or horizon not in trajectory_horizons():
        return None

    source_path = _best_model_path(args, run, unique_id)
    if not source_path.exists():
        raise FileNotFoundError(
            f'No validation-best checkpoint exists at horizon {horizon}: {source_path}'
        )

    root = Path(os.environ['GRASP_TRAJECTORY_CHECKPOINT_DIR']).resolve()
    original = torch.load(source_path, map_location='cpu', weights_only=False)
    common = {
        'schema_version': 1,
        'horizon_epoch': int(horizon),
        'run': int(run),
        'seed': int(args.seed),
        'best_validation_acc': float(best_val),
        'group': None if group is None else group.detach().cpu().to(torch.long),
        'split_idx': {key: value.detach().cpu() for key, value in split_idx.items()},
        'args': dict(vars(args)),
        'training_process_evaluated_attack': False,
        'test_used_for_selection': False,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    exact_payload = dict(common)
    exact_payload.update({
        'state_semantics': 'exact_epoch_post_update',
        'model_state_dict': {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in model.state_dict().items()
        },
        'optimizer_state_dict': optimizer.state_dict(),
    })
    best_payload = dict(common)
    best_payload.update({
        'state_semantics': 'validation_best_so_far',
        'model_state_dict': {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in original['model_state_dict'].items()
        },
        'optimizer_state_dict': original.get('optimizer_state_dict'),
    })

    written = []
    for state_name, payload in (
        ('exact_epoch', exact_payload),
        ('validation_best_so_far', best_payload),
    ):
        destination = root / state_name / f'run_{run}' / f'epoch_{horizon}.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite trajectory checkpoint: {destination}')
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        digest = _sha256(destination)
        destination.with_suffix('.pt.sha256').write_text(
            f'{digest}  {destination.name}\n', encoding='utf-8'
        )
        destination.with_suffix('.json').write_text(
            json.dumps(
                {
                    'schema_version': 1,
                    'run': int(run),
                    'horizon_epoch': int(horizon),
                    'state_semantics': payload['state_semantics'],
                    'best_validation_acc': float(best_val),
                    'sha256': digest,
                    'training_process_evaluated_attack': False,
                    'test_used_for_selection': False,
                },
                indent=2,
                sort_keys=True,
            ) + '\n',
            encoding='utf-8',
        )
        print(f'Trajectory checkpoint saved: {destination}')
        written.append(destination)
    return written[0], written[1]
