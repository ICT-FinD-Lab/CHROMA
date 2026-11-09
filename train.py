import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, coalesce

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args
from diagnostics.selection import export_selector_bundle, validate_group_partition, write_selection_json

from torch_geometric.utils import to_dense_adj
from GraphAttacker import GraphAttacker
from model import make_random_groups
from diagnostics.embeddings import export_same_embedding_diagnostic
from diagnostics.checkpoints import (
    capture_best_checkpoint,
    trajectory_checkpoint_only,
)
import networkx as nx

import os
import csv

import uuid
unique_id = str(uuid.uuid4())


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_A_H_A(H, mode=1):
    if mode == 1:
        H = H - H.mean(dim=1, keepdim=True)
        H = F.normalize(H, p=2, dim=1)
        H = torch.mm(H, H.t())
        H.fill_diagonal_(0.)
        return H

    elif mode == 2:
        H_center = H.mean(dim=0, keepdim=True).detach()
        H = H - H.mean(dim=1, keepdim=True)
        H = F.normalize(H, p=2, dim=1)
        H_center = H_center - H_center.mean(dim=1, keepdim=True)
        H_center = F.normalize(H_center, p=2, dim=1)
        H = torch.mm(H, H_center.t())
        return H


def append_run_summary(args, correlation_aucs, secondary_aucs, accuracies):
    """Append measured metrics to the training CSV, keeping its column names."""
    def metric_fields(values, scale=1):
        if len(values) == 0:
            return '', '', ''
        values = torch.as_tensor(values)
        mean = f'{values.mean().item() * scale:.2f}'
        if values.numel() == 1:
            return mean, '', mean
        std = f'{values.std().item() * scale:.2f}'
        return mean, std, f'{mean} ± {std}'

    first = metric_fields(correlation_aucs, 100)
    second = metric_fields(secondary_aucs, 100)
    acc = metric_fields(accuracies)
    fields = ('gnn', 'dataset', 'defence_auc', 'defence_sigma', 'defence_k', 'defence_p', 'defence_limit')
    row = {field: getattr(args, field) for field in fields}
    row.update(attack_1_auc_mean=first[0], attack_2_auc_mean=second[0], results_mean=acc[0],
               attack_1_auc_std=first[1], attack_2_auc_std=second[1], results_std=acc[1],
               attack_1_auc=first[2], attack_2_auc=second[2], results=acc[2])
    fields = ('defence_stage_train', 'defence_stage_test', 'defense_group_noise', 'defense_group_algorithm',
              'attack_pair_samples', 'group_grasp_mix', 'group_grasp_probability', 'noise_node_fraction',
              'noise_node_energy_match', 'group_bernoulli_noise', 'group_shared_probability',
              'static_correlated_noise', 'correlation_rho', 'static_correlation_use_group_file',
              'correlated_rdp_calibrated')
    row.update((field, getattr(args, field)) for field in fields)
    os.makedirs('results_csv', exist_ok=True)
    with open(f'results_csv/{args.dataset}.csv', 'a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def graph_coloring(edge_index, num_nodes, algorithm="saturation_largest_first"):
    # Remove self-loops and coalesce duplicate edges
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    # Build a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    # Normalize algorithm name
    algorithm = algorithm.lower()

    # Supported strategies and their aliases
    supported_strategies = {
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "connected_sequential",  # alias
        "saturation_largest_first",
        "dsatur"  # alias
    }

    # Map aliases to actual strategies
    alias_map = {
        "connected_sequential": "connected_sequential_bfs",
        "dsatur": "saturation_largest_first"
    }

    # Resolve the alias if needed
    strategy = alias_map.get(algorithm, algorithm)

    # Check if the strategy is supported
    if strategy not in supported_strategies:
        raise ValueError(f"Unsupported algorithm: {algorithm}. "
                         f"Supported algorithms are: {sorted(supported_strategies)}")

    # Perform graph coloring
    coloring = nx.coloring.greedy_color(G, strategy=strategy)

    # Convert coloring dict to a tensor
    color_indices = torch.tensor(
        [coloring.get(i, -1) for i in range(num_nodes)],
        dtype=torch.long
    )

    # Check if any node was not colored
    if (color_indices == -1).any():
        raise ValueError("Some nodes were not colored. Please check the input graph structure.")

    return color_indices


def maximum_independent_set_from_edge_index(edge_index, num_nodes):
    print('Calculating maximum independent set...')
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    min_vertex_cover = nx.algorithms.approximation.min_weighted_vertex_cover(G, weight=1)
    max_indep_set = set(G.nodes) - min_vertex_cover
    max_indep_set = torch.tensor(list(max_indep_set), dtype=torch.long)
    print(f'Done, size: {len(max_indep_set)} / {num_nodes}')


    return max_indep_set


### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(f'Training {args.gnn.upper()} on {args.dataset} (seed {args.seed}, sigma {args.defence_sigma:g})')
if args.selector_trial_output and not args.save_model:
    parser.error('--selector_trial_output requires --save_model')


if args.selector_export_path:
    if not args.save_model:
        parser.error('--selector_export_path requires --save_model')
    if float(args.defence_sigma) != 0.0:
        parser.error('--selector_export_path requires --defence_sigma 0')
    if args.defense_group_noise or args.static_correlated_noise:
        parser.error(
            '--selector_export_path requires clean ungrouped inference'
        )

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

dataset.label = dataset.label.to(device)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)


node_label = dataset.label.squeeze(1)
dataset.graph['node_label'] = node_label
dataset.graph['num_classes'] = c


auc_device_override = os.environ.get('GRASP_AUC_DEVICE', '').strip().lower()
if auc_device_override:
    if auc_device_override != 'cpu':
        raise ValueError(
            "GRASP_AUC_DEVICE currently supports only the exact CPU backend"
        )
    device_auc = torch.device('cpu')
    print('Attack AUC device: CPU (exact scores and labels; no sampling)')
else:
    device_auc = device

A_X_vec = None

if trajectory_checkpoint_only():
    A_vec = None
    print('Trajectory checkpoint-only mode: attack adjacency is not materialized.')
elif args.attack_pair_samples > 0:
    A_vec = None
    print(
        f'Attack AUC mode: balanced sampled pairs '
        f'(requested total={args.attack_pair_samples})'
    )
else:
    A_vec = None
    print('Attack AUC mode: exact all-pairs')

A_Y_vec = None

max_indep_set = None


if args.static_correlated_noise and args.defense_group_noise:
    raise ValueError(
        "--static_correlated_noise and --defense_group_noise are mutually exclusive"
    )
if args.static_correlation_use_group_file and not args.static_correlated_noise:
    raise ValueError(
        "--static_correlation_use_group_file requires --static_correlated_noise"
    )

if not 0.0 <= args.group_grasp_mix <= 1.0:
    raise ValueError("--group_grasp_mix must satisfy 0 <= mix <= 1")
if not 0.0 <= args.group_grasp_probability <= 1.0:
    raise ValueError(
        "--group_grasp_probability must satisfy 0 <= probability <= 1"
    )
if args.group_grasp_mix > 0.0 and not args.defense_group_noise:
    raise ValueError("--group_grasp_mix requires --defense_group_noise")
if not 0.0 <= args.group_shared_probability <= 1.0:
    raise ValueError("--group_shared_probability must satisfy 0 <= p <= 1")
if args.group_bernoulli_noise and not args.defense_group_noise:
    raise ValueError("--group_bernoulli_noise requires --defense_group_noise")
if args.group_bernoulli_noise and args.group_grasp_mix > 0.0:
    raise ValueError(
        "--group_bernoulli_noise and --group_grasp_mix are mutually exclusive"
    )

if args.static_correlated_noise:
    if not 0.0 <= args.correlation_rho < 1.0:
        raise ValueError("--correlation_rho must satisfy 0 <= rho < 1")
    if not args.static_correlation_use_group_file and args.correlation_group_size <= 0:
        raise ValueError("--correlation_group_size must be positive")
    if args.static_correlation_use_group_file:
        save_path = os.path.join(
            args.group_dir,
            args.dataset,
            f"{args.dataset}_{args.defense_group_algorithm}.pt",
        )
        if not os.path.isfile(save_path):
            raise FileNotFoundError(
                f"Static correlation group file not found: {save_path}"
            )
        graph_coloring_set = torch.load(save_path, map_location='cpu')
        if isinstance(graph_coloring_set, dict):
            graph_coloring_set = graph_coloring_set.get('group')
        graph_coloring_set = validate_group_partition(graph_coloring_set, n).to(device)
        group_source = save_path
    else:
        graph_coloring_set = make_random_groups(
            n,
            args.correlation_group_size,
            seed=args.seed + 104729,
            device=device,
        )
        group_source = 'data-independent-random-partition'
    applied_sigma = args.defence_sigma
    if args.correlated_rdp_calibrated:
        applied_sigma /= (1.0 - args.correlation_rho) ** 0.5
    print(
        "Static correlated Gaussian: "
        f"base_sigma={args.defence_sigma:.6f}, "
        f"rho={args.correlation_rho:.6f}, "
        f"group_source={group_source}, "
        f"groups={int(graph_coloring_set.max().item()) + 1}, "
        f"rdp_calibrated={args.correlated_rdp_calibrated}, "
        f"applied_marginal_sigma={applied_sigma:.6f}"
    )

elif args.defense_group_noise:
    # Ensure the save directory exists
    save_dir = os.path.join(args.group_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Specify the save path including the algorithm name
    save_path = os.path.join(
        save_dir,
        f"{args.dataset}_{args.defense_group_algorithm}.pt",
    )

    if os.path.exists(save_path):
        try:
            graph_coloring_set = torch.load(save_path)
            print(f"Loaded existing coloring from {save_path}")
        except Exception as e:
            print(f"Failed to load coloring from {save_path}: {e}. Recomputing...")
            graph_coloring_set = graph_coloring(dataset.graph['edge_index'], n, algorithm=args.defense_group_algorithm)
            torch.save(graph_coloring_set, save_path)
            print(f"Saved new coloring to {save_path}")
    else:
        graph_coloring_set = graph_coloring(dataset.graph['edge_index'], n, algorithm=args.defense_group_algorithm)
        torch.save(graph_coloring_set, save_path)
        print(f"Saved new coloring to {save_path}")

    graph_coloring_set = validate_group_partition(graph_coloring_set, n).to(device)

else:
    graph_coloring_set = None


### Load method ###
model = parse_method(args, n, c, d, device, graph_coloring_set)
model.train()
print('MODEL:', model)
if args.layer_budget_schedule != 'legacy':
    print(
        f'Layer budget schedule: {args.layer_budget_schedule}; '
        f'weights={model.layer_budget_weights.tolist()}; '
        f'sigmas={model.get_layer_noise_sigmas()}; '
        f'precision_sum={sum(1.0 / sigma**2 for sigma in model.get_layer_noise_sigmas() if sigma > 0.0):.8f}'
    )

correlation_aucs = []
secondary_aucs = []
selection_validation_acc_all = []
selection_centered_auc_all = []
selection_whitened_auc_all = []


### Training loop ###
for run in range(args.runs):
    if args.dataset in (
        'coauthor-cs',
        'coauthor-physics',
        'amazon-computer',
        'amazon-photo',
        'cora',
        'citeseer',
        'pubmed',
        'ogbn-arxiv',
        'ogbn-products',
    ):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')



    display_step = 1000
    initial_finished = True

    for epoch in range(args.epochs):

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])


        if args.defence_auc < 1.0:
            attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec,
                                    lr=0.01, weight_decay=0.00000, epochs=5000,
                                    device=device, device_auc=device_auc, max_indep_set=max_indep_set)
            attack_auc = attacker.embedding_auc()
        else:
            attack_auc = args.defence_auc

        if (attack_auc > args.defence_auc) or ((not initial_finished) and (attack_auc > args.defence_auc_initial)):
            model.train()

            for _ in range(2):

                H_A, _ = model(dataset.graph['node_feat'], dataset.graph['edge_index'], get_embedding=True)
                A_H_A = get_A_H_A(H_A, mode=1)

                if args.defence_relu:
                    A_H_A = torch.relu(A_H_A)
                if args.defence_l1:
                    A_H_A = A_H_A.abs()
                if args.defence_l2:
                    A_H_A = A_H_A**2

                A_H_A_positive = A_H_A[dataset.graph['edge_index'][0], dataset.graph['edge_index'][1]]
                mean_positive = A_H_A_positive.mean()
                mean_all = A_H_A.mean()

                if args.defence_positive and not args.defence_all:
                    loss = 1 + mean_positive
                if not args.defence_positive and args.defence_all:
                    loss = 1 - mean_all
                if args.defence_positive and args.defence_all:
                    loss = 2 + mean_positive - mean_all
                if not args.defence_positive and not args.defence_all:
                    raise ValueError('Invalid defence mode')


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            if not initial_finished:
                initial_finished = True

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]


                if args.defence_visual:
                    save_model(args, model, optimizer, run, id=unique_id)

                if args.save_model:
                    save_model(args, model, optimizer, run, id=unique_id)

            model.train()
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))

            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % display_step == 0:
            print(f'Epoch: {epoch:04d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * result[0]:6.2f}%, '
                f'Val: {100 * result[1]:6.2f}%, '
                f'Test: {100 * result[2]:6.2f}%, '
                f'B-Val: {100 * best_val:6.2f}%, '
                f'B-Test: {100 * best_test:6.2f}%')


        torch.cuda.empty_cache()

        capture_best_checkpoint(
            args=args,
            model=model,
            optimizer=optimizer,
            run=run,
            unique_id=unique_id,
            horizon=epoch + 1,
            best_val=best_val,
            split_idx=split_idx,
            group=getattr(model, 'group', None),
        )

    print('--------------------------------------------------')
    logger.print_statistics(run)
    print('--------------------------------------------------')

    if trajectory_checkpoint_only():
        if args.save_model:
            load_model(args, model, optimizer, run, id=unique_id, delete=True)
        print('Trajectory checkpoint-only run complete; skipping attack evaluation.')
        continue


    if args.save_model:
        model, _ = load_model(args, model, optimizer, run, id=unique_id)

        if args.selector_export_path:
            selector_bundle_path = export_selector_bundle(
                model=model,
                dataset=dataset,
                split_idx=split_idx,
                args=args,
                run=run,
            )
            print(f'Validation embedding bundle: {selector_bundle_path}')

        model.attack = True

        attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec,
                                lr=0.01, weight_decay=0.00000, epochs=5000,
                                device=device, device_auc=device_auc, max_indep_set=max_indep_set)
        attack_auc = attacker.embedding_auc()
        correlation_aucs.append(float(attack_auc))
        print(f'Correlation attack AUC: {attack_auc:.8f}')
        selection_validation_acc_all.append(float(best_val))
        if os.environ.get('GRASP_SAME_EMBEDDING_DIAGNOSTIC_DIR'):
            bundle_path, summary_path = export_same_embedding_diagnostic(
                attacker=attacker,
                model=model,
                dataset=dataset,
                split_idx=split_idx,
                args=args,
                run=run,
                exact_auc=float(attack_auc),
                validation_acc=float(best_val),
                test_acc=float(best_test),
            )
            print(f'Same-embedding diagnostic bundle: {bundle_path}')
            print(f'Same-embedding diagnostic summary: {summary_path}')
        if args.selector_robust_attacks:
            if getattr(model, 'group', None) is not None:
                centered_auc, _, _, _, _, _ = attacker.attack(target=5)
                selection_centered_auc_all.append(float(centered_auc))
                print(f'Group-centered attack AUC: {centered_auc:.8f}')
            whitened_auc, _, _, _, _, _ = attacker.attack(target=6)
            selection_whitened_auc_all.append(float(whitened_auc))
            print(f'Whitened attack AUC: {whitened_auc:.8f}')
        if (
            args.static_correlated_noise
            and args.correlation_rho > 0.0
            and args.correlation_group_size > 1
        ):
            centered_auc, _, _, _, _, _ = attacker.attack(target=5)
            secondary_aucs.append(float(centered_auc))
            print(f'Group-centered attack AUC: {centered_auc:.8f}')
        elif args.layer_budget_schedule != 'legacy':
            whitened_auc, _, _, _, _, _ = attacker.attack(target=6)
            secondary_aucs.append(float(whitened_auc))
            print(f'Whitened attack AUC: {whitened_auc:.8f}')

        model.attack = False

        load_model(args, model, optimizer, run, id=unique_id, delete=True)
        del attacker
        torch.cuda.empty_cache()


    if args.defence_visual:
        model, _ = load_model(args, model, optimizer, run, id=unique_id)
        attacker = GraphAttacker(args=args, model=model, graph=dataset.graph, split_idx=split_idx, A_X_vec=A_X_vec, A_Y_vec=A_Y_vec, A_vec=A_vec,
                                lr=0.01, weight_decay=0.00000, epochs=5000,
                                device=device, device_auc=device_auc, max_indep_set=max_indep_set)

        if args.defence_visual_mode == 1:
            _, _, _, _, _, _ = attacker.attack(target=3)
        elif args.defence_visual_mode == 2:
            _, _, _, _, _, _ = attacker.attack(target=4)
        else:
            raise ValueError('Invalid defence visual mode')

        load_model(args, model, optimizer, run, id=unique_id, delete=True)
        del attacker
        torch.cuda.empty_cache()


if trajectory_checkpoint_only():
    print('All trajectory training runs completed without attack evaluation.')
    raise SystemExit(0)

if args.runs > 1:
    print('--------------------------------------------------')
results = logger.print_statistics()
print('--------------------------------------------------')

if len(correlation_aucs) > 1:
    values = torch.tensor(correlation_aucs)
    print(f'Mean correlation attack AUC: {values.mean():.8f} ± {values.std():.8f}')

if args.selector_trial_output:
    raw = torch.tensor(correlation_aucs)
    raw_auc_mean = raw.mean().item()
    raw_auc_std = raw.std(unbiased=False).item()
    centered_auc_mean = torch.tensor(selection_centered_auc_all).mean().item() if selection_centered_auc_all else None
    whitened_auc_mean = torch.tensor(selection_whitened_auc_all).mean().item() if selection_whitened_auc_all else None
    measured = (raw_auc_mean, centered_auc_mean, whitened_auc_mean)
    robust_effective_auc_mean = max(0.5 + abs(auc - 0.5) for auc in measured if auc is not None)
    validation_acc_tensor = torch.tensor(selection_validation_acc_all)
    trial_payload = {
        'method': (
            (
                args.defense_group_algorithm
                if args.group_grasp_mix == 0.0
                else (
                    f'{args.defense_group_algorithm}'
                    f'+GRASP_mix_{args.group_grasp_mix:.2f}'
                )
            )
            if args.defense_group_noise
            else 'GRASP'
        ),
        'dataset': args.dataset,
        'gnn': args.gnn,
        'sigma': float(args.defence_sigma),
        'seed': int(args.seed),
        'runs': int(args.runs),
        'epochs': int(args.epochs),
        'selection_scope': 'validation_nodes_only',
        'attack_node_scope': args.attack_node_scope,
        'attack_pair_samples': int(args.attack_pair_samples),
        'raw_auc_mean': raw_auc_mean,
        'raw_auc_std': raw_auc_std,
        'effective_auc_mean': 0.5 + abs(raw_auc_mean - 0.5),
        'group_demeaned_auc_mean': centered_auc_mean,
        'whitened_auc_mean': whitened_auc_mean,
        'robust_effective_auc_mean': robust_effective_auc_mean,
        'selector_robust_attacks': bool(args.selector_robust_attacks),
        'validation_acc_mean': validation_acc_tensor.mean().item(),
        'validation_acc_std': validation_acc_tensor.std(
            unbiased=False
        ).item(),
        'defense_group_noise': bool(args.defense_group_noise),
        'defense_group_algorithm': (
            args.defense_group_algorithm if args.defense_group_noise else None
        ),
        'defence_limit': bool(args.defence_limit),
        'defence_stage_train': bool(args.defence_stage_train),
        'defence_stage_test': bool(args.defence_stage_test),
        'normalization_profile': os.environ.get('GRASP_NORMALIZATION_PROFILE'),
        'group_grasp_mix': float(args.group_grasp_mix),
        'group_grasp_probability': float(args.group_grasp_probability),
        'group_bernoulli_noise': bool(args.group_bernoulli_noise),
        'group_shared_probability': float(args.group_shared_probability),
        'defence_p': float(args.defence_p),
        'noise_node_fraction': float(args.noise_node_fraction),
        'noise_node_energy_match': bool(args.noise_node_energy_match),
        'warning': (
            'Validation-only trial. Test accuracy was not measured.'
        ),
    }
    write_selection_json(args.selector_trial_output, trial_payload)
    print(f'Validation trial: {args.selector_trial_output}')
else:
    save_result(args, results)
    if not args.defence_visual:
        append_run_summary(args, correlation_aucs, secondary_aucs, results)
