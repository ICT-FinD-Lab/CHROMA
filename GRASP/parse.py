from model import MPNNs


def parse_method(args, n, c, d, device, group=None):
    if args.model != 'MPNN':
        raise ValueError(f'Unsupported model: {args.model}')
    return MPNNs(
        d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
        heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
        res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn,
        defence_sigma=args.defence_sigma, defence_k=args.defence_k,
        defence_p=args.defence_p, defence_limit=args.defence_limit, group=group,
        defence_stage_train=args.defence_stage_train, defence_stage_test=args.defence_stage_test,
        group_grasp_mix=args.group_grasp_mix, group_grasp_probability=args.group_grasp_probability,
        group_bernoulli_noise=args.group_bernoulli_noise, group_shared_probability=args.group_shared_probability,
        static_correlated_noise=args.static_correlated_noise, correlation_rho=args.correlation_rho,
        correlated_rdp_calibrated=args.correlated_rdp_calibrated, noise_node_fraction=args.noise_node_fraction,
        noise_node_energy_match=args.noise_node_energy_match, layer_budget_schedule=args.layer_budget_schedule,
    ).to(device)


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')

    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')

    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', choices=['MPNN'], default='MPNN')
    # GNN
    parser.add_argument('--gnn', choices=['gcn', 'gat', 'sage'], default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument(
        '--selector_export_path',
        type=str,
        default='',
        help=(
            'optional path for exporting the clean best-validation embedding '
            'bundle for comparing candidate partitions'
        ),
    )
    parser.add_argument(
        '--attack_pair_samples',
        type=int,
        default=0,
        help=(
            'total balanced positive/negative pairs for approximate low-memory '
            'attack AUC; 0 keeps the original exact dense evaluation'
        ),
    )
    parser.add_argument(
        '--attack_node_scope',
        type=str,
        default='all',
        choices=['all', 'valid', 'selector_all'],
        help=(
            'node scope used by sampled attack pairs; valid and selector_all '
            'use deterministic selector pairs, while all preserves the '
            'original approximate attack sampler'
        ),
    )
    parser.add_argument(
        '--selector_trial_output',
        type=str,
        default='',
        help=(
            'optional JSON output for a validation-only short-training trial; '
            'when set, test accuracy is not evaluated or written'
        ),
    )
    parser.add_argument(
        '--selector_robust_attacks',
        action='store_true',
        help=(
            'for selector trials, also evaluate the public-group demeaned '
            'attack when applicable and the whitening attack'
        ),
    )

    # defence
    parser.add_argument('--defence_auc', type=float, default=1.0)
    parser.add_argument('--defence_auc_initial', type=float, default=1.0)
    parser.add_argument('--defence_relu', action='store_true')
    parser.add_argument('--defence_l1', action='store_true')
    parser.add_argument('--defence_l2', action='store_true')
    parser.add_argument('--defence_positive', action='store_true')
    parser.add_argument('--defence_all', action='store_true')

    parser.add_argument('--defence_sigma', type=float, default=0.0)
    parser.add_argument('--defence_k', type=float, default=1.5)
    parser.add_argument('--defence_limit', action='store_true')
    parser.add_argument('--defence_visual', action='store_true')
    parser.add_argument('--defence_visual_mode', type=int, default=1)
    parser.add_argument('--defence_p', type=float, default=0.707)
    parser.add_argument(
        '--noise_node_fraction',
        type=float,
        default=1.0,
        help=(
            'fraction of nodes receiving additive noise at each injection; '
            '1.0 preserves the released GRASP behavior'
        ),
    )
    parser.add_argument(
        '--noise_node_energy_match',
        action='store_true',
        help=(
            'scale selected-node noise by 1/sqrt(noise_node_fraction) so the '
            'expected total squared noise matches the all-node mechanism'
        ),
    )
    parser.add_argument('--defence_stage_train', action='store_true')
    parser.add_argument('--defence_stage_test', action='store_true')

    parser.add_argument('--defense_group_noise', action='store_true')
    parser.add_argument('--defense_group_algorithm', type=str, default='saturation_largest_first')
    parser.add_argument(
        '--group_grasp_mix',
        type=float,
        default=0.0,
        help=(
            'variance-preserving mixture weight between coloring noise and '
            'the original GRASP shared-mask noise; 0 is pure coloring and 1 '
            'is pure GRASP'
        ),
    )
    parser.add_argument(
        '--group_grasp_probability',
        type=float,
        default=0.7,
        help='shared-mask probability used by the GRASP component of mixed group noise',
    )
    parser.add_argument(
        '--group_bernoulli_noise',
        action='store_true',
        help=(
            'for a supplied node group, let each node independently choose '
            'group-shared or node-independent Gaussian noise via Bernoulli(p)'
        ),
    )
    parser.add_argument(
        '--group_shared_probability',
        type=float,
        default=0.7,
        help='Bernoulli probability p for group-shared noise',
    )
    parser.add_argument(
        '--group_dir',
        type=str,
        default='group',
        help='root directory containing per-dataset graph-coloring tensors',
    )
    parser.add_argument(
        '--static_correlated_noise',
        action='store_true',
        help='use a data-independent random partition and correlated Gaussian noise',
    )
    parser.add_argument(
        '--static_correlation_use_group_file',
        action='store_true',
        help=(
            'use --group_dir/--defense_group_algorithm for static correlated '
            'noise instead of constructing a random partition'
        ),
    )
    parser.add_argument(
        '--correlation_rho',
        type=float,
        default=0.0,
        help='within-group Gaussian correlation, constrained to [0, 1)',
    )
    parser.add_argument(
        '--correlation_group_size',
        type=int,
        default=1,
        help='maximum size of each data-independent random group',
    )
    parser.add_argument(
        '--correlated_rdp_calibrated',
        action='store_true',
        help='scale marginal sigma by 1/sqrt(1-rho) to match the isotropic worst-case RDP bound',
    )
    parser.add_argument(
        '--layer_budget_schedule',
        type=str,
        default='legacy',
        choices=[
            'legacy',
            'last_only',
            'uniform',
            'front_loaded',
            'back_loaded',
        ],
        help=(
            'allocate a fixed unit-sensitivity RDP budget across GNN layers; '
            'non-legacy schedules satisfy sum_l 1/sigma_l^2 = 1/defence_sigma^2'
        ),
    )
