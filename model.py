import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch.nn as nn


def make_random_groups(num_nodes, group_size, seed=0, device=None):
    """Build a data-independent random partition with bounded group size."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_nodes, generator=generator)
    groups = torch.empty(num_nodes, dtype=torch.long)
    groups[permutation] = torch.arange(num_nodes) // int(group_size)
    return groups.to(device=device)


def center_by_group(x, group):
    """Remove the node-wise mean inside each public random group."""
    if group is None:
        raise ValueError("group is required for group centering")
    if group.ndim != 1 or group.numel() != x.size(0):
        raise ValueError("group must contain one group id per node")

    group = group.to(device=x.device, dtype=torch.long)
    num_groups = int(group.max().item()) + 1
    sums = torch.zeros(
        (num_groups, x.size(1)),
        dtype=x.dtype,
        device=x.device,
    )
    sums.index_add_(0, group, x)
    counts = torch.bincount(group, minlength=num_groups).to(
        device=x.device,
        dtype=x.dtype,
    )
    means = sums / counts.clamp_min(1).unsqueeze(1)
    return x - means[group]


def build_layer_budget_weights(num_layers, schedule):
    """Return per-layer RDP spending fractions whose sum is one."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if schedule == "last_only":
        weights = torch.zeros(num_layers, dtype=torch.float64)
        weights[-1] = 1.0
    elif schedule == "uniform":
        weights = torch.ones(num_layers, dtype=torch.float64)
    elif schedule == "front_loaded":
        weights = torch.arange(
            num_layers,
            0,
            -1,
            dtype=torch.float64,
        )
    elif schedule == "back_loaded":
        weights = torch.arange(
            1,
            num_layers + 1,
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"unsupported layer budget schedule: {schedule}")
    return weights / weights.sum()


def get_noise(
    x,
    sigma=0.0,
    k=0,
    p=0.707,
    group=None,
    group_grasp_mix=0.0,
    group_grasp_probability=0.7,
    group_bernoulli_noise=False,
    group_shared_probability=0.7,
    correlation_rho=0.0,
    static_correlated=False,
    rdp_calibrated=False,
    noise_node_fraction=1.0,
    noise_node_energy_match=False,
):
    n, d = x.size()
    noise_node_fraction = float(noise_node_fraction)
    if not 0.0 < noise_node_fraction <= 1.0:
        raise ValueError("noise_node_fraction must satisfy 0 < fraction <= 1")

    if static_correlated:
        if group is None:
            raise ValueError("static correlated noise requires a group partition")
        if not 0.0 <= correlation_rho < 1.0:
            raise ValueError("correlation_rho must satisfy 0 <= rho < 1")

        # For C=(1-rho)I+rho*BB^T, lambda_min(C)=1-rho.
        # Scaling the marginal sigma by 1/sqrt(1-rho) keeps the conservative
        # worst-case Gaussian RDP bound equal to the isotropic baseline whose
        # standard deviation is supplied as `sigma`.
        marginal_sigma = float(sigma)
        if rdp_calibrated:
            marginal_sigma /= (1.0 - correlation_rho) ** 0.5

        group = group.to(device=x.device, dtype=torch.long)
        num_groups = int(group.max().item()) + 1
        independent = torch.normal(
            0.0,
            marginal_sigma,
            size=x.size(),
            device=x.device,
        )
        shared = torch.normal(
            0.0,
            marginal_sigma,
            size=(num_groups, d),
            device=x.device,
        )
        noise = (
            (1.0 - correlation_rho) ** 0.5 * independent
            + correlation_rho ** 0.5 * shared[group]
        )

    elif group is not None:
        num_colors = int(group.max().item()) + 1
        color_noise = torch.normal(
            0.0, sigma, size=(num_colors, d), device=x.device
        )[group]
        if group_bernoulli_noise:
            if not 0.0 <= group_shared_probability <= 1.0:
                raise ValueError(
                    "group_shared_probability must satisfy 0 <= p <= 1"
                )
            independent = torch.normal(
                0.0, sigma, size=x.size(), device=x.device
            )
            shared_mask = (
                torch.rand((n, 1), device=x.device)
                < group_shared_probability
            )
            noise = torch.where(shared_mask, color_noise, independent)
        elif group_grasp_mix > 0.0:
            grasp_noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)
            identical_mask = (
                torch.rand(n, device=x.device) < group_grasp_probability
            )
            grasp_noise[identical_mask] = torch.normal(
                0.0, sigma, size=(1, d), device=x.device
            )
            noise = (
                (1.0 - group_grasp_mix) ** 0.5 * color_noise
                + group_grasp_mix ** 0.5 * grasp_noise
            )
        else:
            noise = color_noise

    else:
        if k == 0:
            noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)

        elif k == 1.5:
            noise = torch.normal(0.0, sigma, size=x.size(), device=x.device)
            identical_mask = torch.rand(n, device=x.device) < p
            noise[identical_mask] = torch.normal(0.0, sigma, size=(1, d), device=x.device)
            # noise[torch.rand(n) < p**0.5] = torch.normal(0.0, sigma, size=(1, d), device=x.device)

        elif k > 0:
            noise = torch.normal(0.0, sigma, size=(k, d), device=x.device)
            noise = noise[torch.randint(0, k, (n,), device=x.device)]


    if noise_node_fraction < 1.0:
        selected_count = max(1, min(n, int(round(n * noise_node_fraction))))
        selected_nodes = torch.randperm(n, device=x.device)[:selected_count]
        node_mask = torch.zeros((n, 1), dtype=noise.dtype, device=x.device)
        node_mask[selected_nodes] = 1.0
        noise = noise * node_mask
        if noise_node_energy_match:
            noise = noise / (noise_node_fraction ** 0.5)

    return noise


class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
                 dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn',
                 defence_sigma=0.0, defence_k=1, defence_p=0.7, defence_limit=False, group=None,
                 defence_stage_train=True, defence_stage_test=True,
                 group_grasp_mix=0.0, group_grasp_probability=0.7,
                 group_bernoulli_noise=False,
                 group_shared_probability=0.7,
                 static_correlated_noise=False, correlation_rho=0.0,
                 correlated_rdp_calibrated=False,
                 noise_node_fraction=1.0, noise_node_energy_match=False,
                 layer_budget_schedule="legacy"):
        super(MPNNs, self).__init__()

        self.dropout = dropout
        self.pre_ln = pre_ln

        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.defence_sigma = defence_sigma
        self.defence_k = defence_k
        self.defence_p = defence_p
        self.defence_limit = defence_limit

        self.group = group
        self.group_grasp_mix = group_grasp_mix
        self.group_grasp_probability = group_grasp_probability
        self.group_bernoulli_noise = bool(group_bernoulli_noise)
        self.group_shared_probability = float(group_shared_probability)
        self.static_correlated_noise = static_correlated_noise
        self.correlation_rho = correlation_rho
        self.correlated_rdp_calibrated = correlated_rdp_calibrated
        self.noise_node_fraction = float(noise_node_fraction)
        self.noise_node_energy_match = bool(noise_node_energy_match)
        self.layer_budget_schedule = layer_budget_schedule

        self.defence_stage_train = defence_stage_train
        self.defence_stage_test = defence_stage_test

        self.attack = False

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)

        if not self.pre_linear:
            if gnn=='gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

        if self.layer_budget_schedule == "legacy":
            if self.defence_limit:
                self.layer_budget_weights = torch.zeros(
                    len(self.local_convs),
                    dtype=torch.float64,
                )
                self.layer_budget_weights[-1] = 1.0
            else:
                self.layer_budget_weights = None
        else:
            self.layer_budget_weights = build_layer_budget_weights(
                len(self.local_convs),
                self.layer_budget_schedule,
            )


    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def get_layer_noise_sigmas(self):
        if self.layer_budget_schedule == "legacy":
            if self.defence_limit:
                return [
                    0.0
                    if index < len(self.local_convs) - 1
                    else float(self.defence_sigma)
                    for index in range(len(self.local_convs))
                ]
            return [float(self.defence_sigma)] * len(self.local_convs)

        sigmas = []
        for weight in self.layer_budget_weights.tolist():
            if weight == 0.0:
                sigmas.append(0.0)
            else:
                sigmas.append(float(self.defence_sigma) / weight ** 0.5)
        return sigmas


    def forward(self, x, edge_index, get_embedding=False, dense_adj=False):
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        if dense_adj:
            for i, local_conv in enumerate(self.local_convs):
                if self.res:
                    x = edge_index @ local_conv.lin(x) + local_conv.bias + self.lins[i](x)
                else:
                    x = edge_index @ local_conv.lin(x) + local_conv.bias

                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)
                else:
                    pass

                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                layer_sigma = self.get_layer_noise_sigmas()[i]
                if layer_sigma > 0.0:
                    if (self.defence_stage_train and self.training) or (self.defence_stage_test and not self.training):
                        x = x + get_noise(
                            x,
                            sigma=layer_sigma,
                            k=self.defence_k,
                            p=self.defence_p,
                            group=self.group,
                            group_grasp_mix=self.group_grasp_mix,
                            group_grasp_probability=self.group_grasp_probability,
                            group_bernoulli_noise=self.group_bernoulli_noise,
                            group_shared_probability=self.group_shared_probability,
                            correlation_rho=self.correlation_rho,
                            static_correlated=self.static_correlated_noise,
                            rdp_calibrated=self.correlated_rdp_calibrated,
                            noise_node_fraction=self.noise_node_fraction,
                            noise_node_energy_match=self.noise_node_energy_match,
                        )

                if self.jk:
                    x_final = x_final + x
                else:
                    x_final = x

        else:
            for i, local_conv in enumerate(self.local_convs):
                if self.res:
                    x = local_conv(x, edge_index) + self.lins[i](x)
                else:
                    x = local_conv(x, edge_index)

                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)
                else:
                    pass

                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)


                layer_sigma = self.get_layer_noise_sigmas()[i]
                if layer_sigma > 0.0:
                    if (self.defence_stage_train and self.training) or (self.defence_stage_test and not self.training):
                        x = x + get_noise(
                            x,
                            sigma=layer_sigma,
                            k=self.defence_k,
                            p=self.defence_p,
                            group=self.group,
                            group_grasp_mix=self.group_grasp_mix,
                            group_grasp_probability=self.group_grasp_probability,
                            group_bernoulli_noise=self.group_bernoulli_noise,
                            group_shared_probability=self.group_shared_probability,
                            correlation_rho=self.correlation_rho,
                            static_correlated=self.static_correlated_noise,
                            rdp_calibrated=self.correlated_rdp_calibrated,
                            noise_node_fraction=self.noise_node_fraction,
                            noise_node_energy_match=self.noise_node_energy_match,
                        )

                if self.jk:
                    x_final = x_final + x
                else:
                    x_final = x


        x = self.pred_local(x_final)

        if get_embedding:
            return x_final, x
        else:
            return x
