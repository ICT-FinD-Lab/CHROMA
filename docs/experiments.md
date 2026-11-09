# Experiment settings

The main comparison evaluates USP, GRASP, and CHROMA with GCN and GAT.
All methods use the dataset/backbone configuration in `configs/main_table.json`.
Run the commands below from the repository root directory.

## Data and training

| Dataset | Epochs | Split |
| --- | ---: | --- |
| Cora / CiteSeer | 3,000 | Per-seed random 60/20/20 |
| PubMed | 5,000 | Per-seed random 60/20/20 |
| Computer | 5,000 | Included fixed split |
| Photo | 8,000 | Included fixed split |
| CS | 7,000 | Included fixed split |
| Physics | 8,000 | Included fixed split |
| WikiCS | 2,000 | First provided split; validation and stopping masks combined |
| ogbn-arxiv | 5,000 | Per-seed random 60/20/20 |

The ogbn-arxiv experiment uses the random split above rather than the official
OGB chronological split. Amazon and Coauthor features are normalized; Planetoid
features are used without that transform. Residual connections and model
normalization are configured separately for each dataset and backbone.

The first checkpoint attaining the highest validation accuracy is selected.
Classification accuracy is measured on the corresponding test split. Each run
records its actual split and all epoch measurements. Included fixed split
files are copied into the data directory only when absent; conflicting splits
produce an error.

## Perturbation and grouping

`sigma` is the Gaussian standard deviation. GRASP uses Bernoulli probability
0.7. USP uses independent noise. CHROMA samples one Gaussian vector per group,
with independent draws across groups. Noise is applied to the final-layer
representation during both training and evaluation. At zero noise, all methods
use the common `CLEAN` setting.

The CHROMA configuration in `configs/chroma.json` is shared across datasets,
backbones, seeds, and noise levels:

| Parameter | Value |
| --- | ---: |
| Edge-conflict cap `tau_E` | 0.08 |
| Local conflict cap `tau_Q` | 0.35 |
| Nominal group-size cap `tau_M` | 0.60 |
| Initial scaffold cap | 0.70 |
| Candidate budget | 128 |
| Local score weight | 0.5 |
| Knee risk weight | 0.5 |

Independent-set coloring supplies the initial groups. Oversized groups are
split deterministically. Constrained merging produces a sequence of partitions,
and the structural knee selects a partition from that sequence. The effective
group-size cap is `min(0.95, max(tau_M, initial_max_group_ratio + 0.02))`.
Partition construction uses graph structure and is independent of the training
seed, noise level, and attack results.

The eight smaller datasets' final partitions are included. For ogbn-arxiv,
`prepare_groups.py` starts from the included initial partition and completes
merging and selection. `--rebuild-coloring` recomputes its initial coloring.
To reconstruct a partition without replacing a supplied file:

```bash
python prepare_groups.py --dataset cora --output rebuilt_groups
```

Use `--groups rebuilt_groups` when training with that partition. The input
`<dataset>.npy` must be a one-dimensional integer array with one entry per
node in dataset order. Group IDs must run consecutively from zero.

## Evaluation

The reconstruction score is the rectified cosine similarity of row-centered
released embeddings. Evaluation uses all ordered node pairs, including the
zero diagonal negatives. The eight-dataset evaluation uses the dense score
matrix; ogbn-arxiv uses exact blockwise evaluation with no pair sampling or
score quantization. Large graphs require additional memory for attack evaluation.

Each run stores raw AUC and `AUC_sep = max(AUC, 1 - AUC)`. Aggregation applies
this transformation per seed before averaging. Per-seed files use fractions;
summaries report percentages and population standard deviations. A cell is
marked complete when all five seeds, 0–4, share one experiment configuration.
The launcher records a `protocol_sha256` for the training settings, source
code, evaluation backend and input partition. Seed numbers and machine-specific
paths are excluded. Keep this field and the `smoke` column when exporting runs
to CSV. Earlier runs can be summarized from their original directories with
`launch.json`. The two archived reference CSVs are recognized by their file
checksums. A standalone record without configuration information can be read
individually, but cannot be combined with other seeds.

The entry point runs the main-table correlation attack. Reference control
measurements are provided separately in `results/controls_per_seed.csv`.
