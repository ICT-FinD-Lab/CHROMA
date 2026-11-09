# CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation

[中文说明](docs/README_CN.md)


This repository implements **GRASP** and its extension, **CHROMA**, for defending
against attacks that reconstruct graph edges from the similarity of node embeddings.

GRASP combines independent and identical Gaussian noise through Bernoulli
assignment to disrupt the relative ranking of node similarities. CHROMA uses graph
structure to decide which nodes share a perturbation. It first colors the graph
so that nodes of the same color are nonadjacent. It splits oversized color
classes and merges groups subject to conflict and size limits. It selects a
partition by comparing how often adjacent and nonadjacent pairs end up in the
same group. Nodes in the same group receive the same Gaussian noise vector.
Different groups draw their noise independently.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py), [`train.py`](train.py) | Experiment launcher and training loop |
| [`coloring/`](coloring/), [`model.py`](model.py) | Build CHROMA groups and apply noise to node embeddings |
| [`configs/`](configs/), [`assets/`](assets/) | Experiment settings, partitions, and data splits |
| [`scripts/`](scripts/), [`tests/`](tests/) | Experiment commands, input verification, and tests |
| [`results/`](results/), [`summarize.py`](summarize.py) | Per-seed reference results and summary tool |

See [the code guide](docs/code.md) for module details and the files produced by each run.

## Reproduce the experiments

### Install dependencies

Use Python 3.11. Clone the repository and run all commands from its root.

```bash
git clone https://github.com/ICT-FinD-Lab/CHROMA.git
cd CHROMA
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`.

For a CPU environment, including the quick check and tests:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=torch-scatter,torch-sparse -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

For GPU experiments with CUDA 12.4, use these commands in a fresh environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=torch-scatter,torch-sparse -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

### Quick check

Verify the supplied inputs and run two epochs of training and evaluation on Cora:

```bash
python scripts/verify_data.py
python main.py --dataset cora --method CHROMA --device cpu --smoke
```

### Cora


```bash
python main.py --dataset cora --backbone gcn --method USP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

The same commands support `citeseer`, `pubmed`, `amazon-computer`, `amazon-photo`, `coauthor-cs`,
`coauthor-physics`, and `wikics`. Use `--backbone gat` for GAT.
Datasets are downloaded to `data/`; use `--data-dir PATH` to reuse existing data.

### ogbn-arxiv

Construct the CHROMA partition before training:

```bash
python prepare_groups.py --dataset ogbn-arxiv
python main.py --dataset ogbn-arxiv --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset ogbn-arxiv --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

### Main table

The main comparison uses seeds 0–4 and noise standard deviations 0.5, 1.0, 1.5,
and 2.0. Use `--method CLEAN --sigma 0` for the zero-noise baseline. See
[experiment settings](docs/experiments.md) for the data splits and evaluation protocol.

```bash
# List the training commands for the full comparison.
python scripts/make_main_grid.py --output main_grid.csv
# Summarize the supplied per-seed results.
python summarize.py --input results/main_table_per_seed.csv --output results/reference_summary.csv
```

Each training run writes its log, checkpoint, and metrics to `runs/`. To
summarize your own completed runs, use `--input runs --output results/my_summary.csv`.
See [reference results](results/README.md) for the main comparison and controls.

## Citation

```bibtex
@inproceedings{guo2025grasp,
  author    = {Guo, Zhiyu and Liu, Yang and Ao, Xiang and He, Qing},
  title     = {{GRASP}: Differentially Private Graph Reconstruction Defense with Structured Perturbation},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2025},
  pages     = {767--777},
  doi       = {10.1145/3711896.3736992}
}
```

## Acknowledgements

CHROMA builds on [GRASP (Guo et al., KDD 2025)](https://doi.org/10.1145/3711896.3736992)
and its [original implementation](https://github.com/ZhiyuZone/GRASP). The code
uses [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pyg.org/),
[OGB](https://ogb.stanford.edu/), and [NetworkX](https://networkx.org/).
The training logger is adapted from the [OGB repository](https://github.com/snap-stanford/ogb).
