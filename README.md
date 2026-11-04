# CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation

[中文说明](GRASP/docs/README_CN.md)

## Overview

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

The [grouping code](GRASP/coloring/) builds the partition.
[model.py](GRASP/model.py) applies the perturbation to node embeddings.

Original GRASP: [Paper](https://doi.org/10.1145/3711896.3736992) ·
[Code](https://github.com/ZhiyuZone/GRASP)

## Installation

The code uses **Python 3.11**, **PyTorch 2.5.1**, and **PyTorch Geometric 2.6.1**.
All dependencies are listed in [requirements.txt](GRASP/requirements.txt).
The following commands install the CUDA 12.4 environment:

```bash
git clone https://github.com/ICT-FinD-Lab/CHROMA.git
cd CHROMA/GRASP
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

On Windows, replace the activation command with `.\.venv\Scripts\Activate.ps1`.
For other CUDA versions, use the corresponding PyTorch and PyG packages.

## Running the Code

Run the following commands from the `GRASP/` directory. Dataset-specific model
and training parameters are loaded from [configs/main_table.json](GRASP/configs/main_table.json).

For a two-epoch CPU check with the supplied Cora partition, run

```bash
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

Construct the CHROMA partition before training.

```bash
python prepare_groups.py --dataset ogbn-arxiv
python main.py --dataset ogbn-arxiv --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset ogbn-arxiv --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

### Experiment settings and results

The main comparison uses seeds 0–4 and noise standard deviations 0.5, 1.0, 1.5,
and 2.0. For the zero-noise baseline, use `--method CLEAN --sigma 0`.
Detailed settings are in [GRASP/docs/experiments.md](GRASP/docs/experiments.md).

To list all main-table commands, run `python scripts/make_main_grid.py --output main_grid.csv`.
Each training command saves its log, checkpoint, and measurements to `runs/`. To summarize completed runs:

```bash
python summarize.py --input runs --output results/my_summary.csv
```

[Reference results](GRASP/results/README.md) and
[code documentation](GRASP/docs/code.md) are included in `GRASP/`.
Source attribution is listed in [NOTICE.md](GRASP/docs/NOTICE.md).
