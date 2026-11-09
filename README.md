# CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-red.svg" alt="PyTorch 2.5.1">
  <img src="https://img.shields.io/badge/PyG-2.6.1-orange.svg" alt="PyG 2.6.1">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
</p>

[中文说明](docs/README_CN.md)

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

- Grouping and partition logic: [coloring/](coloring/)
- Perturbation and model injection: [model.py](model.py)
- Architecture details: [docs/code.md](docs/code.md)
- Original GRASP: [Paper](https://doi.org/10.1145/3711896.3736992) · [Code](https://github.com/ZhiyuZone/GRASP)

## Repository Structure

```text
CHROMA/
├── assets/                  # Precomputed group partitions, split indices, and checksum manifest
├── coloring/                # Core CHROMA implementation: initialization, merging, and metrics
├── configs/                 # Hyperparameters (chroma.json) and experiment profiles (main_table.json)
├── diagnostics/             # Embedding diagnostics, checkpoint trackers, and partition validation
├── docs/                    # Detailed technical docs (code.md, experiments.md, README_CN.md, NOTICE.md)
├── results/                 # Per-seed benchmark metrics and reference results
├── scripts/                 # Utility scripts (verify_data.py, make_main_grid.py, run_entry.py)
├── tests/                   # Automated unit and consistency tests
├── main.py                  # Main experiment entry point
├── train.py                 # Training loop and evaluation execution
├── model.py                 # GNN model definitions and noise injection
├── requirements.txt         # Runtime Python dependencies
└── LICENSE                  # MIT License
```

## Installation

The code uses **Python 3.11**, **PyTorch 2.5.1**, and **PyTorch Geometric 2.6.1**.
All dependencies are listed in [requirements.txt](requirements.txt).
The following commands install the CUDA 12.4 environment:

```bash
git clone https://github.com/ICT-FinD-Lab/CHROMA.git
cd CHROMA
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

On Windows, replace the activation command with `.\.venv\Scripts\Activate.ps1`.
For other CUDA versions, use the corresponding PyTorch and PyG wheels.

## Running the Code

Run all commands from the repository root directory. Dataset-specific model
and training parameters are loaded from [configs/main_table.json](configs/main_table.json).

### Smoke Test

For a two-epoch CPU check with the supplied Cora partition:

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

Construct the CHROMA partition before training:

```bash
python prepare_groups.py --dataset ogbn-arxiv
python main.py --dataset ogbn-arxiv --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset ogbn-arxiv --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

### Experiment Settings and Results

The main comparison uses seeds 0–4 and noise standard deviations 0.5, 1.0, 1.5,
and 2.0. For the zero-noise baseline, use `--method CLEAN --sigma 0`.
Detailed settings are in [docs/experiments.md](docs/experiments.md).

To list all main-table commands, run:

```bash
python scripts/make_main_grid.py --output main_grid.csv
```

Each training command saves its log, checkpoint, and measurements to `runs/`. To summarize completed runs:

```bash
python summarize.py --input runs --output results/my_summary.csv
```

Reference benchmark records are detailed in [results/README.md](results/README.md).

### Verification & Tests

To verify experiment assets, manifest checksums, and run consistency tests:

```bash
python scripts/verify_data.py
python -m unittest discover -s tests -v
```

## Contributors & Maintainers

This work is maintained by the FinD Lab, [Institute of Computing Technology, Chinese Academy of Sciences (ICT, CAS)](http://iip.ict.ac.cn/).

- **Tong Deng** ([@dt-xde](https://github.com/dt-xde))

Source attribution and third-party dependencies are acknowledged in [docs/NOTICE.md](docs/NOTICE.md).

## Citation

```bibtex
@inproceedings{grasp2025,
  title     = {Differentially Private Graph Reconstruction Defense with Structured Perturbation},
  author    = {Zhiyu Li and others},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2025},
  doi       = {10.1145/3711896.3736992}
}
```

```bibtex
@article{chroma2026,
  title   = {CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation},
  author  = {Tong Deng and others},
  year    = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
