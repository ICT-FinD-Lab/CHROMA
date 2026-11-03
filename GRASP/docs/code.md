# Code structure

All paths below are relative to the `GRASP/` directory.

| Component | Implementation |
| --- | --- |
| Experiment entry point | `main.py` |
| Dataset/backbone settings | `configs/main_table.json` |
| TGRASP settings | `configs/tgrasp.json` |
| GNN and noise injection | `model.py` |
| Data loading and splits | `dataset.py`, `data_utils.py` |
| Training and checkpoints | `train.py`, `logger.py` |
| Initial coloring and group splitting | `coloring/initialization.py` |
| Constrained merging and knee selection | `coloring/merging.py` |
| Partition statistics | `coloring/metrics.py` |
| Similarity-based reconstruction | `GraphAttacker.py` |
| Exact blockwise AUC | `exact_auc.py` |
| Run logging and evaluation dispatch | `scripts/run_entry.py` |
| Validation pairs, embedding diagnostics, and checkpoint export | `diagnostics/` |
| Result aggregation | `summarize.py` |

Start experiments with `python main.py` from `GRASP/`. It loads the configuration,
prepares the partition path and output directory, and dispatches the training
loop in `train.py`.

Each run directory contains:

```text
launch.json          Resolved configuration and code/input hashes
train.log            Training output
epochs.jsonl         Per-epoch metrics
actual_split.npz     Training, validation, and test indices
models/              Selected checkpoint
result.json          Final metrics and selected epoch
exit.json            Process exit status
```

Data checksums are listed in `assets/manifest.json`. The verifier checks the
supplied partitions, splits, configurations, and reference measurements.

To check the implementation and supplied data:

```bash
python scripts/verify_data.py
python -m unittest discover -s tests -v
```
