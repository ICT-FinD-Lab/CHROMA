# Reference results

| File | Runs | Contents |
| --- | ---: | --- |
| `main_table_per_seed.csv` | 1,170 | USP, GRASP, TGRASP, and zero-noise baselines on nine datasets |
| `controls_per_seed.csv` | 640 | Default TGRASP configuration and size-matched random grouping on eight datasets |

The main comparison includes 1,040 runs on the eight smaller datasets and
130 on ogbn-arxiv. Every reported cell contains five seeds.

| Column | Meaning |
| --- | --- |
| `dataset`, `backbone`, `method` | Experiment setting |
| `sigma` | Gaussian standard deviation |
| `seed` | Run seed |
| `raw_auc` | Reconstruction AUC |
| `auc_sep` | `max(raw_auc, 1 - raw_auc)` |
| `accuracy` | Test classification accuracy at the selected checkpoint |
| `campaign`, `source_record` | Source experiment and record identifier |

Metrics in these files are fractions in [0, 1]. To produce a percentage summary:

```bash
python summarize.py --input results/main_table_per_seed.csv --output results/reference_summary.csv
```

The summary applies the AUC orientation correction per seed before averaging.
It reports each cell's seed count, mean, and population standard deviation.
The reference measurements are checksum-verified by `scripts/verify_data.py`.
